# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/default.py
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from typing_extensions import Literal


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='npu')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class CBSADCStrategy:
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_start_iter: int = 8_500
    refine_stop_iter: int = 20_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    # (4) hard cap
    cap_max: int = 6_000_000
    key_for_gradient: Literal["means2d"] = "means2d"

    def step_pre_backward(self, params, optimizers, state, step, info):
        assert self.key_for_gradient in info, "means2d is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info):
        if step < self.refine_stop_iter:
            # Keep track of max radii in image-space for pruning
            visibility_filter = info["gaussian_ids"]
            radii = info["radii"].max(dim=-1).values
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(info["means2d"], visibility_filter)

            if step > self.refine_start_iter and step % self.refine_every == 0:
                #gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                num_gs_densitified = int(max(1, min(params["means"].shape[0] * 0.1, self.cap_max - params["means"].shape[0])))
                self.densify_and_prune2(optimizers, params, state, num_gs_densitified, 0.005, torch.exp)

    def initialize_state(self, percent_dense, scaling_inverse_activation, total_pts, scene_scale, device="npu"):
        self.percent_dense = percent_dense
        self.scaling_inverse_activation = scaling_inverse_activation
        self.device = device
        self.xyz_gradient_accum = torch.zeros((total_pts, 1), device=self.device)
        self.denom = torch.zeros((total_pts, 1), device=self.device)
        self.max_radii2D = torch.zeros((total_pts), device=self.device)
        self.scene_scale = scene_scale

    def _prune_optimizer(self, optimizers, mask):
        # 这里的 optimizers 是一个字典：{"means": opt1, "scales": opt2, ...}
        optimizable_tensors = {}
        for name, optimizer in optimizers.items():
            if name in ['sigma']:  # 过滤不需要裁剪的全局参数
                continue
            for group in optimizer.param_groups:
                stored_state = optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[name] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, optimizer, params_dict, stats_dict, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(optimizer, valid_points_mask)

        # 映射更新到新的参数键名
        params_dict["means"] = optimizable_tensors["means"]
        params_dict["sh0"] = optimizable_tensors["sh0"]
        params_dict["shN"] = optimizable_tensors["shN"]
        params_dict["opacities"] = optimizable_tensors["opacities"]
        params_dict["features_do"] = optimizable_tensors["features_do"]
        params_dict["features_ro"] = optimizable_tensors["features_ro"]
        params_dict["scales"] = optimizable_tensors["scales"]
        params_dict["quats"] = optimizable_tensors["quats"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, optimizers, tensors_dict):
        optimizable_tensors = {}
        for name, optimizer in optimizers.items():
            if name in ['sigma']:
                continue
            group = optimizer.param_groups[0]
            extension_tensor = tensors_dict[name]
            stored_state = optimizer.state.get(group['params'][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[name] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, optimizer, params_dict, stats_dict, new_tensors):
        optimizable_tensors = self.cat_tensors_to_optimizer(optimizer, new_tensors)

        params_dict["means"] = optimizable_tensors["means"]
        params_dict["sh0"] = optimizable_tensors["sh0"]
        params_dict["shN"] = optimizable_tensors["shN"]
        params_dict["opacities"] = optimizable_tensors["opacities"]
        params_dict["features_do"] = optimizable_tensors["features_do"]
        params_dict["features_ro"] = optimizable_tensors["features_ro"]
        params_dict["scales"] = optimizable_tensors["scales"]
        params_dict["quats"] = optimizable_tensors["quats"]

        total_pts = params_dict["means"].shape[0]
        self.xyz_gradient_accum = torch.zeros((total_pts, 1), device=self.device)
        self.denom = torch.zeros((total_pts, 1), device=self.device)
        self.max_radii2D = torch.zeros((total_pts), device=self.device)

    def densify_and_split(self, optimizer, params_dict, stats_dict, grads, grad_threshold, scene_extent, scaling_activation, N=2):
        n_init_points = params_dict["means"].shape[0]
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        scaling_vals = scaling_activation(params_dict["scales"])
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling_vals, dim=1).values > self.percent_dense * scene_extent
        )

        stds = scaling_vals[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(params_dict["quats"][selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + params_dict["means"][selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(scaling_vals[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        new_rotation = params_dict["quats"][selected_pts_mask].repeat(N, 1)
        new_features_dc = params_dict["sh0"][selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = params_dict["shN"][selected_pts_mask].repeat(N, 1, 1)
        new_opacity = params_dict["opacities"][selected_pts_mask].repeat(N)
        new_features_do = params_dict["features_do"][selected_pts_mask].repeat(N, 1, 1) / N
        new_features_ro = params_dict["features_ro"][selected_pts_mask].repeat(N, 1, 1) / N

        d = {
            "means": new_xyz, "sh0": new_features_dc, "shN": new_features_rest,
            "opacities": new_opacity, "features_do": new_features_do, "features_ro": new_features_ro,
            "scales": new_scaling, "quats": new_rotation
        }

        self.densification_postfix(optimizer, params_dict, stats_dict, d)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(optimizer, params_dict, stats_dict, prune_filter)

    def densify_and_clone(self, optimizer, params_dict, stats_dict, grads, grad_threshold, scene_extent, scaling_activation):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        scaling_vals = scaling_activation(params_dict["scales"])
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(scaling_vals, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = params_dict["means"][selected_pts_mask]
        new_features_dc = params_dict["sh0"][selected_pts_mask]
        new_features_rest = params_dict["shN"][selected_pts_mask]
        new_opacities = params_dict["opacities"][selected_pts_mask]

        # params_dict["features_do"][selected_pts_mask] /= 2
        # params_dict["features_ro"][selected_pts_mask] /= 2
        params_dict["features_do"] = torch.where(
            selected_pts_mask.unsqueeze(-1).unsqueeze(-1),
            params_dict["features_do"] / 2,
            params_dict["features_do"]
        )
        params_dict["features_ro"] = torch.where(
            selected_pts_mask.unsqueeze(-1).unsqueeze(-1),
            params_dict["features_ro"] / 2,
            params_dict["features_ro"]
        )
        new_features_do = params_dict["features_do"][selected_pts_mask]
        new_features_ro = params_dict["features_ro"][selected_pts_mask]
        new_scaling = params_dict["scales"][selected_pts_mask]
        new_rotation = params_dict["quats"][selected_pts_mask]

        d = {
            "means": new_xyz, "sh0": new_features_dc, "shN": new_features_rest,
            "opacities": new_opacities, "features_do": new_features_do, "features_ro": new_features_ro,
            "scales": new_scaling, "quats": new_rotation
        }

        self.densification_postfix(optimizer, params_dict, stats_dict, d)

    def densify_and_prune2(self, optimizer, params_dict, stats_dict, top_k, extent, scaling_activation):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grad_norms = torch.norm(grads, dim=-1)
        n_points = grad_norms.shape[0]

        if top_k >= n_points:
            threshold_grad = 0
        else:
            threshold_grad, _ = torch.topk(grad_norms, top_k)
            threshold_grad = threshold_grad[-1].item()

        self.densify_and_clone(optimizer, params_dict, stats_dict, grads, threshold_grad, extent, scaling_activation)
        self.densify_and_split(optimizer, params_dict, stats_dict, grads, threshold_grad, extent, scaling_activation)

        torch.npu.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1