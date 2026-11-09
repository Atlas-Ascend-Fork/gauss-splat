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

def register_sh_hook(param, start_idx=1, scale=20.0, is_channel_first=False):
    """为 sh 或 features 的高阶项注册梯度缩放 hook（支持通道优先布局）"""
    def grad_hook(grad):
        grad_scaled = grad.clone()
        if is_channel_first:
            grad_scaled[start_idx:, :, :] /= scale   # [K,3,N]
        else:
            grad_scaled[:, start_idx:, :] /= scale   # [N,K,3]
        return grad_scaled
    param.register_hook(grad_hook)
    return param


def _state_get(optimizer, param):
    """Identity-based state lookup (avoids Tensor.__eq__ bool ambiguity)."""
    for k, v in optimizer.state.items():
        if k is param:
            return v
    return None


def _state_del(optimizer, param):
    """Identity-based state deletion."""
    for k in list(optimizer.state.keys()):
        if k is param:
            del optimizer.state[k]
            return


def _state_set(optimizer, new_param, stored_state):
    """Set optimizer state for new_param."""
    optimizer.state[new_param] = stored_state


class CBSADCStrategy:
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_start_iter: int = 3_000
    refine_stop_iter: int = 5_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    # (4) hard cap
    cap_max: int = 600_000
    key_for_gradient: Literal["means2d"] = "means2d"

    def step_pre_backward(self, params, optimizers, state, step, info):
        assert self.key_for_gradient in info, "means2d is required but missing."
        if step > self.refine_start_iter and step < self.refine_stop_iter:
            info[self.key_for_gradient].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info):
        if step > self.refine_start_iter and step < self.refine_stop_iter:
            # Keep track of max radii in image-space for pruning.
            # Normalize by image resolution so prune_scale2d (0.15) is a
            # normalized threshold, matching ADC/DefaultStrategy semantics.
            visibility_filter = info["gaussian_ids"]
            radii = info["radii"].max(dim=-1).values
            radii = radii / float(max(info["width"], info["height"]))
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(info["means2d"], visibility_filter)

            if step > self.refine_start_iter and step % self.refine_every == 0:
                #gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                num_gs_densitified = int(max(1, min(self._get_n(params["means"], "means") * 0.1, self.cap_max - self._get_n(params["means"], "means"))))
                self.densify_and_prune2(optimizers, params, state, num_gs_densitified, 0.005, torch.exp)

    def initialize_state(self, percent_dense, scaling_inverse_activation, total_pts, scene_scale, device="npu"):
        self.percent_dense = percent_dense
        self.scaling_inverse_activation = scaling_inverse_activation
        self.device = device
        self.xyz_gradient_accum = torch.zeros((total_pts, 1), device=self.device)
        self.denom = torch.zeros((total_pts, 1), device=self.device)
        self.max_radii2D = torch.zeros((total_pts), device=self.device)
        self.scene_scale = scene_scale

    # Parameters with channel-first [D, N] layout (N on dim=1): means, quats, scales
    # Parameters with SH channel-first [K, 3, N] layout (N on dim=2): sh
    # Parameters with N-first [N, ...] layout: opacities, features_sh
    _CHANNEL_FIRST_PARAMS = frozenset()
    _SH_PARAMS = frozenset()

    def set_channel_first(self, is_channel_first: bool):
        """Set layout mode: channel-first [D,N]/[K,3,N] or N-first [N,D]/[N,K,3]."""
        if is_channel_first:
            self._CHANNEL_FIRST_PARAMS = frozenset({"means", "quats", "scales"})
            self._SH_PARAMS = frozenset({"sh"})
        else:
            self._CHANNEL_FIRST_PARAMS = frozenset()
            self._SH_PARAMS = frozenset()

    def _register_color_hook(self, name, param):
        """Re-register the high-order /20 hook with the correct layout axis.

        - sh: layout follows _SH_PARAMS
        - features_sh: always N-first [N, K, 3]
        - features: 2D [N, D], no SH-style hook
        """
        if name == "sh":
            register_sh_hook(param, is_channel_first=bool(self._SH_PARAMS))
        elif name == "features_sh" and param.dim() == 3:
            register_sh_hook(param, is_channel_first=False)

    def _select(self, tensor, name, mask):
        """Select elements along N dimension, handling [D,N], [K,3,N], and [N,...] layouts."""
        if name in self._CHANNEL_FIRST_PARAMS:
            return tensor[:, mask]
        if name in self._SH_PARAMS:
            return tensor[:, :, mask]
        return tensor[mask]

    def _cat_n(self, existing, extension, name):
        """Concatenate along N dimension, handling [D,N], [K,3,N], and [N,...] layouts."""
        if name in self._CHANNEL_FIRST_PARAMS:
            return torch.cat((existing, extension), dim=1)
        if name in self._SH_PARAMS:
            return torch.cat((existing, extension), dim=2)
        return torch.cat((existing, extension), dim=0)

    def _get_n(self, tensor, name):
        """Get N (number of gaussians) from tensor, handling all layouts."""
        if name in self._CHANNEL_FIRST_PARAMS:
            return tensor.shape[1]
        if name in self._SH_PARAMS:
            return tensor.shape[2]
        return tensor.shape[0]

    def _prune_optimizer(self, optimizers, mask):
        # 这里的 optimizers 是一个字典：{"means": opt1, "scales": opt2, ...}
        optimizable_tensors = {}
        for name, optimizer in optimizers.items():
            if name in ['sigma', 'weight_background']:  # 过滤不需要裁剪的全局参数
                continue
            for group in optimizer.param_groups:
                group_name = group.get("name", name)
                if group_name in ['sigma', 'weight_background']:
                    continue
                stored_state = _state_get(optimizer, group['params'][0])

                if group_name in self._CHANNEL_FIRST_PARAMS or group_name in self._SH_PARAMS:
                    new_param_data = group["params"][0][..., mask]   # [D, N] / [K,3,N]
                else:
                    new_param_data = group["params"][0][mask]         # [N, ...]

                if stored_state is not None:
                    # Prune optimizer state along N dim (same dim as the parameter)
                    if group_name in self._CHANNEL_FIRST_PARAMS:
                        stored_state["exp_avg"] = stored_state["exp_avg"][:, mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:, mask]
                    elif group_name in self._SH_PARAMS:
                        stored_state["exp_avg"] = stored_state["exp_avg"][:, :, mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:, :, mask]
                    else:
                        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    _state_del(optimizer, group['params'][0])
                    new_param = nn.Parameter(new_param_data.requires_grad_(True))

                    # ---> 关键：裁剪后新生成的参数必须重新注册梯度 Hook <---
                    if group_name in ["sh", "features", "features_sh"]:
                        self._register_color_hook(group_name, new_param)

                    _state_set(optimizer, new_param, stored_state)
                    group["params"][0] = new_param
                    optimizable_tensors[group_name] = group["params"][0]
                else:
                    new_param = nn.Parameter(new_param_data.requires_grad_(True))

                    # ---> 关键：重新注册 Hook <---
                    if name in ["sh", "features", "features_sh"]:
                        self._register_color_hook(name, new_param)

                    group["params"][0] = new_param
                    optimizable_tensors[group_name] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, optimizer, params_dict, stats_dict, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(optimizer, valid_points_mask)

        # 映射更新到新的参数键名
        for name, tensor in optimizable_tensors.items():
            params_dict[name] = tensor

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, optimizers, tensors_dict):
        optimizable_tensors = {}
        for name, optimizer in optimizers.items():
            if name in ['sigma', 'weight_background']:
                continue

            for group in optimizer.param_groups:
                group_name = group.get("name", name)
                if group_name in ['sigma', 'weight_background'] or group_name not in tensors_dict:
                    continue
                extension_tensor = tensors_dict[group_name]
                stored_state = _state_get(optimizer, group['params'][0])

                if stored_state is not None:
                    # Cat optimizer state along N dim (dim=1 for [D,N], dim=2 for [K,3,N], dim=0 for [N,...])
                    if group_name in self._CHANNEL_FIRST_PARAMS:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=1)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=1)
                    elif group_name in self._SH_PARAMS:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=2)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=2)
                    else:
                        stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                    _state_del(optimizer, group['params'][0])
                    new_param = nn.Parameter(self._cat_n(group["params"][0], extension_tensor, group_name).requires_grad_(True))
                    # ---> 关键：如果是颜色/特征参数，重新注册 Hook <---
                    if group_name in ["sh", "features", "features_sh"]:
                        self._register_color_hook(group_name, new_param)

                    group["params"][0] = new_param
                    _state_set(optimizer, new_param, stored_state)
                    optimizable_tensors[group_name] = group["params"][0]
                else:
                    new_param = nn.Parameter(self._cat_n(group["params"][0], extension_tensor, group_name).requires_grad_(True))

                    # ---> 关键：如果是颜色/特征参数，重新注册 Hook <---
                    if group_name in ["sh", "features", "features_sh"]:
                        self._register_color_hook(group_name, new_param)

                    group["params"][0] = new_param
                    optimizable_tensors[group_name] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, optimizer, params_dict, stats_dict, new_tensors):
        optimizable_tensors = self.cat_tensors_to_optimizer(optimizer, new_tensors)

        for name, tensor in optimizable_tensors.items():
            params_dict[name] = tensor

        total_pts = self._get_n(params_dict["means"], "means")
        self.xyz_gradient_accum = torch.zeros((total_pts, 1), device=self.device)
        self.denom = torch.zeros((total_pts, 1), device=self.device)
        self.max_radii2D = torch.zeros((total_pts), device=self.device)

    def densify_and_split(self, optimizer, params_dict, stats_dict, grads, grad_threshold, scene_extent, scaling_activation, N=2):
        n_init_points = self._get_n(params_dict["means"], "means")
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        scaling_vals = scaling_activation(params_dict["scales"])
        cf = bool(self._CHANNEL_FIRST_PARAMS)
        if cf:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(scaling_vals, dim=0).values > self.percent_dense * scene_extent
            )
            sel_scales = scaling_vals[:, selected_pts_mask]  # [3, M]
            stds = sel_scales.repeat(1, N)  # [3, N*M]
            samples = torch.normal(mean=torch.zeros_like(stds), std=stds)  # [3, N*M]
            sel_quats = params_dict["quats"][:, selected_pts_mask].t().contiguous()  # [M, 4]
            rots = build_rotation(sel_quats).repeat(N, 1, 1)  # [N*M, 3, 3]
            new_xyz = torch.bmm(rots, samples.t().unsqueeze(-1)).squeeze(-1)  # [N*M, 3]
            sel_means = params_dict["means"][:, selected_pts_mask].t().contiguous()  # [M, 3]
            new_xyz = new_xyz + sel_means.repeat(N, 1)  # [N*M, 3]
            new_xyz = new_xyz.t().contiguous()  # [3, N*M]
            new_scaling = self.scaling_inverse_activation(sel_scales.repeat(1, N) / (0.8 * N))  # [3, N*M]
            sel_quats_full = params_dict["quats"][:, selected_pts_mask]  # [4, M]
            new_rotation = sel_quats_full.repeat(1, N)  # [4, N*M]
            new_features = params_dict["sh"][:, :, selected_pts_mask].repeat(1, 1, N)  # [K, 3, N*M]
        else:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(scaling_vals, dim=-1).values > self.percent_dense * scene_extent
            )
            sel_scales = scaling_vals[selected_pts_mask]  # [M, 3]
            stds = sel_scales.repeat(N, 1)  # [N*M, 3]
            samples = torch.normal(mean=torch.zeros_like(stds), std=stds)  # [N*M, 3]
            sel_quats = params_dict["quats"][selected_pts_mask].contiguous()  # [M, 4]
            rots = build_rotation(sel_quats).repeat(N, 1, 1)  # [N*M, 3, 3]
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)  # [N*M, 3]
            sel_means = params_dict["means"][selected_pts_mask].contiguous()  # [M, 3]
            new_xyz = new_xyz + sel_means.repeat(N, 1)  # [N*M, 3]
            new_xyz = new_xyz.contiguous()  # [N*M, 3]
            new_scaling = self.scaling_inverse_activation(sel_scales.repeat(N, 1) / (0.8 * N))  # [N*M, 3]
            sel_quats_full = params_dict["quats"][selected_pts_mask]  # [M, 4]
            new_rotation = sel_quats_full.repeat(N, 1)  # [N*M, 4]
            new_features = params_dict["sh"][selected_pts_mask].repeat(N, 1, 1)  # [N*M, K, 3]
        new_opacity = params_dict["opacities"][selected_pts_mask].repeat(N)
        d = {
            "means": new_xyz, "sh": new_features,
            "opacities": new_opacity,
            "scales": new_scaling, "quats": new_rotation
        }
        if "features_sh" in params_dict:
            d["features_sh"] = params_dict["features_sh"][selected_pts_mask].repeat(N, 1, 1) / N

        self.densification_postfix(optimizer, params_dict, stats_dict, d)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(optimizer, params_dict, stats_dict, prune_filter)

    def densify_and_clone(self, optimizer, params_dict, stats_dict, grads, grad_threshold, scene_extent, scaling_activation):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        scaling_vals = scaling_activation(params_dict["scales"])
        cf = bool(self._CHANNEL_FIRST_PARAMS)
        if cf:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(scaling_vals, dim=0).values <= self.percent_dense * scene_extent
            )
        else:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(scaling_vals, dim=-1).values <= self.percent_dense * scene_extent
            )

        new_xyz = self._select(params_dict["means"], "means", selected_pts_mask)
        new_features = self._select(params_dict["sh"], "sh", selected_pts_mask)
        new_opacities = params_dict["opacities"][selected_pts_mask]

        if "features_sh" in params_dict:
            params_dict["features_sh"] = torch.where(
                selected_pts_mask.unsqueeze(-1).unsqueeze(-1),
                params_dict["features_sh"] / 2,
                params_dict["features_sh"]
            )

            new_features_sh = params_dict["features_sh"][selected_pts_mask]

        new_scaling = self._select(params_dict["scales"], "scales", selected_pts_mask)
        new_rotation = self._select(params_dict["quats"], "quats", selected_pts_mask)

        d = {
            "means": new_xyz, "sh": new_features,
            "opacities": new_opacities,
            "scales": new_scaling, "quats": new_rotation
        }
        if "features_sh" in params_dict:
            d["features_sh"] = new_features_sh

        self.densification_postfix(optimizer, params_dict, stats_dict, d)

    @torch.no_grad()
    def densify_and_prune2(self, optimizer, params_dict, stats_dict, top_k, extent, scaling_activation):
        if top_k <= 0:
            self._prune_gs(params_dict, optimizer, stats_dict)
        else:
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
            self._prune_gs(params_dict, optimizer, stats_dict)


            torch.npu.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state) -> int:
        is_prune = self.max_radii2D > self.prune_scale2d

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            # 2. 调用已有的裁剪逻辑
            self.prune_points(optimizers, params, None, is_prune)
        return n_prune