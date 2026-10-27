# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.autograd import Function
import gauss_splat.gsplat_c as _C
from .rast_ctx import RastCtx
from gauss_splat.gsplat_c import (
    projection_three_dims_gaussian_forward,
    gaussian_filter as _gaussian_filter,
    fully_fused_projection_bwd,
    quat_scales_to_covars,
)
class ProjectionThreeDimsGaussianFused(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def forward(ctx,
                means: torch.Tensor,
                colors: torch.Tensor,
                covars: torch.Tensor = None,
                quat: torch.Tensor = None,
                scales: torch.Tensor = None,
                opacities: torch.Tensor = None,
                viewmats: torch.Tensor = None,
                ks: torch.Tensor = None,
                vi: torch.Tensor = None,
                sigma: torch.Tensor = None,
                rast_ctx: RastCtx = None,
                width: int = 0,
                height: int = 0,
                eps: float = 0.3,
                near_plane: float = 0.01,
                far_plane: float = 1e10,
                calc_compensations: bool = False,
                camera_model: str = "pinhole"):

        rast_ctx = RastCtx() if rast_ctx is None else rast_ctx
        rast_ctx.prealloc_if_need(means.shape[0], viewmats.shape[1], means.shape[1], means.device)
        if quat is not None:
            if scales is None:
                raise ValueError("'quat' and 'scales' are required together.")
            if covars is not None:
                raise ValueError("Invalid parameter combination: 'covars' \
                                 and ('quat', 'scales')pair are mutually exclusive.")
            quat = quat.permute((0, 2, 1)).contiguous()
            scales = scales.permute((0, 2, 1)).contiguous()
            if not rast_ctx.covars_fused:
                covars = quat_scales_to_covars(quat, scales)
        else:
            covars = covars.permute((0, 2, 3, 1)).contiguous()

        means = means.permute((0, 2, 1)).contiguous()

        proj_quat, proj_scales, proj_covars = (quat, scales, None) if covars is None else (None, None, covars)
        outputs = rast_ctx.outputs_for_proj_forward()
        means2d, depths, conics, compensations, det, radius, covars2d = \
            projection_three_dims_gaussian_forward(
                means,
                proj_covars,
                opacities,
                viewmats,
                ks,
                proj_quat,
                proj_scales,
                width,
                height,
                eps,
                calc_compensations,
                camera_model,
                outputs
        )
        
        det = det.squeeze(-2)
        depths = depths.squeeze(-2)
        if calc_compensations:
            compensations = compensations.squeeze(-2)
        else:
            compensations = torch.ones(det.shape, dtype=torch.float32, device=det.device) if rast_ctx.use_arch35 else None
        
        support_features = ((1 << 1) | int(rast_ctx.if_depth_correct)) if rast_ctx.backend == "cbs" else 0
        if rast_ctx.proj_backward_culling :
            support_features |= (1 << 2)
            if  quat is None or scales is None :
                raise RuntimeError(
                    "covars-direct path does not support foward filter (no quat/scales "
                    "to culling); use the quat/scales path for training.")

        outputs = rast_ctx.outputs_for_filter()
        means_culling, colors_culling, means2d_culling, depths_culling, radius_culling, covars2d_culling, \
        conics_culling, opacities_culling, quats_culling, scales_culling, vi_culling, compensations_culling, \
        proj_filter, cnt_float, filter_float, dense_indices = \
            _gaussian_filter(
                means,
                colors,
                det,
                opacities,
                means2d,
                depths,
                radius,
                conics,
                covars2d,
                compensations,
                quat,
                scales,
                vi,
                sigma,
                width,
                height,
                near_plane,
                far_plane,
                support_features,
                outputs
            )

        cnt = cnt_float.to(torch.int32)
        if rast_ctx.proj_backward_culling:
            colors_culling = colors_culling[..., :3, :].contiguous()
            ctx.save_for_backward(means_culling, conics_culling, viewmats, quats_culling, scales_culling, ks, 
                filter_float, compensations_culling if calc_compensations else None, opacities_culling,
                vi_culling, sigma, depths_culling, cnt, dense_indices)
        else:
            ctx.save_for_backward(means, conics, viewmats, quat, scales, ks, proj_filter, compensations, None, None,
                None, None, None, None)
        ctx.width = width
        ctx.height = height
        ctx.rast_ctx = rast_ctx
        filter = filter_float if rast_ctx.proj_backward_culling else proj_filter
        # `means2d_culling` is a long-lived buffer cached in `rast_ctx` and rewritten in
        # place on every call. Returning that same tensor object from this
        # autograd.Function makes `retain_grad()`/hooks on it effectively one-shot:
        # PyTorch attaches the hook that fills `.grad` to the first step's graph node
        # only, so from the second step on `info["means2d"].grad` silently freezes at the
        # first step's value and the densify strategy keeps acting on a stale, step-0
        # gradient. Hand autograd a fresh tensor each call so the retained gradient
        # tracks the current step.
        return means2d_culling.clone(), depths_culling, conics_culling, opacities_culling,\
            radius_culling, covars2d_culling, colors_culling, cnt, filter, vi_culling

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def backward(
        ctx, *v_args
    ):
        means, conics, viewmats, quats, scales, ks, proj_filter, compensations, opacities, vi, sigma, depths, cnt, \
        dense_indices = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means2d, v_depths, v_conics, v_opacities_culling, v_radii, v_covar2d, v_colors_culling, v_cnt, v_proj_filter, v_vi_culling = v_args
        
        B, C, N = means.shape[0], viewmats.shape[1], means.shape[-1]
        BC = B * C
        max_compact_count = 0
        cnt_flat = None

        ### cbs and depth correct features.
        support_features = 0
        if ctx.rast_ctx.backend == "cbs":
            support_features |= ((1 << 1) | int(ctx.rast_ctx.if_depth_correct))

        ### reshape the culling data from [B, C, ...] --> [BC, ...]
        if ctx.rast_ctx.proj_backward_culling:
            if quats is None or quats.numel() == 0:
                raise RuntimeError(
                    "covars-direct path does not support backward (no quat/scales "
                    "to differentiate); use the quat/scales path for training.")
            support_features |= (1 << 2)
            means = means.reshape(BC, 3, N)
            quats = quats.reshape(BC, 4, N)
            scales = scales.reshape(BC, 3, N)
            conics = conics.reshape(BC, 3, N)
            viewmats = viewmats.reshape(BC, 16)
            ks = ks.reshape(BC, 9)
            v_means2d = v_means2d.reshape(BC, 2, N)
            v_depths = v_depths.reshape(BC, N)
            v_conics = v_conics.reshape(BC, 3, N)
            v_colors_culling = v_colors_culling.reshape(BC, 3, N)
            v_opacities_culling = v_opacities_culling.reshape(BC, N)
            if v_vi_culling is None:
                v_vi_culling = torch.zeros(BC, N, dtype=torch.float32, device=means.device)
            else:
                v_vi_culling = v_vi_culling.reshape(BC, N).contiguous()
            opacities = opacities.reshape(BC, N)
            vi = None if vi is None else vi.reshape(BC, N)
            depths = depths.reshape(BC, N)
            dense_indices = dense_indices.reshape(BC, N)
            compensations = compensations.reshape(BC, N) if compensations is not None else torch.ones_like(depths)

            # Get max compact count from cnt
            cnt_flat = cnt.view(-1).int()
            max_compact_count = max(1, int(cnt.max().item()))

        ### alpha max out and grad abs features
        if ctx.rast_ctx.alpha_max is not None and ctx.rast_ctx.grad_abs is not None:
            support_features |= (1 << 3)
            alpha_max_out_stried = torch.as_strided(ctx.rast_ctx.alpha_max, size=(BC, 1, N), stride=(N, N, 1))
            v_means2d_abs_stried = torch.as_strided(ctx.rast_ctx.grad_abs, size=(BC, 2, N), stride=(N*2, N, 1))
            v_colors_culling = torch.cat((v_colors_culling, alpha_max_out_stried, v_means2d_abs_stried), dim=1)

        outputs = ctx.rast_ctx.outputs_for_proj_backward()
        [output.zero_() for output in outputs if output is not None]
        v_pw, v_quats, v_scales, v_r, v_colors, v_opacities, v_viewmats, v_ks, dL_dvi, dL_dsigma = \
            fully_fused_projection_bwd(
                means,
                quats,
                scales,
                conics,
                viewmats,
                ks,
                v_means2d,
                v_depths,
                v_conics,
                v_colors_culling,
                v_opacities_culling,
                proj_filter,
                compensations,
                v_vi_culling,
                opacities,
                vi,
                sigma,
                depths,
                dense_indices,
                cnt_flat,
                width,
                height,
                max_compact_count,
                support_features,
                outputs
        )

        ### copy alpha max and grad abs out.
        if ctx.rast_ctx.alpha_max is not None and ctx.rast_ctx.grad_abs is not None:
            ctx.rast_ctx.alpha_max.copy_(v_colors[:, 3:4, :].contiguous().view(B, C, -1))
            ctx.rast_ctx.grad_abs.copy_(v_colors[:, 4:6, :].contiguous().view(B, C, 2, -1))
        
        ### forward is [B, N, C], so backword need from [B, C, N] transpose to it.
        if ctx.rast_ctx.use_arch35:
            v_pw = v_pw.transpose(1, 2).contiguous()
            v_quats = v_quats.transpose(1, 2).contiguous()
            v_scales = v_scales.transpose(1, 2).contiguous()

        ### deflate the v_viewmats and v_ks (pose refine)
        v_viewmats = None if v_viewmats is None else v_viewmats.view(B, C, 4, 4)
        v_ks = None if v_ks is None else v_ks.view(B, C, 3, 3)

        ### vi and sigma
        dL_dvi = dL_dvi.reshape(B, C, N) if dL_dvi is not None else None
        dL_dsigma = dL_dsigma[0, 0].unsqueeze(0) if dL_dsigma is not None and ctx.rast_ctx.if_depth_correct else None

        return v_pw, v_colors, None, v_quats, v_scales, v_opacities, \
            v_viewmats, v_ks, dL_dvi, dL_dsigma, \
                None, None, None, None, None, None, None, None

projection_three_dims_gaussian_fused = ProjectionThreeDimsGaussianFused.apply
