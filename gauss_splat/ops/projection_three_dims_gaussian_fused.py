# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.autograd import Function
import torch.nn.functional as F
from gauss_splat.gsplat_c import (
    projection_three_dims_gaussian_forward,
    gaussian_filter as _gaussian_filter,
    fully_fused_projection_bwd,
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
                width: int = 0,
                height: int = 0,
                eps: float = 0.3,
                near_plane: float = 0.01,
                far_plane: float = 1e10,
                calc_compensations: bool = False,
                camera_model: str = "pinhole"):

        if quat is not None:
            if scales is None:
                raise ValueError("'quat' and 'scales' are required together.")
            if covars is not None:
                raise ValueError("Invalid parameter combination: 'covars' \
                                 and ('quat', 'scales')pair are mutually exclusive.")
            quat = quat.permute((0, 2, 1)).contiguous()
            scales = scales.permute((0, 2, 1)).contiguous()
            covars = quat_scales_to_covars(quat, scales)
        else:
            covars = covars.permute((0, 2, 3, 1)).contiguous()
            
        means = means.permute((0, 2, 1)).contiguous()

        means2d, depths, conics, compensations, det, radius, covars2d = \
            projection_three_dims_gaussian_forward(
                means,
                covars,
                opacities,
                viewmats,
                ks,
                width,
                height,
                eps,
                calc_compensations,
                camera_model
        )
        
        det = det.squeeze(-2)
        depths = depths.squeeze(-2)
        if calc_compensations:
            compensations = compensations.squeeze(-2)
        else:
            compensations = None
        
        means_culling, colors_culling, means2d_culling, depths_culling, \
        radius_culling, covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt = \
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
                width,
                height,
                near_plane,
                far_plane)

        ctx.save_for_backward(means, conics, viewmats, quat, scales, ks, proj_filter, compensations)
        ctx.width = width
        ctx.height = height
        return means2d_culling, depths_culling, conics_culling, opacities_culling,\
            radius_culling, covars2d_culling, colors_culling, cnt

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def backward(
        ctx, *v_args
    ):
        means, conics, viewmats, quats, scales, ks, proj_filter, compensations = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        v_means2d, v_depths, v_conics, v_opacities_culling, v_radii, v_covar2d, v_colors_culling, v_cnt = v_args
        v_pw, v_quats, v_scales, v_r, v_colors, v_opacities = fully_fused_projection_bwd(
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
                width,
                height
        )
        return v_pw, v_colors, None, v_quats, v_scales, v_opacities, \
                None, None, None, None, None, None, None, None, None

projection_three_dims_gaussian_fused = ProjectionThreeDimsGaussianFused.apply

