# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from gauss_splat.gsplat_c import gaussian_filter as _gaussian_filter

# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def gaussian_filter(means,
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
            support_features : int = 0):
    B = det.shape[0]
    C = det.shape[1]
    N = det.shape[2]
    color_dims = 7 if support_features & (1 << 2) else colors.shape[1]
    proj_filter_packed_dim = (N + 7) // 8
    dtype = means.dtype
    device = means.device

    means_culling = torch.empty((B, C, 3, N), dtype=dtype, device=device)
    colors_culling = torch.empty((B, C, color_dims, N), dtype=dtype, device=device)
    quats_culling = torch.empty((B, C, 4, N), dtype=dtype, device=device)
    scales_culling = torch.empty((B, C, 3, N), dtype=dtype, device=device)
    means2d_culling = torch.empty((B, C, 2, N), dtype=dtype, device=device)
    depths_culling = torch.empty((B, C, N), dtype=dtype, device=device)
    radius_culling = torch.empty((B, C, 2, N), dtype=dtype, device=device)
    covars2d_culling = torch.empty((B, C, 3, N), dtype=dtype, device=device)
    conics_culling = torch.empty((B, C, 3, N), dtype=dtype, device=device)
    opacities_culling = torch.empty((B, C, N), dtype=dtype, device=device)
    vi_culling = torch.empty((B, C, N), dtype=dtype, device=device)
    compensations_culling = torch.empty((B, C, N), dtype=dtype, device=device)
    proj_filter = torch.empty((B, C, proj_filter_packed_dim), dtype=torch.uint8, device=device)
    filter_float = torch.zeros((B, C, N), dtype=dtype, device=device)
    cnt_float = torch.zeros((B, C), dtype=dtype, device=device)
    dense_indices = torch.empty((B, C, N), dtype=torch.int32, device=device)

    outputs = [means_culling, colors_culling, means2d_culling, depths_culling, radius_culling, covars2d_culling, conics_culling,
        opacities_culling, quats_culling, scales_culling, vi_culling, compensations_culling, proj_filter, cnt_float, filter_float, dense_indices]
    means_culling, colors_culling, means2d_culling, depths_culling, radius_culling, covars2d_culling, conics_culling, \
        opacities_culling, quats_culling, scales_culling, vi_culling, compensations_culling, proj_filter, cnt_float, filter_float, dense_indices = \
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
    return means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
           covars2d_culling, conics_culling, opacities_culling, quats_culling, scales_culling, vi_culling, compensations_culling, \
            proj_filter, cnt, filter_float, dense_indices
