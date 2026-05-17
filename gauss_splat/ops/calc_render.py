# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from gauss_splat.gsplat_c import calc_render_fwd_double_clip_gsids, calc_render_bwd_var_clip_gsids

# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
class CalcRender(torch.autograd.Function):
    """transmittance"""

    @staticmethod
    def forward(
        ctx, means, conic0s, conic1s, conic2s, opacities, colors, depths, tile_coords, offsets, sorted_gs_ids
    ):
        depths_ = depths
        if depths is None:
            depths_ = colors[0:1]
        sorted_gs_ids = sorted_gs_ids.to(torch.int64)
        # tile size < 64 clip
        gs = torch.cat([
            means,
            conic0s.unsqueeze(0),
            conic1s.unsqueeze(0),
            conic2s.unsqueeze(0),
            opacities.unsqueeze(0),
            colors,
            depths_
            ], dim=0).T.contiguous()
        color, depth, last_cumsum, error, gs_clip_index, alpha_clip_index = \
            calc_render_fwd_double_clip_gsids(gs, tile_coords, offsets, sorted_gs_ids)
        
        ctx.save_for_backward(gs, depths, tile_coords, offsets, sorted_gs_ids,
            last_cumsum, error, gs_clip_index, alpha_clip_index)

        if depths is None:
            return color
        else:
            return color, depth

    @staticmethod
    def backward(
        ctx, *v_args
    ):
        v_color, v_depth = None, None
        if len(v_args) == 2:
            v_color, v_depth = v_args
        elif len(v_args) == 1:
            v_color = v_args
            v_depth = torch.zeros_like(v_color[0:1])
        else:
            raise Exception("invalid arguments")
        v_color = v_color.contiguous()
        v_depth = v_depth.contiguous()

        gs, depths, tile_coords, offsets, sorted_gs_ids, last_cumsum, error, gs_clip_index, \
            alpha_clip_index = ctx.saved_tensors

        v_gs = calc_render_bwd_var_clip_gsids(
            v_color, v_depth, last_cumsum, error,
            gs,
            tile_coords, offsets,
            sorted_gs_ids,
            gs_clip_index,
            alpha_clip_index
        )
        v_means = v_gs[:, :2].T         # shape: (2, N)
        v_conic0s = v_gs[:, 2:3].T        # shape: (1, N)
        v_conic1s = v_gs[:, 3:4].T        # shape: (1, N)
        v_conic2s = v_gs[:, 4:5].T        # shape: (1, N)
        v_opacities = v_gs[:, 5:6].T        # shape: (1, N)
        v_colors = v_gs[:, 6:9].T        # shape: (3, N)

        if depths is None:
            return v_means, v_conic0s, v_conic1s, v_conic2s, v_opacities, v_colors, None, None, None, None
        else:
            v_depths = v_gs[:, 9:].T
            return v_means, v_conic0s, v_conic1s, v_conic2s, v_opacities, v_colors, v_depths, None, None, None


calc_render = CalcRender.apply