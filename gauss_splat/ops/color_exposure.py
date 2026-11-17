# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from gauss_splat.gsplat_c import (
    color_exposure_fwd,
    color_exposure_bwd,
)


class ColorExposure(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                colors: torch.Tensor,
                means2d: torch.Tensor,
                exposure: torch.Tensor,
                width: int,
                height: int,
                clamp: bool = True,
                calc_exp_grad: bool = True):
        if colors.dim() != 3 or means2d.dim() != 3:
            raise ValueError("colors and means2d need [B, C, N] dimensions.")
        output, index, weight = color_exposure_fwd(
            colors, means2d, exposure, width, height, clamp)
        ctx.clamp = clamp
        ctx.calc_exp_grad = calc_exp_grad
        ctx.save_for_backward(colors, exposure, index, weight)
        return output

    @staticmethod
    def backward(ctx, *v_args):
        v_colors_in = v_args[0].contiguous()
        colors, exposure, index, weight = ctx.saved_tensors
        v_colors_out, v_exposure = color_exposure_bwd(
            colors, v_colors_in, exposure, index, weight, ctx.clamp, ctx.calc_exp_grad)
        return v_colors_out, None, v_exposure if ctx.calc_exp_grad else None, \
            None, None, None, None

color_exposure = ColorExposure.apply
