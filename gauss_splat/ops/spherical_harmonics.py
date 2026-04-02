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
    spherical_harmonics_forward,
    spherical_harmonics_bwd,
)
class SphericalHarmonicsForward(Function):
    @staticmethod
    def forward(ctx,
                degrees_to_use: int,
                dirs: torch.Tensor,
                coeffs: torch.Tensor):
        if degrees_to_use > 4 or degrees_to_use < 0:
            raise ValueError("Spherical harmonics order should be 0 ~ 4, but got degrees which is not supported.")
        ctx.save_for_backward(dirs, coeffs)
        ctx.degree = degrees_to_use
        output = spherical_harmonics_forward(
                dirs,
                coeffs,
                degrees_to_use
        )
        return output

    @staticmethod
    def backward(ctx, *args):
        v_colors = args[0]
        dirs, coeffs = ctx.saved_tensors
        degree = ctx.degree
        v_dirs, v_coeffs = spherical_harmonics_bwd(
                dirs,
                coeffs,
                v_colors,
                degree
        )
        return None, v_dirs, v_coeffs

spherical_harmonics = SphericalHarmonicsForward.apply

