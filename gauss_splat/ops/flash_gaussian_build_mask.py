# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing_extensions import Literal

import torch
import torch_npu
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from gauss_splat.gsplat_c import flash_gaussian_build_mask as _flash_gaussian_build_mask_cpp

class FlashGaussianBuildMask(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx,
                means2d: torch.Tensor,
                opacity: torch.Tensor,
                conics: torch.Tensor,
                covars2d: torch.Tensor,
                depths: torch.Tensor,
                cnt: torch.Tensor,
                tile_grid: torch.Tensor,
                image_width,
                image_height,
                tile_size=64):

        if opacity is None:
            raise ValueError("Opacity must be Tensor while using FlashGS.")

        tile_sum, tile_offset, tile_depths, gauss_index = _flash_gaussian_build_mask_cpp(
            means2d,
            opacity,
            conics,
            covars2d,
            depths,
            cnt,
            tile_grid,
            float(image_width),
            float(image_height),
            tile_size
        )
        return tile_sum, tile_offset, tile_depths, gauss_index

flash_gaussian_build_mask = FlashGaussianBuildMask.apply