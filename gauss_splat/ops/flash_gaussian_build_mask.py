# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing_extensions import Literal

import torch
import torch_npu
from torch.autograd import Function
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F

import gauss_splat._C

class FlashGaussianBuildMask(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx,
                means2d: torch.Tensor,
                opacity: torch.Tensor,
                conics: torch.Tensor,
                covars2d: torch.Tensor,
                cnt: torch.Tensor,
                tile_grid: torch.Tensor,
                image_width,
                image_height,
                tile_size=64):

        if opacity is None:
            raise ValueError("Opacity must be Tensor while using FlashGS.")

        mask = gauss_splat._C.flash_gaussian_build_mask(
            means2d,
            opacity,
            conics,
            covars2d,
            cnt,
            tile_grid,
            float(image_width),
            float(image_height),
            tile_size
        )
        return mask

flash_gaussian_build_mask = FlashGaussianBuildMask.apply