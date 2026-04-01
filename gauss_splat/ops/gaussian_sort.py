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

import gauss_splat._C
class GaussianSort(Function):
    @staticmethod
    def forward(ctx, all_in_mask: torch.Tensor, depths: torch.Tensor):
        sorted_gs_ids, tile_offsets = gauss_splat._C.gaussian_sort(all_in_mask, depths)
        return sorted_gs_ids, tile_offsets


gaussian_sort = GaussianSort.apply
