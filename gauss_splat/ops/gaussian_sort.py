# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.autograd import Function

from gauss_splat.gsplat_c import gaussian_sort as _gaussian_sort

class GaussianSort(Function):
    @staticmethod
    def forward(
        ctx,
        lb_sched: torch.Tensor,
        gaussian_cnt: torch.Tensor,
        depths: torch.Tensor,
        gs_ids: torch.Tensor,
        sorted_offset: torch.Tensor,
        max_tile_gauss: int,
    ):
        sorted_gs_ids = _gaussian_sort(
            lb_sched, gaussian_cnt, depths, gs_ids, sorted_offset, max_tile_gauss
        )
        return sorted_gs_ids


gaussian_sort = GaussianSort.apply