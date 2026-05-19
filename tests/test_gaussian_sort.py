# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import namedtuple
import gc

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import acl

from gauss_splat import gaussian_sort, get_render_schedule_cpp

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

ExecResults = namedtuple('ExecResults', ['sorted_gs_ids'])
Inputs = namedtuple(
    'Inputs', ['lb_scheds', 'tile_sums', 'tile_depths', 'tile_gaussian_ids', 'sorted_offset', 'max_tile_gauss']
)


def _gaussian_sort(lb_scheds, tile_sums, tile_depths, tile_gaussian_ids, sorted_offset):
    B, C, T = tile_sums.shape
    sorted_gs_ids = torch.zeros(int(sorted_offset[-1].item()), dtype=torch.int64)
    vector_num = acl.get_device_capability(0, 1)[0]
    num_bins = min(vector_num, T)
    for b in range(B):
        for c in range(C):
            base_id = b * C + c
            base_offset = int(sorted_offset[base_id - 1].item()) if base_id > 0 else 0
            scheduled_tile_ids = lb_scheds[b, c, num_bins:num_bins + T]
            tile_offsets = lb_scheds[b, c, num_bins + T:]
            for sched_idx in range(T):
                tile_id = int(scheduled_tile_ids[sched_idx].item())
                sorted_tile_offset = int(tile_offsets[tile_id - 1].item()) if tile_id > 0 else 0
                tile_gauss_num = int(tile_sums[b, c, tile_id].item())
                if tile_gauss_num == 0:
                    continue
                depths_t = tile_depths[b, c, tile_id, :tile_gauss_num]
                gauss_ids_t = tile_gaussian_ids[b, c, tile_id, :tile_gauss_num]
                sort_idx = torch.argsort(depths_t, stable=True)
                write_start = base_offset + sorted_tile_offset
                write_end = write_start + tile_gauss_num
                sorted_gs_ids[write_start:write_end] = gauss_ids_t[sort_idx]
    return sorted_gs_ids


class TestGaussianSort(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.camera_num = 3
        self.test_cases = [[2, 8], [2, 123], [10, 1234], [112, 23456], [222, 234567]]

    def gen_inputs(self, tile_num, gaussian_num):
        B, C, T, G = self.batch_size, self.camera_num, tile_num, gaussian_num
        mask = torch.rand(B, C, T, G) > 0.5
        tile_sums = mask.sum(dim=-1).to(torch.int64)
        tile_offsets = tile_sums.cumsum(dim=-1)
        sorted_cnts = tile_offsets[:, :, -1].flatten()
        sorted_offset = torch.cumsum(sorted_cnts, dim=0)

        gaussian_ids = torch.arange(G).view(1, 1, 1, G).expand(B, C, T, G)
        depths = torch.rand(B, C, T, G)

        tile_gaussian_ids = torch.zeros(B, C, T, G)
        tile_depths = torch.zeros(B, C, T, G)

        for b in range(B):
            for c in range(C):
                for t in range(T):
                    mask_t = mask[b, c, t]
                    if mask_t.any():
                        tile_gaussian_ids[b, c, t, : mask_t.sum()] = gaussian_ids[b, c, t, mask_t]
                        tile_depths[b, c, t, : mask_t.sum()] = depths[b, c, t, mask_t]
        vector_num = acl.get_device_capability(0, 1)[0]
        num_bins = min(vector_num, T)
        lb_scheds = get_render_schedule_cpp(tile_sums.cpu().to(torch.int64), num_bins)
        max_tile_gauss = tile_sums.max().item()

        return Inputs(lb_scheds, tile_sums, tile_depths, tile_gaussian_ids, sorted_offset, max_tile_gauss)

    def cpu_to_exec(self, inputs):
        sorted_gs_ids = _gaussian_sort(
            inputs.lb_scheds, inputs.tile_sums, inputs.tile_depths, inputs.tile_gaussian_ids, inputs.sorted_offset
        )
        return ExecResults(sorted_gs_ids=sorted_gs_ids.detach().to(torch.int64))

    def npu_to_exec(self, inputs):
        sorted_gs_ids = gaussian_sort(
            inputs.lb_scheds.npu(),
            inputs.tile_sums.npu().to(torch.int32),
            inputs.tile_depths.npu(),
            inputs.tile_gaussian_ids.npu(),
            inputs.sorted_offset.npu(),
            int(inputs.max_tile_gauss),
        )
        return ExecResults(sorted_gs_ids=sorted_gs_ids.detach().to(torch.int64))

    def test_gaussian_sort(self):
        for tile_num, gaussian_num in self.test_cases:
            with self.subTest(tile_num=tile_num, gaussian_num=gaussian_num):
                inputs = self.gen_inputs(tile_num, gaussian_num)
                cpu_results = self.cpu_to_exec(inputs)
                npu_results = self.npu_to_exec(inputs)
                self.assertRtolEqual(cpu_results.sorted_gs_ids.numpy(), npu_results.sorted_gs_ids.cpu().numpy())
                del inputs, cpu_results, npu_results
                gc.collect()
                torch.npu.empty_cache()


if __name__ == "__main__":
    run_tests()