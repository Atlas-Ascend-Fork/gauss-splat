# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import namedtuple

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import gaussian_sort

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

ExecResults = namedtuple('ExecResults', ['sorted_gs_ids', 'tile_offsets'])
Inputs = namedtuple('Inputs', ['all_in_mask', 'depths'])


def _gaussian_sort(all_in_mask, depths):
    # tile offset
    tile_offsets = torch.sum(all_in_mask, dim=0).cumsum(dim=0)
    sorted_gs_ids = torch.zeros(tile_offsets[-1], dtype=torch.int32, device=all_in_mask.device)
    tile_num = all_in_mask.shape[1]
    for tile_id in range(tile_num):
        prev_offset = tile_offsets[tile_id - 1] if tile_id > 0 else 0
        tile_in_mask = all_in_mask[:, tile_id]
        tile_depths = depths[tile_in_mask]
        tile_gs_ids = tile_in_mask.nonzero()[:, 0]
        _, local_sort_index = torch.sort(tile_depths, stable=True)
        sorted_gs_ids[prev_offset:tile_offsets[tile_id]] = tile_gs_ids[local_sort_index]
    return sorted_gs_ids, tile_offsets


class TestGaussianSort(TestCase):
    def setUp(self):
        self.test_cases = [[1, 123], [10, 1234], [112, 23456], [222, 234567]]
        self.test_results = self.gen_results()

    def gen_inputs(self, shape):
        tile_num, gaussian_num = shape
        all_in_mask = torch.randint(0, 2, size=(tile_num, gaussian_num), dtype=torch.int32).float()
        depths = torch.rand(gaussian_num).float()
        return Inputs(all_in_mask, depths)

    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            inputs = self.gen_inputs(test_case)
            cpu_results = self.cpu_to_exec(inputs)
            npu_results = self.npu_to_exec(inputs)
            test_results.append((cpu_results, npu_results))
        return test_results

    def cpu_to_exec(self, inputs):
        all_in_mask = inputs.all_in_mask.cpu()
        depths = inputs.depths.cpu()
        sorted_gs_ids, tile_offsets = _gaussian_sort(all_in_mask.T.to(torch.bool), depths)
        return ExecResults(sorted_gs_ids=sorted_gs_ids.detach().int(), tile_offsets=tile_offsets.detach().int())

    def npu_to_exec(self, inputs):
        all_in_mask = inputs.all_in_mask.npu()
        depths = inputs.depths.npu()
        sorted_gs_ids, tile_offsets = gaussian_sort(all_in_mask, depths)
        return ExecResults(sorted_gs_ids=sorted_gs_ids.detach().int(), tile_offsets=tile_offsets.detach().int())

    def test_gaussian_sort(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.sorted_gs_ids.numpy(), npu_results.sorted_gs_ids.cpu().numpy())
            self.assertRtolEqual(cpu_results.tile_offsets.numpy(), npu_results.tile_offsets.cpu().numpy())


if __name__ == "__main__":
    run_tests()
