# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import namedtuple

import acl
import torch
from render_test_utils import (
    assert_render_results, cpu_inputs, gen_render_inputs, render_with_gsids,
)
from torch_npu.testing.testcase import TestCase, run_tests
from gauss_splat import calc_render, get_render_schedule_cpp


ExecResults = namedtuple('ExecResults', ['render_colors', 'render_depths'])




class TestCalcRenderForwardDoubleClipGsids(TestCase):

    def setUp(self):
        super().setUp()
        torch.npu.set_device('npu:0')
        self.test_cases = [
            *[[n, 32, 32, 32] for n in range(1, 10)],
            [102000, 32 * 3, 32 * 4, 32],
        ]

    def gen_inputs(self, shape):
        return gen_render_inputs(shape)

    def gen_results(self):
        for test_case in self.test_cases:
            tile_size = test_case[-1]
            op_inputs = self.gen_inputs(test_case)
            cpu_results = self.cpu_to_exec(op_inputs, tile_size)
            npu_results = self.npu_to_exec(op_inputs)
            yield test_case, cpu_results, npu_results
    
    @torch.no_grad()
    def cpu_to_exec(self, op_inputs, tile_size):
        op_inputs = cpu_inputs(op_inputs)
        means2d, colors, opacities, conics_0, conics_1, conics_2, sorted_gs_ids, \
                        tile_offsets, _, depths, padded_width, padded_height = op_inputs
        img_size = (padded_width, padded_height)
        gs_attr = (means2d, conics_0, conics_1, conics_2, colors, opacities, depths)

        render_color_gt, render_depth_gt = render_with_gsids(img_size, gs_attr, tile_offsets, sorted_gs_ids, tile_size)

        return ExecResults(
            render_colors=render_color_gt.detach().float(),
            render_depths=render_depth_gt.detach().float(),
            )
    
    def npu_to_exec(self, op_inputs):
        means2d, colors, opacities, conics_0, conics_1, conics_2, sorted_gs_ids, \
            tile_offsets, pix_coords, depths, _, _ = op_inputs
        nums = torch.cat([tile_offsets[:1], tile_offsets[1:] - tile_offsets[:-1]])
        num_vector_core, ret = acl.get_device_capability(0, 1)
        lb_sched = get_render_schedule_cpp(nums.cpu().to(torch.int64), 
                                           num_vector_core).clone().detach().to(torch.int64).npu()
        cf_means2 = means2d.transpose(0, 1)
        cf_colors3 = colors.transpose(0, 1)
        cf_render_colors, cf_render_depths = calc_render(cf_means2,
                                                        conics_0, conics_1, conics_2,
                                                        opacities.squeeze(dim=-1),
                                                        cf_colors3,
                                                        depths[None, :],
                                                        pix_coords,
                                                        lb_sched,
                                                        sorted_gs_ids
                                                        )
        return ExecResults(
            render_colors=cf_render_colors.detach().float(),
            render_depths=cf_render_depths.detach().float(),
            )
    
    def test_calc_render_forward_double_clip_gsids(self):
        for test_case, cpu_results, npu_results in self.gen_results():
            assert_render_results(self, cpu_results, npu_results, test_case)

if __name__ == "__main__":
    run_tests()


