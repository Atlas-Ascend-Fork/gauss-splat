# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from collections import namedtuple
import struct
import random
import math
from typing import Optional, Tuple
from typing_extensions import Literal, assert_never

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch import Tensor
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import flash_gaussian_build_mask

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=np.inf)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

ExecResults = namedtuple('ExecResults', ['tile_sum', 'tile_offset'])
Inputs = namedtuple('Inputs', ['means2d', 'opacity', 'conics', 'covars2d', 'cnt', 'tile_grid', 'depth'])

LN2 = 0.69314718055


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def get_rect_v3(means_x, means_y, w, h, width, height):
    rect_min_0 = torch.clamp(means_x - w, 0, width - 1.0)
    rect_min_1 = torch.clamp(means_y - h, 0, height - 1.0)
    rect_max_0 = torch.clamp(means_x + w, 0, width - 1.0)
    rect_max_1 = torch.clamp(means_y + h, 0, height - 1.0)
    return rect_min_0, rect_min_1, rect_max_0, rect_max_1


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def block_contains_center(means_x, means_y, pix_min_x, pix_min_y, pix_max_x, pix_max_y):
    x_res = torch.logical_and(pix_min_x <= means_x[:, None], means_x[:, None] <= pix_max_x)
    y_res = torch.logical_and(pix_min_y <= means_y[:, None], means_y[:, None] <= pix_max_y)
    return torch.logical_and(x_res, y_res)


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def block_intersect_ellipse(pix_min_x, pix_min_y, pix_max_x, pix_max_y, 
                            means_x, means_y, conic_00, conic_01, conic_11, power):
    dx = torch.where(means_x[:, None] * 2 < pix_min_x + pix_max_x, \
                     means_x[:, None] - pix_min_x, means_x[:, None] - pix_max_x)

    w = 2 * power
    a = conic_11[:, None]
    b = -2 * conic_01[:, None] * dx
    c = conic_00[:, None] * dx * dx - w

    flag1 = segment_intersect_ellipse(a, b, c, means_y, pix_min_y, pix_max_y)

    dy = torch.where(means_y[:, None] * 2 < pix_min_y + pix_max_y, \
                     means_y[:, None] - pix_min_y, means_y[:, None] - pix_max_y)

    a = conic_00[:, None]
    b = -2 * conic_01[:, None] * dy
    c = conic_11[:, None] * dy * dy - w

    flag2 = segment_intersect_ellipse(a, b, c, means_x, pix_min_x, pix_max_x)

    return flag1 | flag2


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def segment_intersect_ellipse(a, b, c, mean, pix_min, pix_max):
    delta = b * b - 4 * a * c
    t1 = (pix_min - mean[:, None]) * (2 * a) + b
    t2 = (pix_max - mean[:, None]) * (2 * a) + b
    return (delta >= 0.0) & ((t1 <= 0.0) | (t1 * t1 <= delta)) & ((t2 >= 0.0) | (t2 * t2 <= delta))


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def golden(means2d, opacity, conics, covars2d, tile_grid, image_width, image_height, tile_size):
    means_x, means_y = means2d[..., 0], means2d[..., 1]
    cov00, cov11 = covars2d[..., 0], covars2d[..., 2]
    conic_00, conic_01, conic_11 = conics[..., 0], conics[..., 1], conics[..., 2]

    num_gs = means_x.shape[0]
    num_tile = tile_grid.shape[0]
    power = LN2 * 8 + LN2 * torch.log2(opacity)

    w = (torch.sqrt(2 * cov00[:, None] * power) + 1).floor().squeeze()
    h = (torch.sqrt(2 * cov11[:, None] * power) + 1).floor().squeeze()

    rmin_w, rmin_h, rmax_w, rmax_h = get_rect_v3(means_x, means_y, w, h, image_width, image_height)

    w_right_bound = torch.clamp(rmax_w[:, None].expand(num_gs, num_tile), \
                    max=tile_grid[None, :, 1] + tile_size).expand(num_gs, num_tile)
    w_left_bound = torch.clamp(rmin_w[:, None].expand(num_gs, num_tile), \
                   min=tile_grid[None, :, 1]).expand(num_gs, num_tile)
    h_upper_bound = torch.clamp(rmax_h[:, None].expand(num_gs, num_tile), \
                    max=tile_grid[None, :, 0] + tile_size).expand(num_gs, num_tile)
    h_lower_bound = torch.clamp(rmin_h[:, None].expand(num_gs, num_tile), \
                   min=tile_grid[None, :, 0]).expand(num_gs, num_tile)

    all_in_mask = (w_right_bound > w_left_bound) & (h_upper_bound > h_lower_bound)

    pix_min_x = tile_grid[None, :, 1]
    pix_max_x = tile_grid[None, :, 1] + tile_size - 1
    pix_min_y = tile_grid[None, :, 0]
    pix_max_y = tile_grid[None, :, 0] + tile_size - 1

    center_flag = block_contains_center(means_x, means_y, pix_min_x, pix_min_y, pix_max_x, pix_max_y)
    ellipse_isect_flag = block_intersect_ellipse(pix_min_x, pix_min_y, \
                                                 pix_max_x, pix_max_y, means_x, 
                                                 means_y, conic_00, conic_01, conic_11, power)
    all_in_mask = all_in_mask & (center_flag | ellipse_isect_flag)
    return all_in_mask


class TestFlashGaussianBuildMask(TestCase):
    def setUp(self):
        self.batch_size = 5
        self.camera_num = 3
        self.test_cases = [
            [6789, 9, 11, 64],
            [128, 16, 16, 16],
            [135791, 160, 120, 40],
            [112233, 72, 72, 18]
        ]
        self.test_results = self.gen_results()

    def gen_inputs(self, shape):
        batch_size, camera_num, gaussian_num, image_width, image_height, tile_size = shape

        means2d = torch.rand(batch_size, camera_num, gaussian_num, 2).float()
        conics = torch.rand(batch_size, camera_num, gaussian_num, 3).float()
        opacity = torch.rand(batch_size, camera_num, gaussian_num, 1).float()
        covars2d = torch.rand(batch_size, camera_num, gaussian_num, 3).float()
        depth = torch.rand(batch_size, camera_num, gaussian_num, 1).float()
        cnt = torch.rand(batch_size, camera_num, 1).float()
        cnt.uniform_(1, gaussian_num)
        padded_width = math.ceil(image_width / tile_size) * tile_size
        padded_height = math.ceil(image_height / tile_size) * tile_size
        tile_grid = torch.stack(torch.meshgrid(torch.arange(0, padded_height, tile_size), \
                    torch.arange(0, padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).float()

        return Inputs(means2d, opacity, conics, covars2d, cnt.int(), tile_grid, depth)

    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def cpu_to_exec(self, inputs, image_width, image_height, tile_size, b, c):
        means2d = inputs.means2d[b, c].npu()
        conics = inputs.conics[b, c].npu()
        covars2d = inputs.covars2d[b, c].npu()
        cnt = inputs.cnt[b, c].item()
        tile_grid = inputs.tile_grid.npu()
        opacity = inputs.opacity[b, c].npu()

        mask = golden(means2d, opacity, conics, covars2d, tile_grid, image_width, image_height, tile_size).permute(1, 0)

        mask[:, cnt:] = 0
        tile_sum = torch.sum(mask, dim=1).reshape(-1, 1)
        tile_offset = torch.cumsum(tile_sum, dim=0).reshape(-1, 1)

        return ExecResults(
            tile_sum=tile_sum.detach().float(),
            tile_offset=tile_offset.detach().float()
        )

    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def npu_to_exec(self, inputs, image_width, image_height, tile_size):
        means2d = inputs.means2d.permute(0, 1, 3, 2).contiguous().npu()
        conics = inputs.conics.permute(0, 1, 3, 2).contiguous().npu()
        covars2d = inputs.covars2d.permute(0, 1, 3, 2).contiguous().npu()
        cnt = inputs.cnt.npu()
        tile_grid = inputs.tile_grid.npu()
        opacity = inputs.opacity.permute(0, 1, 3, 2).npu()
        depth = inputs.depth.permute(0, 1, 3, 2).npu()

        tile_sum, tile_offset, tile_depths, gauss_index = flash_gaussian_build_mask(
            means2d, opacity, conics, covars2d, depth, cnt, tile_grid, image_width, image_height, tile_size
        )

        return ExecResults(
            tile_sum=tile_sum.detach().float(),
            tile_offset=tile_offset.detach().float()
        )

    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            gaussian_num, image_width, image_height, tile_size = test_case
            inputs = self.gen_inputs([self.batch_size, self.camera_num, \
                     gaussian_num, image_width, image_height, tile_size])

            npu_results = self.npu_to_exec(inputs, image_width, image_height, tile_size)
            cpu_results = ExecResults(
                            tile_sum=torch.zeros_like(npu_results.tile_sum, device=npu_results.tile_sum.device),
                            tile_offset=torch.zeros_like(npu_results.tile_offset, device=npu_results.tile_offset.device)
                        )
            for b in range(self.batch_size):
                for c in range(self.camera_num):
                    cpu_result = self.cpu_to_exec(inputs, image_width, image_height, tile_size, b, c)
                    cpu_results.tile_sum[b, c] = cpu_result.tile_sum
                    cpu_results.tile_offset[b, c] = cpu_result.tile_offset
            test_results.append((cpu_results, npu_results))
        return test_results

    def test_flash_gaussian_build_mask(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.tile_sum.cpu().numpy(), npu_results.tile_sum.cpu().numpy())
            self.assertRtolEqual(cpu_results.tile_offset.cpu().numpy(), npu_results.tile_offset.cpu().numpy())


if __name__ == "__main__":
    run_tests()