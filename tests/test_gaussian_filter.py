# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import unittest
import struct
import math
import random
from typing import Optional, Tuple
from collections import namedtuple
from typing_extensions import Literal, assert_never
 
import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch import Tensor
 
from gauss_splat import gaussian_filter
 
torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=np.inf)
 
option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)
 
ExecResults = namedtuple('ExecResults', ['means_culling', 'colors_culling', 'means2d_culling', 'depths_culling', \
                'radius_culling', 'covars2d_culling', 'conics_culling', 'opacities_culling', 'proj_filter', 'cnt'])
Inputs = namedtuple('Inputs', ['means', 'colors', 'det', 'opacities', 'means2d', 'depths', \
        'radius', 'conics', 'covars2d', 'compensations'])
 
 
# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def _gaussian_filter(means, colors, det, opacities, means2d, depths, radius_in,
                     conics, covars2d, compensations, width, height, near_plane, far_plane):
    batch_dims, camera_nums, gaussian_nums = det.shape
    if compensations is not None:
        opacities = opacities.unsqueeze(1) * compensations
    else:
        opacities = opacities.unsqueeze(1).repeat(1, camera_nums, 1)
    means = means.float().permute(0, 2, 1).contiguous()
    means2d = means2d.float().permute(0, 1, 3, 2).contiguous()
    radius = radius_in.float().permute(0, 1, 3, 2).contiguous()
    radius_out = radius_in.float().permute(0, 1, 3, 2).contiguous()
    conics = conics.float().permute(0, 1, 3, 2).contiguous()
    colors = colors.float().permute(0, 2, 1).contiguous()
    covars2d = covars2d.float().permute(0, 1, 3, 2).contiguous()
    det = det.float()
    opacities = opacities.float()
    depths = depths.float()
    if compensations is not None:
        compensations = compensations.float()
    
    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0
    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0
    radii = radius.int()
    proj_filter = torch.logical_and(inside, valid)
    means_culling = torch.zeros_like(conics)
    radius_culling = torch.zeros_like(radius)
    means2d_culling = torch.zeros_like(means2d)
    depths_culling = torch.zeros_like(depths)
    opacities_culling = torch.zeros_like(depths)
    conics_culling = torch.zeros_like(conics)
    colors_culling = torch.zeros_like(conics)
    covars2d_culling = torch.zeros_like(covars2d)

    for b in range(batch_dims):
        for c in range(camera_nums):
            radius_culling[b, c, :proj_filter[b, c].sum()] = radius_out[b, c, proj_filter[b, c]]
            means_culling[b, c, :proj_filter[b, c].sum()] = means[b, proj_filter[b, c]]
            means2d_culling[b, c, :proj_filter[b, c].sum()] = means2d[b, c, proj_filter[b, c]]
            depths_culling[b, c, :proj_filter[b, c].sum()] = depths[b, c, proj_filter[b, c]]
            conics_culling[b, c, :proj_filter[b, c].sum()] = conics[b, c, proj_filter[b, c]]
            colors_culling[b, c, :proj_filter[b, c].sum()] = colors[b, proj_filter[b, c]]
            covars2d_culling[b, c, :proj_filter[b, c].sum()] = covars2d[b, c, proj_filter[b, c]]
            opacities_culling[b, c, :proj_filter[b, c].sum()] = opacities[b, c, proj_filter[b, c]]

    cnt = proj_filter.sum(-1)

    filter_bool = proj_filter.bool()
    remainder = gaussian_nums % 8
    if remainder != 0:
        pad_size = 8 - remainder
        filter_bool = F.pad(proj_filter, (0, pad_size), mode='constant', value=False)
    matrix_r = (gaussian_nums + 7) // 8
    filter_reshaped = filter_bool.reshape(batch_dims, camera_nums, matrix_r, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], 
                         dtype=torch.uint8, device=proj_filter.device)
    filter_uint8 = (filter_reshaped.to(torch.uint8) * powers).sum(dim=-1, dtype=torch.uint8)
    
    means_culling = means_culling.permute(0, 1, 3, 2).contiguous()
    radius_culling = radius_culling.permute(0, 1, 3, 2).contiguous()
    means2d_culling = means2d_culling.permute(0, 1, 3, 2).contiguous()
    conics_culling = conics_culling.permute(0, 1, 3, 2).contiguous()
    colors_culling = colors_culling.permute(0, 1, 3, 2).contiguous()
    covars2d_culling = covars2d_culling.permute(0, 1, 3, 2).contiguous()

    return means_culling.float(), colors_culling.float(), means2d_culling.float(), \
           depths_culling.float(), radius_culling.float(), covars2d_culling.float(), \
           conics_culling.float(), opacities_culling.float(), filter_uint8, cnt.to(torch.int32)
 
 
class TestSphericalHarmonicsForward(TestCase):
    def setUp(self):
        self.test_cases = [
            [1, 1, 10000],
            [2, 7, 18471],
            [1, 1, 117611],
            [1, 1, 188152]
            ]
        self.test_results = self.gen_results()
 
    def gen_inputs(self, shape):
        b = shape[0]
        c = shape[1]
        n = shape[2]
        means = torch.rand(b, 3, n).float() + 1.0
        colors = torch.rand(b, 3, n).float() + 1.0
        det = torch.rand(b, c, n).float() + 1.0
        opacities = torch.rand(b, n).float() + 1.0
        compensations = torch.rand(b, c, n).float() + 1.0
        means2d = torch.rand(b, c, 2, n).float() + 1.0
        depths = torch.rand(b, c, n).float() + 1.0
        radius = torch.rand(b, c, 2, n).float() + 1.0
        conics = torch.rand(b, c, 3, n).float() + 1.0
        covars2d = torch.rand(b, c, 3, n).float() + 1.0

 
        return Inputs(means, colors, det, opacities, means2d, depths, radius, conics, covars2d, compensations),\
               Inputs(means, colors, det, opacities, means2d, depths, radius, conics, covars2d, compensations),

    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            inputs0, inputs1 = self.gen_inputs(test_case)
            cpu_results0 = self.cpu_to_exec(inputs0, 0, 600, 1.0, 2.0)
            npu_results0 = self.npu_to_exec(inputs0, 0, 600, 1.0, 2.0)
            test_results.append((cpu_results0, npu_results0))
 
            cpu_results1 = self.cpu_to_exec(inputs1, 0, 600, 1.0, 2.0)
            npu_results1 = self.npu_to_exec(inputs1, 0, 600, 1.0, 2.0)
            test_results.append((cpu_results1, npu_results1))

        return test_results
 
    def cpu_to_exec(self, inputs, width, height, near_plane, far_plane):
        means = inputs.means
        colors = inputs.colors
        det = inputs.det
        opacities = inputs.opacities
        compensations = inputs.compensations
        means2d = inputs.means2d
        depths = inputs.depths
        radius = inputs.radius
        conics = inputs.conics
        covars2d = inputs.covars2d
        means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
            covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt = _gaussian_filter(means,\
            colors, det, opacities, means2d, depths, radius, conics, covars2d,\
            compensations, width, height, near_plane, far_plane)
        return ExecResults(
            means_culling=means_culling.detach().float(),
            colors_culling=colors_culling.detach().float(),
            means2d_culling=means2d_culling.detach().float(),
            depths_culling=depths_culling.detach().float(),
            radius_culling=radius_culling.detach().float(),
            covars2d_culling=covars2d_culling.detach().float(),
            conics_culling=conics_culling.detach().float(),
            opacities_culling=opacities_culling.detach().float(),
            proj_filter=proj_filter.detach().float(),
            cnt=cnt.detach().float()
        )
 
 
    def npu_to_exec(self, inputs, width, height, near_plane, far_plane):
        means = inputs.means.npu()
        colors = inputs.colors.npu()
        det = inputs.det.npu()
        opacities = inputs.opacities.npu()
        compensations = inputs.compensations
        if compensations is not None:
            compensations = compensations.npu()
        means2d = inputs.means2d.npu()
        depths = inputs.depths.npu()
        radius = inputs.radius.npu()
        conics = inputs.conics.npu()
        covars2d = inputs.covars2d.npu()
        means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
            covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt = gaussian_filter(means, colors, det,\
            opacities, means2d, depths, radius, conics, covars2d, compensations, width, height, near_plane, far_plane)
        return ExecResults(
            means_culling=means_culling.detach().float(),
            colors_culling=colors_culling.detach().float(),
            means2d_culling=means2d_culling.detach().float(),
            depths_culling=depths_culling.detach().float(),
            radius_culling=radius_culling.detach().float(),
            covars2d_culling=covars2d_culling.detach().float(),
            conics_culling=conics_culling.detach().float(),
            opacities_culling=opacities_culling.detach().float(),
            proj_filter=proj_filter.detach().float(),
            cnt=cnt.detach().float()
        )
 
    def test_gaussian_filter(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.means_culling.numpy(), npu_results.means_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.colors_culling.numpy(), npu_results.colors_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.means2d_culling.numpy(), npu_results.means2d_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.depths_culling.numpy(), npu_results.depths_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.radius_culling.numpy(), npu_results.radius_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.covars2d_culling.numpy(), npu_results.covars2d_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.conics_culling.numpy(), npu_results.conics_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.opacities_culling.numpy(), npu_results.opacities_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.proj_filter.numpy(), npu_results.proj_filter.cpu().numpy())
            self.assertRtolEqual(cpu_results.cnt.numpy(), npu_results.cnt.cpu().numpy())
 
if __name__ == "__main__":
    run_tests()