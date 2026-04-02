"""
Copyright (c) 2022 Hust Vision Lab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
Licensed under the MIT License.
"""
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
 
from gauss_splat.gsplat_c import spherical_harmonics_bwd
torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=np.inf)
 
option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)
 
ExecResults = namedtuple('ExecResults', ['output1', 'output2'])
Inputs = namedtuple('Inputs', ['dirs', 'coeffs', 'v_colors'])
 
 
def _spherical_harmonics_bwd(
    degree, # int
    dirs, # [..., 3]
    coeffs, # [..., K, 3]
    v_colors, # [..., 3]
):
    v_colors = v_colors.permute(0, 2, 1).contiguous()
    # degree = int(math.sqrt(coeffs.shape[-2]) - 1)
    v_coeffs = torch.zeros_like(coeffs)
    v_dirs = torch.zeros_like(dirs)
    # case l=0, m=0
    c00 = 0.2820947917738781
    v_coeffs[..., 0, :3] = c00 * v_colors[..., :3]

    if degree == 0:
        # e.g degree=0
        return v_dirs, v_coeffs

    inorm = torch.rsqrt((dirs ** 2).sum(-1, keepdim=True))
    dirs = F.normalize(dirs, p=2, dim=-1)
    x, y, z = dirs.unbind(-1)
    
    v_coeffs[..., 1, :3] = -0.48860251190292 * y[..., None] * v_colors[..., :3]
    v_coeffs[..., 2, :3] = 0.48860251190292 * z[..., None] * v_colors[..., :3]
    v_coeffs[..., 3, :3] = -0.48860251190292 * x[..., None] * v_colors[..., :3]
    
    # v_dir degree=1
    v_x = (-0.48860251190292 * coeffs[..., 3, :3] * v_colors[..., :3]).sum(-1, keepdim=True)
    v_y = (-0.48860251190292 * coeffs[..., 1, :3] * v_colors[..., :3]).sum(-1, keepdim=True)
    v_z = (0.48860251190292 * coeffs[..., 2, :3] * v_colors[..., :3]).sum(-1, keepdim=True)
    
    # vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm
    
    if degree == 1:
        v_dir_n = torch.concat([v_x, v_y, v_z], dim=-1)
        v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
        v_dirs = v_dirs + v_d
        return v_dirs, v_coeffs
    
    z2 = z * z
    ftmp_0b = -1.092548430592079 * z
    fc1 = x * x - y * y
    fs1 = 2.0 * x * y
    p_sh6 = (0.9461746957575601 * z2 - 0.3153915652525201)
    p_sh7 = ftmp_0b * x
    p_sh5 = ftmp_0b * y
    p_sh8 = 0.5462742152960395 * fc1
    p_sh4 = 0.5462742152960395 * fs1
    v_coeffs[..., 4, :3] = p_sh4[..., None] * v_colors[..., :3] # 系数shape pSH4增加一维度用于broadcast: [B,N] --> [B,N,1]
    v_coeffs[..., 5, :3] = p_sh5[..., None] * v_colors[..., :3]
    v_coeffs[..., 6, :3] = p_sh6[..., None] * v_colors[..., :3]
    v_coeffs[..., 7, :3] = p_sh7[..., None] * v_colors[..., :3]
    v_coeffs[..., 8, :3] = p_sh8[..., None] * v_colors[..., :3]

    # v_dir degree=2
    ftmp_0b_z = -1.092548430592079
    fc1_x = 2.0 * x
    fc1_y = -2.0 * y
    fs1_x = 2.0 * y
    fs1_y = 2.0 * x
    p_sh6_z = 2.0 * 0.9461746957575601 * z
    p_sh7_x = ftmp_0b
    p_sh7_z = ftmp_0b_z * x
    p_sh5_y = ftmp_0b
    p_sh5_z = ftmp_0b_z * y
    p_sh8_x = 0.5462742152960395 * fc1_x
    p_sh8_y = 0.5462742152960395 * fc1_y
    p_sh4_x = 0.5462742152960395 * fs1_x
    p_sh4_y = 0.5462742152960395 * fs1_y

    v_x = v_x + (v_colors * (p_sh4_x[..., None] * coeffs[..., 4, :3] + p_sh8_x[..., None] * coeffs[..., 8, :3] +
                p_sh7_x[..., None] * coeffs[..., 7, :3])).sum(-1, keepdim=True)
    v_y = v_y + (v_colors * (p_sh4_y[..., None] * coeffs[..., 4, :3] + p_sh8_y[..., None] * coeffs[..., 8, :3] +
                p_sh5_y[..., None] * coeffs[..., 5, :3])).sum(-1, keepdim=True)
    v_z = v_z + (v_colors * (p_sh6_z[..., None] * coeffs[..., 6, :3] + p_sh7_z[..., None] * coeffs[..., 7, :3] +
                p_sh5_z[..., None] * coeffs[..., 5, :3])).sum(-1, keepdim=True)
    if degree < 3:
        v_dir_n = torch.concat([v_x, v_y, v_z], dim=-1)
        v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
        v_dirs = v_dirs + v_d
        return v_dirs, v_coeffs
    
    ftmp_0c = -2.285228997322329 * z2 + 0.4570457994644658
    ftmp_1b = 1.445305721320277 * z
    fc2 = x * fc1 - y * fs1
    fs2 = x * fs1 + y * fc1
    p_sh_12 = z * (1.865881662950577 * z2 - 1.119528997770346)
    p_sh_13 = ftmp_0c * x
    p_sh_11 = ftmp_0c * y
    p_sh_14 = ftmp_1b * fc1
    p_sh_10 = ftmp_1b * fs1
    p_sh_15 = -0.5900435899266435 * fc2
    p_sh_9 = -0.5900435899266435 * fs2
    v_coeffs[..., 9, :3] = p_sh_9[..., None] * v_colors[..., :3]
    v_coeffs[..., 10, :3] = p_sh_10[..., None] * v_colors[..., :3]
    v_coeffs[..., 11, :3] = p_sh_11[..., None] * v_colors[..., :3]
    v_coeffs[..., 12, :3] = p_sh_12[..., None] * v_colors[..., :3]
    v_coeffs[..., 13, :3] = p_sh_13[..., None] * v_colors[..., :3]
    v_coeffs[..., 14, :3] = p_sh_14[..., None] * v_colors[..., :3]
    v_coeffs[..., 15, :3] = p_sh_15[..., None] * v_colors[..., :3]

    ftmp_0c_z = -2.285228997322329 * 2.0 * z
    ftmp_1b_z = 1.445305721320277
    fc2_x = fc1 + x * fc1_x - y * fs1_x
    fc2_y = x * fc1_y - fs1 - y * fs1_y
    fs2_x = fs1 + x * fs1_x + y * fc1_x
    fs2_y = x * fs1_y + fc1 + y * fc1_y
    p_sh12_z = 3.0 * 1.865881662950577 * z2 - 1.119528997770346
    p_sh13_x = ftmp_0c
    p_sh13_z = ftmp_0c_z * x
    p_sh11_y = ftmp_0c
    p_sh11_z = ftmp_0c_z * y
    p_sh14_x = ftmp_1b * fc1_x
    p_sh14_y = ftmp_1b * fc1_y
    p_sh14_z = ftmp_1b_z * fc1
    p_sh10_x = ftmp_1b * fs1_x
    p_sh10_y = ftmp_1b * fs1_y
    p_sh10_z = ftmp_1b_z * fs1
    p_sh15_x = -0.5900435899266435 * fc2_x
    p_sh15_y = -0.5900435899266435 * fc2_y
    p_sh9_x = -0.5900435899266435 * fs2_x
    p_sh9_y = -0.5900435899266435 * fs2_y

    v_x = v_x + (v_colors *
                (p_sh9_x[..., None] * coeffs[..., 9, :3] + p_sh15_x[..., None] * coeffs[..., 15, :3] +
                    p_sh10_x[..., None] * coeffs[..., 10, :3] + p_sh14_x[..., None] * coeffs[..., 14, :3] +
                    p_sh13_x[..., None] * coeffs[..., 13, :3])).sum(dim=-1, keepdim=True)

    v_y = v_y + (v_colors *
                (p_sh9_y[..., None] * coeffs[..., 9, :3] + p_sh15_y[..., None] * coeffs[..., 15, :3] +
                    p_sh10_y[..., None] * coeffs[..., 10, :3] + p_sh14_y[..., None] * coeffs[..., 14, :3] +
                    p_sh11_y[..., None] * coeffs[..., 11, :3])).sum(dim=-1, keepdim=True)

    v_z = v_z + (v_colors *
                (p_sh12_z[..., None] * coeffs[..., 12, :3] + p_sh13_z[..., None] * coeffs[..., 13, :3] +
                p_sh11_z[..., None] * coeffs[..., 11, :3] + p_sh14_z[..., None] * coeffs[..., 14, :3] +
                p_sh10_z[..., None] * coeffs[..., 10, :3])).sum(dim=-1, keepdim=True)
    
    if degree == 3:
        v_dir_n = torch.concat([v_x, v_y, v_z], dim=-1)
        v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
        v_dirs = v_dirs + v_d
        return v_dirs, v_coeffs
    
    ftmp_0d = z * (-4.683325804901025 * z2 + 2.007139630671868)
    ftmp_1c = 3.31161143515146 * z2 - 0.47308734787878
    ftmp_2b = -1.770130769779931 * z
    fc3 = x * fc2 - y * fs2
    fs3 = x * fs2 + y * fc2
    p_sh_20 = (1.984313483298443 * z * p_sh_12 + -1.006230589874905 * p_sh6)
    p_sh_21 = ftmp_0d * x
    p_sh_19 = ftmp_0d * y
    p_sh_22 = ftmp_1c * fc1
    p_sh_18 = ftmp_1c * fs1
    p_sh_23 = ftmp_2b * fc2
    p_sh_17 = ftmp_2b * fs2
    p_sh_24 = 0.6258357354491763 * fc3
    p_sh_16 = 0.6258357354491763 * fs3

    v_coeffs[..., 16, :3] = p_sh_16[..., None] * v_colors[..., :3]
    v_coeffs[..., 17, :3] = p_sh_17[..., None] * v_colors[..., :3]
    v_coeffs[..., 18, :3] = p_sh_18[..., None] * v_colors[..., :3]
    v_coeffs[..., 19, :3] = p_sh_19[..., None] * v_colors[..., :3]
    v_coeffs[..., 20, :3] = p_sh_20[..., None] * v_colors[..., :3]
    v_coeffs[..., 21, :3] = p_sh_21[..., None] * v_colors[..., :3]
    v_coeffs[..., 22, :3] = p_sh_22[..., None] * v_colors[..., :3]
    v_coeffs[..., 23, :3] = p_sh_23[..., None] * v_colors[..., :3]
    v_coeffs[..., 24, :3] = p_sh_24[..., None] * v_colors[..., :3]

    ftmp_0d_z = 3.0 * -4.683325804901025 * z2 + 2.007139630671868
    ftmp_1c_z = 2.0 * 3.31161143515146 * z
    ftmp_2b_z = -1.770130769779931
    fc3_x = fc2 + x * fc2_x - y * fs2_x
    fc3_y = x * fc2_y - fs2 - y * fs2_y
    fs3_x = fs2 + y * fc2_x + x * fs2_x
    fs3_y = x * fs2_y + fc2 + y * fc2_y
    p_sh20_z = 1.984313483298443 * (p_sh_12 + z * p_sh12_z) + (-1.006230589874905 * p_sh6_z)
    p_sh21_x = ftmp_0d
    p_sh21_z = ftmp_0d_z * x
    p_sh19_y = ftmp_0d
    p_sh19_z = ftmp_0d_z * y
    p_sh22_x = ftmp_1c * fc1_x
    p_sh22_y = ftmp_1c * fc1_y
    p_sh22_z = ftmp_1c_z * fc1
    p_sh18_x = ftmp_1c * fs1_x
    p_sh18_y = ftmp_1c * fs1_y
    p_sh18_z = ftmp_1c_z * fs1
    p_sh23_x = ftmp_2b * fc2_x
    p_sh23_y = ftmp_2b * fc2_y
    p_sh23_z = ftmp_2b_z * fc2
    p_sh17_x = ftmp_2b * fs2_x
    p_sh17_y = ftmp_2b * fs2_y
    p_sh17_z = ftmp_2b_z * fs2
    p_sh24_x = 0.6258357354491763 * fc3_x
    p_sh24_y = 0.6258357354491763 * fc3_y
    p_sh16_x = 0.6258357354491763 * fs3_x
    p_sh16_y = 0.6258357354491763 * fs3_y

    v_x = v_x + (v_colors *
            (p_sh16_x[..., None] * coeffs[..., 16, :3] + p_sh24_x[..., None] * coeffs[..., 24, :3] +
            p_sh17_x[..., None] * coeffs[..., 17, :3] + p_sh23_x[..., None] * coeffs[..., 23, :3] +
            p_sh18_x[..., None] * coeffs[..., 18, :3] + p_sh22_x[..., None] * coeffs[..., 22, :3] +
            p_sh21_x[..., None] * coeffs[..., 21, :3])).sum(dim=-1, keepdim=True)
    v_y = v_y + (v_colors *
            (p_sh16_y[..., None] * coeffs[..., 16, :3] + p_sh24_y[..., None] * coeffs[..., 24, :3] +
            p_sh17_y[..., None] * coeffs[..., 17, :3] + p_sh23_y[..., None] * coeffs[..., 23, :3] +
            p_sh18_y[..., None] * coeffs[..., 18, :3] + p_sh22_y[..., None] * coeffs[..., 22, :3] +
            p_sh19_y[..., None] * coeffs[..., 19, :3])).sum(dim=-1, keepdim=True)
    v_z = v_z + (v_colors *
            (p_sh20_z[..., None] * coeffs[..., 20, :3] + p_sh21_z[..., None] * coeffs[..., 21, :3] +
            p_sh19_z[..., None] * coeffs[..., 19, :3] + p_sh22_z[..., None] * coeffs[..., 22, :3] +
            p_sh18_z[..., None] * coeffs[..., 18, :3] + p_sh23_z[..., None] * coeffs[..., 23, :3] +
            p_sh17_z[..., None] * coeffs[..., 17, :3])).sum(dim=-1, keepdim=True)
    
    v_dir_n = torch.concat([v_x, v_y, v_z], dim=-1)
    v_d = (v_dir_n - torch.einsum('bni,bni->bn', v_dir_n, dirs)[..., None] * dirs) * inorm
    v_dirs = v_dirs + v_d
    return v_dirs, v_coeffs
 
 
class TestSphericalHarmonicsForward(TestCase):
    def setUp(self):
        self.test_cases = [
            3872,
            21983
            ]
        self.test_results = self.gen_results()
 
    def gen_inputs(self, shape):
        task_num = shape

        dirs = torch.rand(1, task_num, 3).float()
        v_colors = torch.rand(1, 3, task_num).float()
        coeffs0 = torch.rand(1, task_num, 1, 3).float()
        coeffs1 = torch.rand(1, task_num, 4, 3).float()
        coeffs2 = torch.rand(1, task_num, 9, 3).float()
        coeffs3 = torch.rand(1, task_num, 16, 3).float()
        coeffs4 = torch.rand(1, task_num, 25, 3).float()
 
        return Inputs(dirs, coeffs0, v_colors), Inputs(dirs, coeffs1, v_colors), Inputs(dirs, coeffs2, v_colors),\
               Inputs(dirs, coeffs3, v_colors), Inputs(dirs, coeffs4, v_colors)
 
    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            inputs0, inputs1, inputs2, inputs3, inputs4 = self.gen_inputs(test_case)
            cpu_results0 = self.cpu_to_exec(inputs0, 0)
            npu_results0 = self.npu_to_exec(inputs0, 0)
            test_results.append((cpu_results0, npu_results0))
 
            cpu_results1 = self.cpu_to_exec(inputs1, 1)
            npu_results1 = self.npu_to_exec(inputs1, 1)
            test_results.append((cpu_results1, npu_results1))
 
            cpu_results2 = self.cpu_to_exec(inputs2, 2)
            npu_results2 = self.npu_to_exec(inputs2, 2)
            test_results.append((cpu_results2, npu_results2))
 
            cpu_results3 = self.cpu_to_exec(inputs3, 3)
            npu_results3 = self.npu_to_exec(inputs3, 3)
            test_results.append((cpu_results3, npu_results3))
 
            cpu_results4 = self.cpu_to_exec(inputs4, 4)
            npu_results4 = self.npu_to_exec(inputs4, 4)
            test_results.append((cpu_results4, npu_results4))
        return test_results
 
    def cpu_to_exec(self, inputs, degree):
        dirs = inputs.dirs
        coeffs = inputs.coeffs
        v_colors = inputs.v_colors
        output1, output2 = _spherical_harmonics_bwd(degree, dirs, coeffs, v_colors)
        return ExecResults(
            output1=output1.detach().float(),
            output2=output2.detach().float()
        )
 
 
    def npu_to_exec(self, inputs, degree):
        dirs = inputs.dirs.npu()
        coeffs = inputs.coeffs.npu()
        v_colors = inputs.v_colors.npu()
        output1, output2 = spherical_harmonics_bwd(dirs, coeffs, v_colors, degree)
        return ExecResults(
            output1=output1.detach().float(),
            output2=output2.detach().float()
        )
 
    def test_spherical_harmonics_forward(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.output1.numpy(), npu_results.output1.cpu().numpy())
            self.assertRtolEqual(cpu_results.output2.numpy(), npu_results.output2.cpu().numpy())
 
if __name__ == "__main__":
    run_tests()