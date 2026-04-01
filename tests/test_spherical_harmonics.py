# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from collections import namedtuple
import struct
import random
from typing import Optional, Tuple
from typing_extensions import Literal, assert_never

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch import Tensor
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import spherical_harmonics

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=np.inf)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

ExecResults = namedtuple('ExecResults', ['output'])
Inputs = namedtuple('Inputs', ['dirs', 'coeffs'])


def _eval_sh_bases_fast(basis_dim: int, dirs: Tensor):
    """
    Evaluate spherical harmonics bases at unit direction for high orders
    using approach described by
    Efficient Spherical Harmonic Evaluation, Peter-Pike Sloan, JCGT 2013
    https://jcgt.org/published/0002/02/06/


    :param basis_dim: int SH basis dim. Currently, only 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., basis_dim)

    See reference C++ code in https://jcgt.org/published/0002/02/06/code.zip
    """
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )

    result[..., 0] = 0.2820947917738781

    if basis_dim <= 1:
        return result

    x, y, z = dirs.unbind(-1)

    ftmp_a = -0.48860251190292
    result[..., 2] = -ftmp_a * z
    result[..., 3] = ftmp_a * x
    result[..., 1] = ftmp_a * y

    if basis_dim <= 4:
        return result

    z2 = z * z
    ftmp_b = -1.092548430592079 * z
    ftmp_a = 0.5462742152960395
    fc_1 = x * x - y * y
    fs_1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = ftmp_b * x
    result[..., 5] = ftmp_b * y
    result[..., 8] = ftmp_a * fc_1
    result[..., 4] = ftmp_a * fs_1

    if basis_dim <= 9:
        return result

    ftmp_c = -2.285228997322329 * z2 + 0.4570457994644658
    ftmp_b = 1.445305721320277 * z
    ftmp_a = -0.5900435899266435
    fc_2 = x * fc_1 - y * fs_1
    fs_2 = x * fs_1 + y * fc_1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = ftmp_c * x
    result[..., 11] = ftmp_c * y
    result[..., 14] = ftmp_b * fc_1
    result[..., 10] = ftmp_b * fs_1
    result[..., 15] = ftmp_a * fc_2
    result[..., 9] = ftmp_a * fs_2

    if basis_dim <= 16:
        return result

    ftmp_d = z * (-4.683325804901025 * z2 + 2.007139630671868)
    ftmp_c = 3.31161143515146 * z2 - 0.47308734787878
    ftmp_b = -1.770130769779931 * z
    ftmp_a = 0.6258357354491763
    fc_3 = x * fc_2 - y * fs_2
    fs_3 = x * fs_2 + y * fc_2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = ftmp_d * x
    result[..., 19] = ftmp_d * y
    result[..., 22] = ftmp_c * fc_1
    result[..., 18] = ftmp_c * fs_1
    result[..., 23] = ftmp_b * fc_2
    result[..., 17] = ftmp_b * fs_2
    result[..., 24] = ftmp_a * fc_3
    result[..., 16] = ftmp_a * fs_3
    return result


def _spherical_harmonics(
    degrees_to_use: int,
    dirs: torch.Tensor,  # [..., 3]
    coeffs: torch.Tensor,  # [..., K, 3]
):
    """Pytorch implementation of `gsplat.cuda._wrapper.spherical_harmonics()`."""
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    batch_dims = dirs.shape[:-1]
    assert dirs.shape == batch_dims + (3,), dirs.shape
    assert (
        (len(coeffs.shape) == len(batch_dims) + 2)
        and coeffs.shape[:-2] == batch_dims
        and coeffs.shape[-1] == 3
    ), coeffs.shape
    dirs = F.normalize(dirs, p=2, dim=-1)
    num_bases = (degrees_to_use + 1) ** 2
    bases = torch.zeros_like(coeffs[..., 0])
    bases[..., :num_bases] = _eval_sh_bases_fast(num_bases, dirs)
    return (bases[..., None] * coeffs).sum(dim=-2)


class TestSphericalHarmonicsForward(TestCase):
    def setUp(self):
        self.test_cases = [
            [4, 3872],
            [3, 21983],
            [5, 89210],
            [1, 119987],
            [6, 40000]
        ]
        self.test_results = self.gen_results()

    def gen_inputs(self, shape):
        batch_size, task_num = shape
        k = random.randint(0, 5)

        dirs = torch.rand(batch_size, task_num, 3).float()
        coeffs0 = torch.rand(batch_size, task_num, k + 1, 3).float()
        coeffs1 = torch.rand(batch_size, task_num, k + 4, 3).float()
        coeffs2 = torch.rand(batch_size, task_num, k + 9, 3).float()
        coeffs3 = torch.rand(batch_size, task_num, k + 16, 3).float()
        coeffs4 = torch.rand(batch_size, task_num, k + 25, 3).float()

        return Inputs(dirs, coeffs0), Inputs(dirs, coeffs1), \
               Inputs(dirs, coeffs2), Inputs(dirs, coeffs3), Inputs(dirs, coeffs4)

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

        output = _spherical_harmonics(
            degree, dirs, coeffs
        )
        
        return ExecResults(
            output=output.permute(0, 2, 1).detach().float()
        )

    def npu_to_exec(self, inputs, degree):
        dirs = inputs.dirs.npu()
        coeffs = inputs.coeffs.npu()

        output = spherical_harmonics(
            degree, dirs, coeffs
        )
        
        return ExecResults(
            output=output.detach().float()
        )

    def test_spherical_harmonics_forward(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.output.cpu().numpy(), npu_results.output.cpu().numpy())

if __name__ == "__main__":
    run_tests()