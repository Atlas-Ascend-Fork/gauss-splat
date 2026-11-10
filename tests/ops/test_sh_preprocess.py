# coding: utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import sh_preprocess


class TestShPreprocess(TestCase):
    def test_channel_first_forward_and_backward(self):
        torch.manual_seed(29)
        for n in (1, 7, 8, 9, 65, 4097):
            with self.subTest(n=n):
                means_cpu = torch.randn(3, n, dtype=torch.float32, requires_grad=True)
                cameras_cpu = torch.eye(4, dtype=torch.float32).unsqueeze(0)
                cameras_cpu[0, :3, 3] = torch.tensor([0.25, -0.5, 0.75])
                cameras_cpu.requires_grad_(True)
                upstream_cpu = torch.randn(1, 3, n)
                delta = means_cpu - cameras_cpu[0, :3, 3].unsqueeze(1)
                expected = (delta / delta.norm(dim=0, keepdim=True)).unsqueeze(0)
                v_camera_cpu, v_means_cpu = torch.autograd.grad(
                    expected, (cameras_cpu, means_cpu), upstream_cpu,
                )

                means_npu = means_cpu.detach().npu().requires_grad_(True)
                cameras_npu = cameras_cpu.detach().npu().requires_grad_(True)
                actual = sh_preprocess(cameras_npu, means_npu)
                v_camera_npu, v_means_npu = torch.autograd.grad(
                    actual, (cameras_npu, means_npu), upstream_cpu.npu(),
                )
                torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=1e-4, atol=1e-5)
                torch.testing.assert_close(v_means_npu.cpu(), v_means_cpu, rtol=2e-4, atol=2e-5)
                torch.testing.assert_close(v_camera_npu.cpu(), v_camera_cpu, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    run_tests()
