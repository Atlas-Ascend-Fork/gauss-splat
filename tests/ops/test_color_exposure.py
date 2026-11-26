# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.

from collections import namedtuple

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat.gsplat_c import color_exposure_bwd, color_exposure_fwd
from gauss_splat.ops.color_exposure import color_exposure

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

Inputs = namedtuple('Inputs', ['colors', 'means2d', 'exposure', 'cnt', 'width', 'height'])
FwdOut = namedtuple('FwdOut', ['output', 'index', 'weight'])
BwdOut = namedtuple('BwdOut', ['v_colors', 'v_exposure'])


def _sat01(x):
    return torch.clamp(x, 0.0, 1.0)


def _f32_inv(value):
    return torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(value), dtype=torch.float32)


def _cell(f, grid_dim):
    idx = torch.trunc(f)
    w = f - idx
    hit = idx == float(grid_dim - 1)
    idx = torch.where(hit, idx - 1, idx)
    w = torch.where(hit, torch.ones_like(w), w)
    return idx.long(), w


def _interp_affine(grid, ix, iy, iz, wx, wy, wz):
    m = None
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                w = (wx if cx else 1 - wx) * (wy if cy else 1 - wy) * (wz if cz else 1 - wz)
                term = w.unsqueeze(-1) * grid[ix + cx, iy + cy, iz + cz]
                m = term if m is None else m + term
    return m


def _valid_mask(cnt, batch, camera, n):
    pos = torch.arange(n)
    return (pos < cnt.reshape(batch, camera, 1)).unsqueeze(2)  # [B, C, 1, N]


def _apply_affine(m, colors, clamp):
    b, c, n, _ = m.shape
    aff = m.reshape(b, c, n, 3, 4)
    matrix = aff[:, :, :, :, :3]
    bias = aff[:, :, :, :, 3]
    out = torch.einsum('bcnij,bcjn->bcin', matrix, colors) + bias.permute(0, 1, 3, 2)
    if clamp:
        out = torch.clamp_min(out, 0.0)
    return out


def golden_fwd(colors, means2d, exposure, cnt, width, height, clamp):
    b, c, _, n = colors.shape
    if exposure.numel() > 12:
        gx, gy, gz = exposure.shape[0], exposure.shape[1], exposure.shape[2]
        grid = exposure.reshape(gx, gy, gz, 12)
        with torch.no_grad():
            inv_w = _f32_inv(width)
            inv_h = _f32_inv(height)
            grey = 0.299 * colors[:, :, 0] + 0.587 * colors[:, :, 1] + 0.114 * colors[:, :, 2]
            fx = _sat01(means2d[:, :, 0] * inv_w) * (gx - 1)
            fy = _sat01(means2d[:, :, 1] * inv_h) * (gy - 1)
            fz = _sat01(grey) * (gz - 1)
            ix, wx = _cell(fx, gx)
            iy, wy = _cell(fy, gy)
            iz, wz = _cell(fz, gz)
        m = _interp_affine(grid, ix, iy, iz, wx, wy, wz)
        index = torch.stack([ix, iy, iz], dim=2).to(torch.int32)
        weight = torch.stack([wx, wy, wz], dim=2)
    else:
        m = exposure.reshape(12).view(1, 1, 1, 12).expand(b, c, n, 12)
        index = torch.empty(0, dtype=torch.int32)
        weight = torch.empty(0, dtype=torch.float32)
    output = _apply_affine(m, colors, clamp)
    valid = _valid_mask(cnt, b, c, n)
    output = torch.where(valid, output, torch.zeros((), dtype=output.dtype))
    if index.numel() > 0:
        index = torch.where(valid, index, torch.zeros((), dtype=index.dtype))
        weight = torch.where(valid, weight, torch.zeros((), dtype=weight.dtype))
    return FwdOut(output, index, weight)


def golden_bwd(colors, v_colors, exposure, cnt, index, weight, clamp, calc_exp_grad):
    b, c, _, n = colors.shape
    use_grid = exposure.numel() > 12
    if use_grid:
        gx, gy, gz = exposure.shape[0], exposure.shape[1], exposure.shape[2]
        grid = exposure.reshape(gx, gy, gz, 12)
        ix, iy, iz = index[:, :, 0].long(), index[:, :, 1].long(), index[:, :, 2].long()
        wx, wy, wz = weight[:, :, 0], weight[:, :, 1], weight[:, :, 2]
        m = _interp_affine(grid, ix, iy, iz, wx, wy, wz)
    else:
        m = exposure.reshape(12).view(1, 1, 1, 12).expand(b, c, n, 12)

    valid = _valid_mask(cnt, b, c, n)
    v = v_colors * valid  # cnt 之外的高斯不产生任何梯度

    aff = m.reshape(b, c, n, 3, 4)
    matrix = aff[:, :, :, :, :3]
    bias = aff[:, :, :, :, 3]
    if clamp:
        pre = torch.einsum('bcnij,bcjn->bcin', matrix, colors) + bias.permute(0, 1, 3, 2)
        v = torch.where(pre < 0, torch.zeros((), dtype=v.dtype), v)
    v_colors_out = torch.einsum('bcnij,bcin->bcjn', matrix, v)

    v_exposure = None
    if calc_exp_grad:
        g = torch.zeros(b, c, n, 3, 4, dtype=torch.float64)
        g[:, :, :, :, :3] = torch.einsum('bcin,bcjn->bcnij', v.double(), colors.double())
        g[:, :, :, :, 3] = v.permute(0, 1, 3, 2).double()
        g12 = g.reshape(b, c, n, 12)
        if use_grid:
            flat = torch.zeros(gx * gy * gz, 12, dtype=torch.float64)
            for cx in (0, 1):
                for cy in (0, 1):
                    for cz in (0, 1):
                        w = (wx if cx else 1 - wx) * (wy if cy else 1 - wy) * \
                            (wz if cz else 1 - wz)
                        ids = ((ix + cx) * gy + (iy + cy)) * gz + (iz + cz)
                        flat.index_add_(0, ids.reshape(-1),
                                        (w.unsqueeze(-1).double() * g12).reshape(-1, 12))
            v_exposure = flat.reshape(exposure.shape).float()
        else:
            v_exposure = g12.sum(dim=(0, 1, 2)).reshape(exposure.shape).float()
    return BwdOut(v_colors_out, v_exposure)


def _assert_close(testcase, actual, expected, rtol=1e-5, atol=1e-5, msg=''):
    testcase.assertEqual(actual.shape, expected.shape)
    if expected.numel() == 0:
        return
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual.float() - expected.float()).abs().max().item()
        testcase.fail(f"{msg} max_abs_diff={diff:.3e} (rtol={rtol}, atol={atol})")


def _vexp_tolerance(total_gauss):
    return 2e-4, 1e-5 + 5e-8 * total_gauss


def _gen_cnt(batch, camera, n_gauss, mode):
    if mode == 'full':
        cnt = torch.full((batch, camera), n_gauss, dtype=torch.int32)
    elif mode == 'zero':
        cnt = torch.zeros(batch, camera, dtype=torch.int32)
    elif mode == 'mixed':
        cnt = torch.empty(batch, camera, dtype=torch.int32)
        flat = cnt.reshape(-1)
        for i in range(flat.numel()):
            if i % 4 == 0:
                flat[i] = 0
            elif i % 4 == 1:
                flat[i] = 1
            elif i % 4 == 2:
                flat[i] = n_gauss // 2
            else:
                flat[i] = n_gauss
    else:
        raise ValueError(f"unknown cnt mode: {mode}")
    return cnt


class ColorExposureTestBase(TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.width = 320
        self.height = 240

    def gen_inputs(self, batch, n_gauss, grid_shape=None, exposure_shape='flat',
                   camera_num=1, cnt_mode='full'):
        # 布局约定: [B, C, D, N] = [batch, camera, channel, gauss]
        colors = torch.rand(batch, camera_num, 3, n_gauss, dtype=torch.float32) - 0.5
        means2d = torch.empty(batch, camera_num, 2, n_gauss, dtype=torch.float32)
        means2d[:, :, 0] = torch.rand(batch, camera_num, n_gauss) * self.width * 1.2 - 0.1 * self.width
        means2d[:, :, 1] = torch.rand(batch, camera_num, n_gauss) * self.height * 1.2 - 0.1 * self.height

        fixed_xy = [
            (0.0, 0.0),
            (float(self.width), float(self.height)),
            (float(self.width - 1), float(self.height - 1)),
            (-4.0, -2.0),
            (float(self.width + 3), float(self.height + 1)),
            (float(self.width), 0.0),
            (0.0, float(self.height)),
        ]
        fixed_colors = [
            (0.0, 0.0, 0.0),
            (2.0, 2.0, 2.0),
            (-2.0, -2.0, -2.0),
            (0.5, -0.3, 0.2),
            (0.1, 0.2, 0.3),
            (-0.5, 0.5, 0.25),
            (0.299, 0.587, 0.114),
        ]
        k = min(len(fixed_xy), n_gauss)
        if k > 0:
            fxy = torch.tensor(fixed_xy[:k], dtype=torch.float32).t()
            fcol = torch.tensor(fixed_colors[:k], dtype=torch.float32).t()
            for b in range(batch):
                for c in range(camera_num):
                    means2d[b, c, :, :k] = fxy
                    colors[b, c, :, :k] = fcol

        if grid_shape is None:
            exposure = torch.randn(12, dtype=torch.float32) * 0.1
            if exposure_shape == '4x3':
                exposure = exposure.view(4, 3)
            elif exposure_shape == '1x1x1':
                exposure = exposure.view(1, 1, 1, 12)
        else:
            exposure = torch.randn(*grid_shape, 12, dtype=torch.float32) * 0.1
            if exposure_shape == 'grid5d':
                exposure = exposure.reshape(*grid_shape, 4, 3)
        cnt = _gen_cnt(batch, camera_num, n_gauss, cnt_mode)
        return Inputs(colors, means2d, exposure, cnt, self.width, self.height)

    def gen_v_colors(self, inputs):
        return torch.randn(inputs.colors.shape, dtype=torch.float32) * 0.5

    def npu_fwd(self, inputs, clamp):
        out, index, weight = color_exposure_fwd(
            inputs.colors.npu(), inputs.means2d.npu(), inputs.exposure.npu(),
            inputs.cnt.npu(), inputs.width, inputs.height, 1 if clamp else 0)
        return FwdOut(out.detach().cpu(), index.detach().cpu(), weight.detach().cpu()), \
            (index, weight)

    def npu_bwd(self, inputs, v_colors, index_npu, weight_npu, clamp, calc_exp_grad):
        v_out, v_exp = color_exposure_bwd(
            inputs.colors.npu(), v_colors.npu(), inputs.exposure.npu(),
            inputs.cnt.npu(), index_npu, weight_npu, 1 if clamp else 0,
            1 if calc_exp_grad else 0)
        v_exp = v_exp.detach().cpu() if v_exp.numel() > 0 else v_exp
        return BwdOut(v_out.detach().cpu(), v_exp)


class TestColorExposureFwd(ColorExposureTestBase):
    def _run_fwd(self, inputs, clamp):
        expected = golden_fwd(inputs.colors, inputs.means2d, inputs.exposure, inputs.cnt,
                              inputs.width, inputs.height, clamp)
        actual, _ = self.npu_fwd(inputs, clamp)
        b, c, _, n = inputs.colors.shape
        msg = f"b={b} c={c} n={n} exposure={list(inputs.exposure.shape)} " \
              f"cnt={inputs.cnt.tolist()} clamp={clamp}"
        _assert_close(self, actual.output, expected.output, msg=f"output {msg}")
        if expected.index.numel() == 0:
            self.assertEqual(actual.index.numel(), 0, f"index should be empty {msg}")
            self.assertEqual(actual.weight.numel(), 0, f"weight should be empty {msg}")
        else:
            self.assertTrue(torch.equal(actual.index, expected.index),
                            f"index mismatch {msg}")
            _assert_close(self, actual.weight, expected.weight, msg=f"weight {msg}")

    def test_fwd_global_random(self):
        cases = [
            (1, 1, 4096, 'flat'),
            (1, 2, 123, '4x3'),
            (2, 3, 257, '1x1x1'),  # numel == 12 → 仍走全局路径
            (1, 1, 100000, 'flat'),  # 多 taskLoop 分块
        ]
        for batch, camera, n_gauss, exp_shape in cases:
            for clamp in (False, True):
                with self.subTest(b=batch, c=camera, n=n_gauss, exp=exp_shape, clamp=clamp):
                    inputs = self.gen_inputs(batch, n_gauss, exposure_shape=exp_shape,
                                             camera_num=camera)
                    self._run_fwd(inputs, clamp)

    def test_fwd_grid_random(self):
        cases = [
            (1, 1, 4096, (7, 7, 8)),
            (1, 2, 123, (3, 5, 4)),
            (2, 3, 13, (2, 2, 2)),    # 多 batch/camera + 非 8 对齐 N (尾部 DataCopyPad 路径)
            (1, 1, 7, (5, 4, 3)),     # 极小 N (尾部补齐路径)
            (1, 1, 1, (2, 2, 2)),     # 单高斯 (taskNum < 对齐粒度)
            (1, 1, 100000, (12, 12, 8)),  # 多 taskLoop 分块 + 大网格
        ]
        for batch, camera, n_gauss, grid_shape in cases:
            for clamp in (False, True):
                with self.subTest(b=batch, c=camera, n=n_gauss, grid=grid_shape, clamp=clamp):
                    inputs = self.gen_inputs(batch, n_gauss, grid_shape=grid_shape,
                                             camera_num=camera)
                    self._run_fwd(inputs, clamp)

    def test_fwd_grid_max_16x16x8(self):
        # 最大网格边界: 16×16×8 = 2048 单元 (MAX_GRID_DIMS 硬校验上限)
        for clamp in (False, True):
            with self.subTest(clamp=clamp):
                inputs = self.gen_inputs(2, 517, grid_shape=(16, 16, 8), camera_num=2)
                self._run_fwd(inputs, clamp)

    def test_fwd_cnt_filter(self):
        # cnt 过滤: 每个 (b, c) 只处理前 cnt 个高斯, 其余位置保持 0
        cases = [
            (2, 3, 257, None, 'mixed'),
            (2, 3, 257, (4, 5, 6), 'mixed'),
            (2, 2, 100000, (7, 7, 8), 'mixed'),  # cnt 跨 taskLoop 分块边界
            (2, 2, 123, (3, 3, 4), 'zero'),      # cnt 全 0: 整个输出为 0
            (1, 1, 64, (2, 2, 2), 'mixed'),      # cnt 恰好对齐分块粒度
        ]
        for batch, camera, n_gauss, grid_shape, cnt_mode in cases:
            for clamp in (False, True):
                with self.subTest(b=batch, c=camera, n=n_gauss, grid=grid_shape,
                                  cnt=cnt_mode, clamp=clamp):
                    inputs = self.gen_inputs(batch, n_gauss, grid_shape=grid_shape,
                                             camera_num=camera, cnt_mode=cnt_mode)
                    self._run_fwd(inputs, clamp)

    def test_fwd_grid_5d_exposure(self):
        inputs = self.gen_inputs(2, 512, grid_shape=(4, 5, 6), exposure_shape='grid5d',
                                 camera_num=3)
        self._run_fwd(inputs, clamp=True)

    def test_fwd_boundary(self):
        inputs = self.gen_inputs(2, 64, grid_shape=(7, 7, 8), camera_num=2)
        actual, _ = self.npu_fwd(inputs, clamp=True)
        expected = golden_fwd(inputs.colors, inputs.means2d, inputs.exposure, inputs.cnt,
                              inputs.width, inputs.height, True)
        msg = "boundary"
        _assert_close(self, actual.output, expected.output, msg=f"output {msg}")
        self.assertTrue(torch.equal(actual.index, expected.index), f"index {msg}")
        _assert_close(self, actual.weight, expected.weight, msg=f"weight {msg}")
        gx, gy, gz = 7, 7, 8

        # n=0: (0,0)/colors=0 → 全部首格点, 右权重 0
        self.assertEqual(int(actual.index[0, 0, 0, 0]), 0, msg)
        self.assertEqual(int(actual.index[0, 0, 1, 0]), 0, msg)
        self.assertEqual(int(actual.index[0, 0, 2, 0]), 0, msg)
        _assert_close(self, actual.weight[0, 0, :, 0], torch.zeros(3), msg=f"zero row {msg}")

        # n=1: (W,H)/grey>1 → 三维端点闭包: idx=G-2, 右权重 1
        self.assertEqual(int(actual.index[0, 0, 0, 1]), gx - 2, msg)
        self.assertEqual(int(actual.index[0, 0, 1, 1]), gy - 2, msg)
        self.assertEqual(int(actual.index[0, 0, 2, 1]), gz - 2, msg)
        _assert_close(self, actual.weight[0, 0, :, 1], torch.ones(3), msg=f"endpoint row {msg}")

        # n=3: 负向出界 (-4,-2) 且 grey<0 → clamp01 后全钳到首格点
        self.assertEqual(int(actual.index[0, 0, 0, 3]), 0, msg)
        self.assertEqual(int(actual.index[0, 0, 1, 3]), 0, msg)
        self.assertEqual(int(actual.index[0, 0, 2, 3]), 0, msg)
        _assert_close(self, actual.weight[0, 0, :, 3], torch.zeros(3), msg=f"oob row {msg}")


class TestColorExposureBwd(ColorExposureTestBase):
    def _run_bwd(self, inputs, clamp, calc_exp_grad):
        v_colors = self.gen_v_colors(inputs)
        golden_f = golden_fwd(inputs.colors, inputs.means2d, inputs.exposure, inputs.cnt,
                              inputs.width, inputs.height, clamp)
        expected = golden_bwd(inputs.colors, v_colors, inputs.exposure, inputs.cnt,
                              golden_f.index, golden_f.weight, clamp, calc_exp_grad)
        _, (index_npu, weight_npu) = self.npu_fwd(inputs, clamp)
        actual = self.npu_bwd(inputs, v_colors, index_npu, weight_npu, clamp,
                              calc_exp_grad)
        total = int(inputs.cnt.sum())
        b, c, _, n = inputs.colors.shape
        msg = f"b={b} c={c} n={n} exposure={list(inputs.exposure.shape)} " \
              f"cnt={inputs.cnt.tolist()} clamp={clamp} calc={calc_exp_grad}"

        errors = []
        if not torch.allclose(actual.v_colors, expected.v_colors,
                              rtol=1e-5, atol=1e-5):
            diff = (actual.v_colors - expected.v_colors).abs().max().item()
            errors.append(f"v_colors max_abs_diff={diff:.3e}")
        if calc_exp_grad:
            rtol, atol = _vexp_tolerance(total)
            if not torch.allclose(actual.v_exposure, expected.v_exposure,
                                  rtol=rtol, atol=atol):
                diff = (actual.v_exposure - expected.v_exposure).abs().max().item()
                errors.append(f"v_exposure max_abs_diff={diff:.3e}")
        elif actual.v_exposure.numel() != 0:
            errors.append("v_exposure should be empty when calc_exp_grad=False")
        if errors:
            self.fail(f"{msg}: " + "; ".join(errors))

    def test_bwd_global_random(self):
        cases = [
            (1, 1, 4096, 'flat'),
            (1, 2, 123, '4x3'),
            (2, 3, 257, 'flat'),
        ]
        for batch, camera, n_gauss, exp_shape in cases:
            for clamp in (False, True):
                for calc_exp_grad in (False, True):
                    with self.subTest(b=batch, c=camera, n=n_gauss, exp=exp_shape,
                                      clamp=clamp, calc=calc_exp_grad):
                        inputs = self.gen_inputs(batch, n_gauss,
                                                 exposure_shape=exp_shape,
                                                 camera_num=camera)
                        self._run_bwd(inputs, clamp, calc_exp_grad)

    def test_bwd_grid_random(self):
        cases = [
            (1, 1, 4096, (7, 7, 8)),
            (1, 2, 123, (3, 5, 4)),
            (2, 3, 13, (2, 2, 2)),
        ]
        for batch, camera, n_gauss, grid_shape in cases:
            for clamp in (False, True):
                for calc_exp_grad in (False, True):
                    with self.subTest(b=batch, c=camera, n=n_gauss, grid=grid_shape,
                                      clamp=clamp, calc=calc_exp_grad):
                        inputs = self.gen_inputs(batch, n_gauss, grid_shape=grid_shape,
                                                 camera_num=camera)
                        self._run_bwd(inputs, clamp, calc_exp_grad)

    def test_bwd_grid_max_16x16x8(self):
        inputs = self.gen_inputs(2, 517, grid_shape=(16, 16, 8), camera_num=2)
        for clamp in (False, True):
            with self.subTest(clamp=clamp):
                self._run_bwd(inputs, clamp, True)

    def test_bwd_cnt_filter(self):
        # cnt 过滤: cnt 之外高斯不产生 v_colors / v_exposure 梯度
        cases = [
            (2, 3, 257, (4, 5, 6), 'mixed'),
            (2, 2, 100000, (12, 12, 8), 'mixed'),  # cnt 跨 taskLoop 分块边界
            (2, 2, 123, (3, 3, 4), 'zero'),        # cnt 全 0: 梯度全 0
        ]
        for batch, camera, n_gauss, grid_shape, cnt_mode in cases:
            for clamp in (False, True):
                with self.subTest(b=batch, c=camera, n=n_gauss, grid=grid_shape,
                                  cnt=cnt_mode, clamp=clamp):
                    inputs = self.gen_inputs(batch, n_gauss, grid_shape=grid_shape,
                                             camera_num=camera, cnt_mode=cnt_mode)
                    self._run_bwd(inputs, clamp, True)

    def test_bwd_grid_5d_exposure(self):
        inputs = self.gen_inputs(2, 517, grid_shape=(5, 5, 6), exposure_shape='grid5d',
                                 camera_num=2)
        for clamp in (False, True):
            with self.subTest(clamp=clamp):
                self._run_bwd(inputs, clamp, True)

    def test_bwd_clamp_mask(self):
        inputs = self.gen_inputs(1, 257, grid_shape=(4, 4, 4), camera_num=2)
        grid = inputs.exposure.clone()
        grid[..., 3] = -0.05
        grid[..., 7] = 0.1
        grid[..., 11] = 0.0
        inputs = inputs._replace(exposure=grid)

        v_colors = self.gen_v_colors(inputs)
        golden_f = golden_fwd(inputs.colors, inputs.means2d, inputs.exposure, inputs.cnt,
                              inputs.width, inputs.height, True)
        expected_masked = golden_bwd(inputs.colors, v_colors, inputs.exposure, inputs.cnt,
                                     golden_f.index, golden_f.weight, True, True)
        expected_free = golden_bwd(inputs.colors, v_colors, inputs.exposure, inputs.cnt,
                                   golden_f.index, golden_f.weight, False, True)
        # 掩码确实生效: 两种 clamp 取值下梯度不同
        self.assertFalse(torch.allclose(expected_masked.v_colors,
                                        expected_free.v_colors, rtol=1e-4, atol=1e-4))

        _, (index_npu, weight_npu) = self.npu_fwd(inputs, clamp=True)
        actual = self.npu_bwd(inputs, v_colors, index_npu, weight_npu, True, True)
        _assert_close(self, actual.v_colors, expected_masked.v_colors,
                      msg="clamp-masked v_colors")
        rtol, atol = _vexp_tolerance(int(inputs.cnt.sum()))
        _assert_close(self, actual.v_exposure, expected_masked.v_exposure,
                      rtol=rtol, atol=atol, msg="clamp-masked v_exposure")

    def test_bwd_large(self):
        inputs = self.gen_inputs(1, 100000, grid_shape=(12, 12, 8), camera_num=2)
        self._run_bwd(inputs, clamp=True, calc_exp_grad=True)

    def test_bwd_v_colors_transpose(self):
        k = 0.7
        for grid_shape in (None, (4, 5, 6)):
            for basis_idx, src, dst in ((1, 0, 1), (8, 2, 0)):
                # basis_idx = 4*i+j: A[i][j]=k → v_{c_j}(dst) = k·v_i(src)
                with self.subTest(grid=grid_shape, basis=basis_idx):
                    inputs = self.gen_inputs(2, 257, grid_shape=grid_shape, camera_num=2)
                    if grid_shape is None:
                        exposure = torch.zeros(12, dtype=torch.float32)
                        exposure[basis_idx] = k
                    else:
                        exposure = torch.zeros(*grid_shape, 12, dtype=torch.float32)
                        exposure[..., basis_idx] = k
                    inputs = inputs._replace(exposure=exposure)

                    v_colors = self.gen_v_colors(inputs)
                    expected = torch.zeros_like(v_colors)
                    expected[:, :, dst, :] = k * v_colors[:, :, src, :]

                    golden_f = golden_fwd(inputs.colors, inputs.means2d, exposure,
                                          inputs.cnt, inputs.width, inputs.height, False)
                    gold = golden_bwd(inputs.colors, v_colors, exposure, inputs.cnt,
                                      golden_f.index, golden_f.weight, False, False)
                    # golden 自检: A^T·v 与显式基矩阵模式一致
                    _assert_close(self, gold.v_colors, expected,
                                   msg=f"golden grid={grid_shape} basis={basis_idx}")

                    _, (index_npu, weight_npu) = self.npu_fwd(inputs, False)
                    actual = self.npu_bwd(inputs, v_colors, index_npu, weight_npu,
                                          False, False)
                    _assert_close(self, actual.v_colors, expected,
                                  msg=f"npu grid={grid_shape} basis={basis_idx}")


class TestColorExposureAutograd(ColorExposureTestBase):
    def _run_autograd(self, inputs, clamp, calc_exp_grad):
        v_colors = self.gen_v_colors(inputs)
        golden_f = golden_fwd(inputs.colors, inputs.means2d, inputs.exposure, inputs.cnt,
                              inputs.width, inputs.height, clamp)
        expected_bwd = golden_bwd(inputs.colors, v_colors, inputs.exposure, inputs.cnt,
                                  golden_f.index, golden_f.weight, clamp,
                                  calc_exp_grad)
        total = int(inputs.cnt.sum())
        msg = f"exposure={list(inputs.exposure.shape)} clamp={clamp} calc={calc_exp_grad}"

        colors_npu = inputs.colors.npu().requires_grad_()
        exposure_npu = inputs.exposure.npu().requires_grad_()
        out = color_exposure(colors_npu, inputs.means2d.npu(), exposure_npu,
                             inputs.cnt.npu(), inputs.width, inputs.height, clamp,
                             calc_exp_grad)
        _assert_close(self, out.detach().cpu(), golden_f.output, msg=f"output {msg}")
        out.backward(v_colors.npu())

        _assert_close(self, colors_npu.grad.cpu(), expected_bwd.v_colors,
                      msg=f"colors.grad {msg}")
        if calc_exp_grad:
            rtol, atol = _vexp_tolerance(total)
            _assert_close(self, exposure_npu.grad.cpu(), expected_bwd.v_exposure,
                          rtol=rtol, atol=atol, msg=f"exposure.grad {msg}")
        else:
            self.assertIsNone(exposure_npu.grad, f"exposure.grad should be None {msg}")

    def test_autograd_function(self):
        for clamp in (False, True):
            for calc_exp_grad in (False, True):
                with self.subTest(clamp=clamp, calc=calc_exp_grad):
                    inputs = self.gen_inputs(2, 257, grid_shape=(7, 7, 8), camera_num=3)
                    self._run_autograd(inputs, clamp, calc_exp_grad)

    def test_autograd_global(self):
        inputs = self.gen_inputs(1, 513, exposure_shape='flat', camera_num=2)
        self._run_autograd(inputs, clamp=True, calc_exp_grad=True)

    def test_autograd_cnt_filter(self):
        # 端到端验证 cnt 过滤语义: 无效高斯位置输出 0、梯度 0
        inputs = self.gen_inputs(2, 257, grid_shape=(7, 7, 8), camera_num=3,
                                 cnt_mode='mixed')
        self._run_autograd(inputs, clamp=True, calc_exp_grad=True)

    def test_autograd_grad_contract(self):
        inputs = self.gen_inputs(2, 128, grid_shape=(4, 5, 6),
                                 exposure_shape='grid5d', camera_num=2)
        v = torch.randn(2, 2, 128, 3).npu().permute(0, 1, 3, 2)  # 非连续 [B,C,3,N]
        self.assertFalse(v.is_contiguous())
        v_cpu = v.contiguous().cpu()
        golden_f = golden_fwd(inputs.colors, inputs.means2d, inputs.exposure, inputs.cnt,
                              inputs.width, inputs.height, True)

        colors_npu = inputs.colors.npu().requires_grad_()
        exposure_npu = inputs.exposure.npu().requires_grad_()
        out = color_exposure(colors_npu, inputs.means2d.npu(), exposure_npu,
                             inputs.cnt.npu(), inputs.width, inputs.height, True, False)
        out.backward(v)
        self.assertIsNone(exposure_npu.grad, "exposure.grad should be None")
        expected = golden_bwd(inputs.colors, v_cpu, inputs.exposure, inputs.cnt,
                              golden_f.index, golden_f.weight, True, False)
        _assert_close(self, colors_npu.grad.cpu(), expected.v_colors,
                      msg="non-contiguous grad v_colors")

        colors_npu = inputs.colors.npu().requires_grad_()
        exposure_npu = inputs.exposure.npu().requires_grad_()
        out = color_exposure(colors_npu, inputs.means2d.npu(), exposure_npu,
                             inputs.cnt.npu(), inputs.width, inputs.height, True, True)
        out.backward(v)
        self.assertEqual(exposure_npu.grad.shape, inputs.exposure.shape,
                         "exposure.grad should keep exposure layout")
        expected = golden_bwd(inputs.colors, v_cpu, inputs.exposure, inputs.cnt,
                              golden_f.index, golden_f.weight, True, True)
        _assert_close(self, colors_npu.grad.cpu(), expected.v_colors,
                      msg="calc=True v_colors")
        rtol, atol = _vexp_tolerance(int(inputs.cnt.sum()))
        _assert_close(self, exposure_npu.grad.cpu(), expected.v_exposure,
                      rtol=rtol, atol=atol, msg="calc=True v_exposure")

    def test_wrapper_input_validation(self):
        colors = torch.rand(1, 3, 16)
        means2d = torch.rand(1, 2, 2, 16)
        exposure = torch.rand(12)
        cnt = torch.full((1, 2), 16, dtype=torch.int32)
        with self.assertRaises(ValueError):
            color_exposure(colors, means2d, exposure, cnt, self.width, self.height)
        with self.assertRaises(ValueError):
            color_exposure(colors.unsqueeze(2), means2d[0], exposure, cnt,
                           self.width, self.height)

    def test_autograd_vs_torch_grad(self):
        errors = []
        for grid_shape in (None, (4, 5, 6)):
            inputs = self.gen_inputs(2, 333, grid_shape=grid_shape, camera_num=2,
                                     cnt_mode='mixed')
            clamp = True
            v_colors = self.gen_v_colors(inputs)

            colors_ref = inputs.colors.clone().requires_grad_()
            exposure_ref = inputs.exposure.clone().requires_grad_()
            out_ref = golden_fwd(colors_ref, inputs.means2d, exposure_ref, inputs.cnt,
                                 inputs.width, inputs.height, clamp).output
            grad_colors, grad_exposure = torch.autograd.grad(
                out_ref, (colors_ref, exposure_ref), v_colors)

            colors_npu = inputs.colors.npu().requires_grad_()
            exposure_npu = inputs.exposure.npu().requires_grad_()
            out = color_exposure(colors_npu, inputs.means2d.npu(), exposure_npu,
                                 inputs.cnt.npu(), inputs.width, inputs.height, clamp, True)
            out.backward(v_colors.npu())

            _assert_close(self, out.detach().cpu(), out_ref.detach(),
                          msg=f"output grid={grid_shape}")
            if not torch.allclose(colors_npu.grad.cpu(), grad_colors,
                                  rtol=1e-5, atol=1e-5):
                diff = (colors_npu.grad.cpu() - grad_colors).abs().max().item()
                errors.append(f"grid={grid_shape}: colors.grad max_abs_diff={diff:.3e} "
                              f"(期望 A^T·v)")
            rtol, atol = _vexp_tolerance(int(inputs.cnt.sum()))
            if not torch.allclose(exposure_npu.grad.cpu(), grad_exposure,
                                  rtol=rtol, atol=atol):
                diff = (exposure_npu.grad.cpu() - grad_exposure).abs().max().item()
                errors.append(f"grid={grid_shape}: exposure.grad max_abs_diff={diff:.3e}")
        if errors:
            self.fail("; ".join(errors))


if __name__ == "__main__":
    run_tests()
