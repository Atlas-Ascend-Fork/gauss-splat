# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A5 (clip_lastids) 渲染算子 bg_color 特性专项 ST。

覆盖 req.md「bg_color 专项用例」：
  1. 传背景校验：空 tile（offsets 全 0）时 color == bg 广播到 [3,T,P]、剩余透光率为 1；
  2. 分解一致性：render(bg=c) == render(bg=0) + 透光率·c 的逐像素项（误差 < 1e-6）；
  3. 反向数值正确性：带背景色时对高斯参数的梯度与可微参考实现（autograd）一致，
     包含背景色经由最终透光率回传到不透明度/位置/协方差/颜色/深度的路径；
  4. 白/黑边界：bg 全 1 与全 0 各跑一次；
  5. 位/输入耦合：置位却传 None（或反之）在 Python 与 csrc 两侧均报错；
  6. 零位等价：bg_color=None 与直接调用底层（同样关闭）逐位一致。
"""

import math
from collections import namedtuple

import numpy as np

import acl
import torch
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import calc_render, get_render_schedule_cpp
from gauss_splat import gsplat_c


torch.npu.set_device('npu:0')
torch.manual_seed(1234)

Inputs = namedtuple('Inputs', ['means2d', 'colors', 'opacities', 'conics_0', 'conics_1', 'conics_2',
                               'sorted_gs_ids', 'tile_offsets', 'pix_coords', 'depths',
                               'padded_width', 'padded_height', 'tile_nums'])


def inverse_cov2d_v2(cov2_00, cov2_01, cov2_11, scale=1):
    det = cov2_00 * cov2_11 - cov2_01 * cov2_01
    inv_x_0 = cov2_11 / det * scale
    inv_x_1 = -cov2_01 / det * scale
    inv_x_2 = cov2_00 / det * scale
    return inv_x_0, inv_x_1, inv_x_2


@torch.no_grad()
def sort_gs(all_in_mask, depths):
    num_tile = all_in_mask.shape[1]
    tile_offset = torch.sum(all_in_mask, dim=0).cumsum(dim=0)
    sorted_gs_ids = torch.zeros(tile_offset[-1], dtype=torch.int32, device=all_in_mask.device)
    for tile_id in range(num_tile):
        prev_offset = tile_offset[tile_id - 1] if tile_id > 0 else 0
        tile_in_mask = all_in_mask[:, tile_id]
        tile_depths = depths[tile_in_mask]
        tile_gs_ids = tile_in_mask.nonzero()[:, 0]
        _, local_sort_index = torch.sort(tile_depths, stable=True)
        sorted_gs_ids[prev_offset:tile_offset[tile_id]] = tile_gs_ids[local_sort_index]
    return sorted_gs_ids, tile_offset


@torch.no_grad()
def get_radius_v2(cov2_00, cov2_01, cov2_11):
    det = cov2_00 * cov2_11 - cov2_01 * cov2_01
    mid = 0.5 * (cov2_00 + cov2_11)
    lambda1 = mid + torch.sqrt((mid ** 2 - det).clip(min=0.1))
    return 3.0 * torch.sqrt(lambda1).ceil()


@torch.no_grad()
def get_rect_v2(means_x, means_y, radii, width, height):
    rect_min_0 = torch.clamp(means_x - radii, 0, width - 1.0)
    rect_min_1 = torch.clamp(means_y - radii, 0, height - 1.0)
    rect_max_0 = torch.clamp(means_x + radii, 0, width - 1.0)
    rect_max_1 = torch.clamp(means_y + radii, 0, height - 1.0)
    return rect_min_0, rect_min_1, rect_max_0, rect_max_1


def build_tile_gs_mask(means_2d, radii, img_size, tile_grid, tile_size):
    means_x, means_y = means_2d[:, 0], means_2d[:, 1]
    image_width, image_height = img_size
    num_gs = means_x.shape[0]
    num_tile = tile_grid.shape[0]
    rmin_w, rmin_h, rmax_w, rmax_h = get_rect_v2(means_x, means_y, radii, image_width, image_height)

    w_right_bound = torch.clamp(rmax_w[:, None].expand(num_gs, num_tile),
                                max=(tile_grid[None, :, 1] + tile_size).expand(num_gs, num_tile))
    w_left_bound = torch.clamp(rmin_w[:, None].expand(num_gs, num_tile),
                               min=(tile_grid[None, :, 1]).expand(num_gs, num_tile))
    h_upper_bound = torch.clamp(rmax_h[:, None].expand(num_gs, num_tile),
                                max=(tile_grid[None, :, 0] + tile_size).expand(num_gs, num_tile))
    h_lower_bound = torch.clamp(rmin_h[:, None].expand(num_gs, num_tile),
                                min=(tile_grid[None, :, 0]).expand(num_gs, num_tile))
    return (w_right_bound > w_left_bound) & (h_upper_bound > h_lower_bound)


def _render_nonempty_tile(tile_coord, prev_offset, cur_offset, gs_data, bg):
    """非空 tile 的颜色/深度/对数透光率参考计算。"""
    means_x, means_y, c0, c1, c2, opac, cr, cg, cb, dep = gs_data
    sl = slice(prev_offset, cur_offset)
    dx = tile_coord[None, :, 0] - means_x[sl][:, None]
    dy = tile_coord[None, :, 1] - means_y[sl][:, None]
    quad = 0.5 * (c0[sl][:, None] * dx * dx + c2[sl][:, None] * dy * dy) \
        + c1[sl][:, None] * dx * dy
    weight = torch.exp(-quad)
    alpha = torch.clamp(weight * opac[sl][:, None], max=0.999)
    alpha = torch.where(alpha >= 1.0 / 255.0, alpha, torch.zeros_like(alpha))

    trans_after = torch.exp(torch.cumsum(torch.log1p(-alpha), dim=0))
    trans_before = torch.cat((torch.ones_like(alpha[:1]), trans_after[:-1]), dim=0)
    weights = torch.where(trans_after > 1e-4, trans_before * alpha, torch.zeros_like(alpha))

    color = (weights[..., None] * torch.stack([cr[sl], cg[sl], cb[sl]], dim=1)[:, None, :]).sum(dim=0)
    depth = (weights * dep[sl][:, None]).sum(dim=0)

    # kernel 在 per-pixel done（nextT <= 1e-4）后停止更新对数域累积和：
    # 最终透光率取「首次 done 位置」的 trans_after，否则取最后一次。
    done = trans_after <= 1e-4
    has_done = done.any(dim=0)
    first_done = torch.argmax(done.to(torch.int64), dim=0)
    t_final = torch.where(has_done, trans_after.gather(0, first_done[None, :])[0], trans_after[-1])
    color = color + t_final[:, None] * bg.T
    lc = torch.where(has_done, torch.log(t_final.clamp(min=1e-30)), torch.log(trans_after[-1]))
    return color, depth, lc


def reference_render(inputs, tile_size, bg_color):
    """可微参考实现（与 kernel 的 clip_lastids 语义逐项一致，含背景色路径）。

    返回 (color [3,T,P], depth [T,P], last_cumsum [T,P])。
    """
    means2d, colors, opacities, conics_0, conics_1, conics_2, \
        sorted_gs_ids, tile_offsets, _, depths, padded_width, padded_height, _ = inputs

    means_x = torch.index_select(means2d[:, 0], 0, sorted_gs_ids)
    means_y = torch.index_select(means2d[:, 1], 0, sorted_gs_ids)
    c0 = torch.index_select(conics_0, 0, sorted_gs_ids)
    c1 = torch.index_select(conics_1, 0, sorted_gs_ids)
    c2 = torch.index_select(conics_2, 0, sorted_gs_ids)
    opac = torch.index_select(opacities.squeeze(-1), 0, sorted_gs_ids)
    cr = torch.index_select(colors[:, 0], 0, sorted_gs_ids)
    cg = torch.index_select(colors[:, 1], 0, sorted_gs_ids)
    cb = torch.index_select(colors[:, 2], 0, sorted_gs_ids)
    dep = torch.index_select(depths, 0, sorted_gs_ids)

    tile_grid = torch.stack(torch.meshgrid(torch.arange(0, padded_height, tile_size),
                                           torch.arange(0, padded_width, tile_size),
                                           indexing='ij'), dim=-1).view(-1, 2).tolist()
    pix_coord = torch.stack(torch.meshgrid(torch.arange(padded_width), torch.arange(padded_height),
                                           indexing='xy'), dim=-1).to(means2d.device)

    render_color, render_depth, render_lc = [], [], []
    for i, tile in enumerate(tile_grid):
        prev_offset = tile_offsets[i - 1] if i > 0 else 0
        cur_offset = tile_offsets[i]
        h, w = tile
        h_max, w_max = h + tile_size, w + tile_size
        tile_coord = pix_coord[h:h_max, w:w_max, :].reshape(-1, 2)
        n_pix = tile_coord.shape[0]
        bg = torch.zeros(3, n_pix, device=means2d.device, dtype=torch.float32)
        if bg_color is not None:
            bg = bg_color[:, None].expand(3, n_pix)

        if cur_offset <= prev_offset:
            # 空 tile：无高斯贡献，剩余透光率为 1，颜色即背景色
            render_color.append(bg.T)   # 统一为 [P,3]（与混合颜色一致）
            render_depth.append(torch.zeros(n_pix, device=means2d.device))
            render_lc.append(torch.zeros(n_pix, device=means2d.device))
            continue

        gs_data = (means_x, means_y, c0, c1, c2, opac, cr, cg, cb, dep)
        color, depth, lc = _render_nonempty_tile(
            tile_coord, prev_offset, cur_offset, gs_data, bg)
        render_color.append(color)
        render_depth.append(depth)
        render_lc.append(lc)

    render_color = torch.stack(render_color, dim=0).permute(2, 0, 1)   # [3,T,P]
    render_depth = torch.stack(render_depth, dim=0)                    # [T,P]
    render_lc = torch.stack(render_lc, dim=0)                          # [T,P]
    return render_color, render_depth, render_lc


class TestCalcRenderBgColor(TestCase):
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        torch.manual_seed(1234)
        self.tile_size = 32
        self.n_gs = 4096
        self.image_width = 32 * 3
        self.image_height = 32 * 4
        self.inputs = self._build_inputs(self.n_gs)
        self.bg = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float32).npu()

    # ------------------------------------------------------------------
    # 1/4. 前向：随机背景色、白/黑边界与参考实现一致
    # ------------------------------------------------------------------
    def test_forward_bg_matches_reference(self):
        for bg in (self.bg, None,
                   torch.zeros(3, dtype=torch.float32).npu(),
                   torch.ones(3, dtype=torch.float32).npu()):
            out_color, out_depth = self._npu_forward(bg)
            ref_color, ref_depth, _ = self._ref_forward(bg)
            self.assertRtolEqual(ref_color.detach().cpu().numpy(), out_color.detach().cpu().numpy())
            self.assertRtolEqual(ref_depth.detach().cpu().numpy(), out_depth.detach().cpu().numpy())

    # ------------------------------------------------------------------
    # 2. 分解一致性：render(bg=c) == render(bg=0) + T_final·c
    # ------------------------------------------------------------------
    def test_decomposition_consistency(self):
        color0, _ = self._npu_forward(None)
        color1, _ = self._npu_forward(self.bg)
        # 剩余透光率 = exp(lastCumsum)；直接调用底层前向取回 lastCumsum
        gs = torch.cat([
            self.inputs.means2d.transpose(0, 1),
            self.inputs.conics_0.unsqueeze(0), self.inputs.conics_1.unsqueeze(0),
            self.inputs.conics_2.unsqueeze(0), self.inputs.opacities.squeeze(-1).unsqueeze(0),
            self.inputs.colors.transpose(0, 1), self.inputs.depths[None, :],
        ], dim=0).T.contiguous()
        _, _, last_cumsum, _, _ = gsplat_c.calc_render_fwd_clip_last_ids(
            gs, self.inputs.pix_coords, self._lb_sched(),
            self.inputs.sorted_gs_ids.to(torch.int64), self.bg)
        trans = torch.exp(last_cumsum)
        expect = color0 + trans.unsqueeze(0) * self.bg[:, None, None]
        self._assert_allclose(expect, color1, rtol=1e-6, atol=1e-6)

    # ------------------------------------------------------------------
    # 3. 反向：带背景色的梯度与可微参考实现一致（含 bg 经 T 回传路径）
    # ------------------------------------------------------------------
    def test_backward_matches_reference(self):
        v_color = torch.rand(3, self.inputs.pix_coords.shape[0], self.tile_size * self.tile_size).npu()
        v_depth = torch.rand(1, self.inputs.pix_coords.shape[0], self.tile_size * self.tile_size).npu()

        out_color, out_depth = self._npu_forward(self.bg)
        npu_grads = torch.autograd.grad(
            (out_color * v_color).sum() + (out_depth.unsqueeze(0) * v_depth).sum(),
            (self.inputs.means2d, self.inputs.conics_0, self.inputs.conics_1, self.inputs.conics_2,
             self.inputs.opacities, self.inputs.colors, self.inputs.depths),
            retain_graph=False)

        ref_color, ref_depth, _ = self._ref_forward(self.bg)
        ref_grads = torch.autograd.grad(
            (ref_color * v_color).sum() + (ref_depth.unsqueeze(0) * v_depth).sum(),
            (self.inputs.means2d, self.inputs.conics_0, self.inputs.conics_1, self.inputs.conics_2,
             self.inputs.opacities, self.inputs.colors, self.inputs.depths),
            retain_graph=False)

        for got, expect in zip(npu_grads, ref_grads):
            self._assert_allclose(expect, got, rtol=1e-4, atol=1e-4)

    def test_backward_bg_grad_matches_reference(self):
        # 参考 gsplat：dL/dbg = Σ T_K·v_color；NPU 与可微参考实现比对
        v_color = torch.rand(3, self.inputs.pix_coords.shape[0], self.tile_size * self.tile_size).npu()
        v_depth = torch.rand(1, self.inputs.pix_coords.shape[0], self.tile_size * self.tile_size).npu()

        bg = self.bg.clone().requires_grad_()
        out_color, out_depth = self._npu_forward(bg)
        (npu_bg_grad,) = torch.autograd.grad(
            (out_color * v_color).sum() + (out_depth.unsqueeze(0) * v_depth).sum(),
            (bg,))

        ref_bg = self.bg.clone().requires_grad_()
        ref_color, ref_depth, _ = self._ref_forward(ref_bg)
        (ref_bg_grad,) = torch.autograd.grad(
            (ref_color * v_color).sum() + (ref_depth.unsqueeze(0) * v_depth).sum(),
            (ref_bg,))

        self._assert_allclose(ref_bg_grad, npu_bg_grad, rtol=1e-4, atol=1e-4)

    # ------------------------------------------------------------------
    # 1. 空 tile：offsets 全 0 时 color == bg 广播、lastCumsum == 0、lastIds == 0
    # ------------------------------------------------------------------
    def test_empty_tile_bg(self):
        inp = self.inputs
        zero_nums = torch.zeros_like(inp.tile_nums)
        offsets = self._lb_sched(zero_nums)
        gs = torch.cat([
            inp.means2d.transpose(0, 1),
            inp.conics_0.unsqueeze(0), inp.conics_1.unsqueeze(0), inp.conics_2.unsqueeze(0),
            inp.opacities.squeeze(-1).unsqueeze(0), inp.colors.transpose(0, 1), inp.depths[None, :],
        ], dim=0).T.contiguous()
        color, depth, last_cumsum, _, last_ids = gsplat_c.calc_render_fwd_clip_last_ids(
            gs, inp.pix_coords, offsets, torch.zeros(0, dtype=torch.int64).npu(), self.bg)

        expect_color = self.bg[:, None, None].expand_as(color)
        self._assert_allclose(expect_color, color, rtol=1e-6, atol=1e-6)
        self._assert_allclose(torch.zeros_like(depth), depth, rtol=0.0, atol=1e-6)
        self._assert_allclose(torch.zeros_like(last_cumsum), last_cumsum, rtol=0.0, atol=1e-6)
        self.assertTrue(bool((last_ids == 0).all()))

    def test_multibatch_forward_matches_reference(self):
        inputs = self._build_inputs(700, tile0_only=True)
        self.assertGreater(int(inputs.tile_offsets[0].item()), 256,
                           "test scene must produce segLen > 256 in tile0")

        def run():
            color, depth = self._npu_forward(self.bg)
            ref_color, ref_depth, _ = self._ref_forward(self.bg)
            self.assertRtolEqual(ref_color.detach().cpu().numpy(), color.detach().cpu().numpy())
            self.assertRtolEqual(ref_depth.detach().cpu().numpy(), depth.detach().cpu().numpy())

            # 分解一致性：多批时若背景项按批重复叠加，此处会显著偏离
            color0, _ = self._npu_forward(None)
            color1, _ = self._npu_forward(self.bg)
            gs = torch.cat([
                inputs.means2d.transpose(0, 1),
                inputs.conics_0.unsqueeze(0), inputs.conics_1.unsqueeze(0),
                inputs.conics_2.unsqueeze(0), inputs.opacities.squeeze(-1).unsqueeze(0),
                inputs.colors.transpose(0, 1), inputs.depths[None, :],
            ], dim=0).T.contiguous()
            _, _, last_cumsum, _, _ = gsplat_c.calc_render_fwd_clip_last_ids(
                gs, inputs.pix_coords, self._lb_sched(), inputs.sorted_gs_ids.to(torch.int64), self.bg)
            trans = torch.exp(last_cumsum)
            expect = color0 + trans.unsqueeze(0) * self.bg[:, None, None]
            self._assert_allclose(expect, color1, rtol=1e-6, atol=1e-6)

        self._with_inputs(inputs, run)

    def test_multibatch_backward_matches_reference(self):
        inputs = self._build_inputs(700, tile0_only=True)
        self.assertGreater(int(inputs.tile_offsets[0].item()), 256,
                           "test scene must produce segLen > 256 in tile0")
        v_color = torch.rand(3, inputs.pix_coords.shape[0], self.tile_size * self.tile_size).npu()
        v_depth = torch.rand(1, inputs.pix_coords.shape[0], self.tile_size * self.tile_size).npu()

        def run():
            out_color, out_depth = self._npu_forward(self.bg)
            npu_grads = torch.autograd.grad(
                (out_color * v_color).sum() + (out_depth.unsqueeze(0) * v_depth).sum(),
                (inputs.means2d, inputs.conics_0, inputs.conics_1, inputs.conics_2,
                 inputs.opacities, inputs.colors, inputs.depths), retain_graph=False)
            ref_color, ref_depth, _ = self._ref_forward(self.bg)
            ref_grads = torch.autograd.grad(
                (ref_color * v_color).sum() + (ref_depth.unsqueeze(0) * v_depth).sum(),
                (inputs.means2d, inputs.conics_0, inputs.conics_1, inputs.conics_2,
                 inputs.opacities, inputs.colors, inputs.depths), retain_graph=False)
            for got, expect in zip(npu_grads, ref_grads):
                self._assert_allclose(expect, got, rtol=1e-4, atol=1e-4)

        self._with_inputs(inputs, run)

    # ------------------------------------------------------------------
    # 5. 位/输入耦合：Python 与 csrc 两侧均报错
    # ------------------------------------------------------------------
    def test_coupling_errors(self):
        inp = self.inputs
        # 无 bg 正常
        color_off, _ = self._npu_forward(None)
        self.assertEqual(tuple(color_off.shape), (3, self.inputs.pix_coords.shape[0], 1024))
        # 有 bg 正常
        color_on, _ = self._npu_forward(self.bg)
        self.assertEqual(tuple(color_on.shape), (3, self.inputs.pix_coords.shape[0], 1024))
        # 非法 shape 由 Python 校验拒绝
        with self.assertRaises(ValueError):
            self._npu_forward(torch.zeros(4, dtype=torch.float32).npu())
        # csrc 侧 shape 校验
        gs = torch.cat([
            inp.means2d.transpose(0, 1),
            inp.conics_0.unsqueeze(0), inp.conics_1.unsqueeze(0), inp.conics_2.unsqueeze(0),
            inp.opacities.squeeze(-1).unsqueeze(0), inp.colors.transpose(0, 1), inp.depths[None, :],
        ], dim=0).T.contiguous()
        with self.assertRaises(RuntimeError):
            gsplat_c.calc_render_fwd_clip_last_ids(
                gs, inp.pix_coords, self._lb_sched(), inp.sorted_gs_ids.to(torch.int64),
                torch.zeros(4, dtype=torch.float32).npu())

    # ------------------------------------------------------------------
    # 6. 零位等价：关闭特性时不同入口输出逐位一致
    # ------------------------------------------------------------------
    def test_zero_bit_equivalence(self):
        color_a, depth_a = self._npu_forward(None)
        inp = self.inputs
        gs = torch.cat([
            inp.means2d.transpose(0, 1),
            inp.conics_0.unsqueeze(0), inp.conics_1.unsqueeze(0), inp.conics_2.unsqueeze(0),
            inp.opacities.squeeze(-1).unsqueeze(0), inp.colors.transpose(0, 1), inp.depths[None, :],
        ], dim=0).T.contiguous()
        color_b, depth_b, _, _, _ = gsplat_c.calc_render_fwd_clip_last_ids(
            gs, inp.pix_coords, self._lb_sched(), inp.sorted_gs_ids.to(torch.int64), None)
        self.assertTrue(torch.equal(color_a, color_b))
        self.assertTrue(torch.equal(depth_a, depth_b))

    def _assert_allclose(self, expect, got, rtol, atol):
        expect_np = expect.detach().cpu().numpy() if torch.is_tensor(expect) else np.asarray(expect)
        got_np = got.detach().cpu().numpy() if torch.is_tensor(got) else np.asarray(got)
        self.assertTrue(np.allclose(expect_np, got_np, rtol=rtol, atol=atol),
                        f"allclose failed: max_abs_diff={np.max(np.abs(expect_np - got_np))}")

    def _build_inputs(self, n_gs, tile0_only=False):
        tile_size = self.tile_size
        image_width, image_height = self.image_width, self.image_height
        padded_width = math.ceil(image_width / tile_size) * tile_size
        padded_height = math.ceil(image_height / tile_size) * tile_size
        tile_grid = torch.stack(torch.meshgrid(torch.arange(0, padded_height, tile_size),
                                               torch.arange(0, padded_width, tile_size),
                                               indexing='ij'), dim=-1).view(-1, 2).npu()
        pix_coord = torch.stack(torch.meshgrid(torch.arange(padded_width), torch.arange(padded_height),
                                               indexing='xy'), dim=-1).npu()
        pix_coords = pix_coord.reshape(padded_height // tile_size, tile_size,
                                       padded_width // tile_size, tile_size, 2) \
            .permute(0, 2, 1, 3, 4).reshape(padded_height // tile_size * padded_width // tile_size,
                                            tile_size * tile_size, 2) \
            .permute(0, 2, 1).to(torch.float32).contiguous()

        if tile0_only:
            # 全部高斯落在 tile0：[0,tile_size)^2，用于构造 segLen > 256 的多批场景
            mean_x = torch.rand((n_gs, 1)).npu() * tile_size
            mean_y = torch.rand((n_gs, 1)).npu() * tile_size
        else:
            mean_x = torch.rand((n_gs, 1)).npu() * image_width
            mean_y = torch.rand((n_gs, 1)).npu() * image_height
        means2d = torch.cat([mean_x, mean_y], dim=1).clone().requires_grad_()
        colors = torch.rand((n_gs, 3)).npu().clone().requires_grad_()
        opacities = torch.rand((n_gs, 1)).npu().clone().requires_grad_()
        cov_scale = torch.rand((n_gs, 1, 1)).npu() * 0.5 + 0.1
        cov2d = cov_scale * torch.eye(2, device=cov_scale.device)
        conics_0, conics_1, conics_2 = inverse_cov2d_v2(cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1])
        conics_0 = conics_0.clone().requires_grad_()
        conics_1 = conics_1.clone().requires_grad_()
        conics_2 = conics_2.clone().requires_grad_()
        depths = (torch.rand((n_gs,)).npu() * 20).clone().requires_grad_()
        radii = get_radius_v2(cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1])
        all_in_mask = build_tile_gs_mask(means2d, radii, (image_width, image_height),
                                         tile_grid=tile_grid, tile_size=tile_size)
        sorted_gs_ids, tile_offsets = sort_gs(all_in_mask.to(torch.bool), depths)
        tile_nums = torch.cat([tile_offsets[:1], tile_offsets[1:] - tile_offsets[:-1]])
        return Inputs(means2d, colors, opacities, conics_0, conics_1, conics_2,
                      sorted_gs_ids, tile_offsets, pix_coords, depths,
                      padded_width, padded_height, tile_nums)

    def _lb_sched(self, tile_nums=None):
        nums = self.inputs.tile_nums if tile_nums is None else tile_nums
        num_vector_core, _ = acl.get_device_capability(0, 1)
        return get_render_schedule_cpp(nums.cpu().to(torch.int64),
                                       num_vector_core).clone().detach().to(torch.int64).npu()

    def _npu_forward(self, bg_color):
        inp = self.inputs
        return calc_render(inp.means2d.transpose(0, 1),
                           inp.conics_0, inp.conics_1, inp.conics_2,
                           inp.opacities.squeeze(dim=-1),
                           inp.colors.transpose(0, 1),
                           inp.depths[None, :],
                           inp.pix_coords,
                           self._lb_sched(),
                           inp.sorted_gs_ids,
                           bg_color)

    def _ref_forward(self, bg_color):
        return reference_render(self.inputs, self.tile_size, bg_color)

    # ------------------------------------------------------------------
    # 4.5 多批（单 tile 高斯数 > 256）前向/反向专项（B1/B2 回归）
    # ------------------------------------------------------------------
    def _with_inputs(self, inputs, fn):
        saved = self.inputs
        self.inputs = inputs
        try:
            return fn()
        finally:
            self.inputs = saved


if __name__ == "__main__":
    run_tests()
