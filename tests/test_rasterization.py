# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import gc

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import Rasterizer, gsplat_c

torch.npu.set_device('npu:0')
torch.manual_seed(42)
np.set_printoptions(threshold=np.inf)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)


def _assert_shape(test_instance, tensor, expected_shape, msg=""):
    actual = tuple(tensor.shape)
    if actual != expected_shape:
        raise AssertionError(f"{msg}: expected {expected_shape}, got {actual}")


def _gen_splats(n_gs, sh_degree, device='npu'):
    means = (torch.rand(n_gs, 3, dtype=torch.float32) - 0.5) * 0.3
    quats = torch.randn(n_gs, 4, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.log(torch.ones(n_gs, 3, dtype=torch.float32) * 0.01)
    opacities = torch.logit(torch.full((n_gs,), 0.5, dtype=torch.float32))

    k = (sh_degree + 1) ** 2
    sh0 = torch.zeros(n_gs, 1, 3, dtype=torch.float32)
    shN = torch.randn(n_gs, k - 1, 3, dtype=torch.float32) * 0.01

    return {
        "means": means.to(device),
        "quats": quats.to(device),
        "scales": scales.to(device),
        "opacities": opacities.to(device),
        "sh0": sh0.to(device),
        "shN": shN.to(device),
    }


def _gen_camera(width, height, device='npu'):
    camtoworlds = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    camtoworlds[0, 2, 3] = 4.0

    fx = fy = 50.0
    cx, cy = width / 2.0, height / 2.0
    Ks = torch.tensor([[[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]]], dtype=torch.float32, device=device)

    render_mode = "RGB"
    return (camtoworlds, Ks, render_mode)


def _golden_rasterization(splats, cam, size, tile_size, active_sh_degree, camera_model):
    width, height = size
    camtoworlds, Ks, render_mode = cam
    viewmats = torch.linalg.inv(camtoworlds)

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors_sh = torch.cat([splats["sh0"], splats["shN"]], 1)

    C = viewmats.shape[0]
    N = means.shape[0]
    B = 1

    rays_o = camtoworlds[0, :3, 3]
    rays_d = means - rays_o
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    k = (active_sh_degree + 1) ** 2
    shs = colors_sh.unsqueeze(0).expand(C, -1, -1, -1)
    from gauss_splat import spherical_harmonics
    colors = spherical_harmonics(active_sh_degree, rays_d.reshape(B, N, 3),
                                 shs[0, :, :k, :].reshape(B, N, k, 3))
    colors = (colors + 0.5).clip(min=0.0)

    from gauss_splat import projection_three_dims_gaussian_fused
    vi = torch.ones((B, C, N), device=means.device)
    sigma = torch.ones((1,), device=means.device)
    means2d, depths, conics, opacities_proj, radius, covars2d, colors_proj, cnt, _proj_filter, _vi_culling = \
        projection_three_dims_gaussian_fused(
            means.reshape(B, N, 3), colors, None,
            quats.reshape(B, N, 4), scales.reshape(B, N, 3), opacities.reshape(B, N),
            viewmats.reshape(B, C, 4, 4).contiguous(), Ks.reshape(B, C, 3, 3),
            vi, sigma, None, width, height, 0.3, 0.01)

    padded_h = math.ceil(height / tile_size) * tile_size
    padded_w = math.ceil(width / tile_size) * tile_size
    tile_grid = torch.stack(torch.meshgrid(
        torch.arange(0, padded_h, tile_size),
        torch.arange(0, padded_w, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(means.device)

    from gauss_splat import flash_gaussian_build_mask, gaussian_sort, calc_render, get_render_schedule_cpp
    import acl

    ts = tile_size
    pix_coord = torch.stack(torch.meshgrid(
        torch.arange(padded_w), torch.arange(padded_h), indexing='xy'), dim=-1).to(means.device)
    pix_coords = pix_coord.reshape(padded_h // ts, ts, padded_w // ts, ts, 2) \
        .permute(0, 2, 1, 3, 4).reshape(padded_h // ts * padded_w // ts, ts * ts, 2) \
        .permute(0, 2, 1).to(torch.float32).contiguous()

    with torch.no_grad():
        tile_sums_g, tile_offsets_g, tile_depths_g, gauss_index_g = flash_gaussian_build_mask(
            means2d, opacities_proj.unsqueeze(2), conics, covars2d, depths.unsqueeze(2),
            cnt.unsqueeze(2), tile_grid.float(), width, height, tile_size)

        sorted_cnts = tile_offsets_g.squeeze(-1)[:, :, -1]
        sorted_cnts_flatten = sorted_cnts.flatten()
        sorted_offset = torch.cumsum(sorted_cnts_flatten, dim=0)
        tile_sums_3d = tile_sums_g.squeeze(-1)
        tile_sums_cpu = tile_sums_3d.cpu().to(torch.int64)

        vector_num = acl.get_device_capability(0, 1)[0]
        lb_sched_cpu = get_render_schedule_cpp(tile_sums_cpu, vector_num)
        lb_sched_tensor = lb_sched_cpu.npu()
        max_tile_gauss = torch.amax(tile_sums_g).item()

        sorted_gs_ids = gaussian_sort(
            lb_sched_tensor, tile_sums_3d.to(torch.int32), tile_depths_g, gauss_index_g,
            sorted_offset, max_tile_gauss)

    render_colors_list = []
    render_depths_list = []
    for _cam_view in range(C):
        cf_means2 = means2d[0, _cam_view]
        cf_colors3 = colors_proj[0, _cam_view]
        cf_opacity = opacities_proj[0, _cam_view]
        inv_x_0 = conics[0, _cam_view, 0, :]
        inv_x_1 = conics[0, _cam_view, 1, :]
        inv_x_2 = conics[0, _cam_view, 2, :]
        # render metric uses inverse depth (1/z); tile sort below uses direct z
        cf_depth_r = (1.0 / depths[0, _cam_view].clamp(min=1e-6)).unsqueeze(0)
        lb_sched = lb_sched_tensor[0, _cam_view, :]

        view_idx = _cam_view
        start_index = sorted_offset[view_idx - 1] if view_idx > 0 else 0
        end_index = sorted_offset[view_idx]
        sorted_gs_id = sorted_gs_ids[start_index:end_index]

        cf_render_colors, cf_render_depths = calc_render(
            cf_means2, inv_x_0, inv_x_1, inv_x_2, cf_opacity,
            cf_colors3, cf_depth_r, pix_coords, lb_sched, sorted_gs_id)

        nh = padded_h // ts
        nw = padded_w // ts
        rc = cf_render_colors.permute(1, 2, 0).reshape(nh, nw, ts, ts, -1) \
             .transpose(1, 2).reshape(nh * ts, nw * ts, -1).permute(2, 0, 1)[:, :height, :width]
        if cf_render_depths.dim() == 2:
            cf_render_depths = cf_render_depths.unsqueeze(-1)
        rd = cf_render_depths.permute(1, 2, 0).reshape(nh, nw, ts, ts, -1) \
             .transpose(1, 2).reshape(nh * ts, nw * ts, -1).permute(2, 0, 1)[:, :height, :width]

        render_colors_list.append(rc.permute(1, 2, 0))
        render_depths_list.append(rd.permute(1, 2, 0))

    return torch.stack(render_colors_list), torch.stack(render_depths_list)


class TestRasterizerRasterization(TestCase):

    def setUp(self):
        self.test_cases = [
            {"n_gs": 10, "width": 32, "height": 32, "tile_size": 32, "sh_degree": 0},
            {"n_gs": 100, "width": 64, "height": 64, "tile_size": 32, "sh_degree": 1},
            {"n_gs": 500, "width": 128, "height": 128, "tile_size": 32, "sh_degree": 3},
        ]

    def _gen_inputs(self, test_case):
        n_gs = test_case["n_gs"]
        width = test_case["width"]
        height = test_case["height"]
        sh_degree = test_case["sh_degree"]
        tile_size = test_case["tile_size"]

        splats = _gen_splats(n_gs, sh_degree)
        cam = _gen_camera(width, height)
        size = (width, height)

        return splats, cam, size, tile_size, sh_degree

    def _npu_exec(self, splats, cam, size, tile_size, sh_degree):
        rasterizer = Rasterizer()
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam,
            size=size,
            tile_size=tile_size,
            active_sh_degree=sh_degree,
            splats=splats,
            camera_model="pinhole",
        )
        return render_colors.detach().float(), render_depths.detach().float(), meta

    def _golden_exec(self, splats, cam, size, tile_size, sh_degree):
        render_colors, render_depths = _golden_rasterization(
            splats, cam, size, tile_size, sh_degree, "pinhole")
        return render_colors.detach().float(), render_depths.detach().float()

    def test_rasterization_output_shape(self):
        for tc in self.test_cases:
            with self.subTest(**tc):
                splats, cam, size, tile_size, sh_degree = self._gen_inputs(tc)
                npu_colors, npu_depths, meta = self._npu_exec(splats, cam, size, tile_size, sh_degree)
                C = 1
                _assert_shape(self, npu_colors, (C, tc["height"], tc["width"], 3), "colors")
                _assert_shape(self, npu_depths, (C, tc["height"], tc["width"], 1), "depths")
                del splats, cam, npu_colors, npu_depths, meta
                gc.collect()
                torch.npu.empty_cache()

    def test_rasterization_colors_match(self):
        for tc in self.test_cases:
            with self.subTest(**tc):
                splats, cam, size, tile_size, sh_degree = self._gen_inputs(tc)
                golden_colors, golden_depths = self._golden_exec(splats, cam, size, tile_size, sh_degree)
                npu_colors, npu_depths, _ = self._npu_exec(splats, cam, size, tile_size, sh_degree)
                diff = (golden_colors.cpu() - npu_colors.cpu()).abs()
                self.assertLess(diff.max().item(), 0.02,
                    f"max color diff {diff.max().item():.6f}")
                del splats, cam, golden_colors, golden_depths, npu_colors, npu_depths
                gc.collect()
                torch.npu.empty_cache()

    def test_rasterization_depths_match(self):
        for tc in self.test_cases:
            with self.subTest(**tc):
                splats, cam, size, tile_size, sh_degree = self._gen_inputs(tc)
                golden_colors, golden_depths = self._golden_exec(splats, cam, size, tile_size, sh_degree)
                npu_colors, npu_depths, _ = self._npu_exec(splats, cam, size, tile_size, sh_degree)
                self.assertRtolEqual(
                    golden_depths.cpu().numpy(),
                    npu_depths.cpu().numpy(),
                )
                del splats, cam, golden_colors, golden_depths, npu_colors, npu_depths
                gc.collect()
                torch.npu.empty_cache()

    def test_rasterization_meta(self):
        for tc in self.test_cases:
            with self.subTest(**tc):
                splats, cam, size, tile_size, sh_degree = self._gen_inputs(tc)
                _, _, meta = self._npu_exec(splats, cam, size, tile_size, sh_degree)
                self.assertEqual(meta["width"], tc["width"])
                self.assertEqual(meta["height"], tc["height"])
                self.assertEqual(meta["n_cameras"], 1)
                self.assertIn("means2d", meta)
                self.assertIn("radii", meta)
                if getattr(gsplat_c, "use_arch35", False):
                    self.assertIsNotNone(meta["gaussian_ids"])
                    ids = meta["gaussian_ids"]
                    self.assertLessEqual(ids.numel(), tc["n_gs"])
                    if ids.numel():
                        self.assertGreaterEqual(ids.min().item(), 0)
                        self.assertLess(ids.max().item(), tc["n_gs"])
                        self.assertEqual(torch.unique(ids).numel(), ids.numel())
                del splats, cam, meta
                gc.collect()
                torch.npu.empty_cache()


class TestRasterizerStateCaching(TestCase):

    def setUp(self):
        torch.manual_seed(99)

    def test_state_cached_across_calls(self):
        rasterizer = Rasterizer()
        n_gs = 20
        sh_degree = 0
        width, height, tile_size = 32, 32, 32

        splats = _gen_splats(n_gs, sh_degree)
        cam = _gen_camera(width, height)

        rasterizer.rasterization(
            cam=cam, size=(width, height), tile_size=tile_size,
            active_sh_degree=sh_degree, splats=splats, camera_model="pinhole",
        )
        self.assertIsNotNone(rasterizer.tile_grid)
        self.assertIsNotNone(rasterizer.pix_coord)
        self.assertEqual(rasterizer.padded_width, 32)
        self.assertEqual(rasterizer.padded_height, 32)
        self.assertEqual(rasterizer._geometry_key, (width, height, tile_size, splats["means"].device))

        cached_tile_grid = rasterizer.tile_grid
        cached_pix_coord = rasterizer.pix_coord

        splats2 = _gen_splats(n_gs, sh_degree)
        rasterizer.rasterization(
            cam=cam, size=(width, height), tile_size=tile_size,
            active_sh_degree=sh_degree, splats=splats2, camera_model="pinhole",
        )
        self.assertIs(rasterizer.tile_grid, cached_tile_grid)
        self.assertIs(rasterizer.pix_coord, cached_pix_coord)

    def test_fresh_rasterizer_no_cache(self):
        rasterizer = Rasterizer()
        self.assertIsNone(rasterizer.tile_grid)
        self.assertIsNone(rasterizer.pix_coord)
        self.assertIsNone(rasterizer.padded_width)
        self.assertIsNone(rasterizer.padded_height)
        self.assertIsNone(rasterizer._geometry_key)


class TestRasterizerNonAlignedSize(TestCase):

    def test_non_aligned_output_shape(self):
        test_cases = [
            {"n_gs": 30, "width": 48, "height": 48, "tile_size": 32, "sh_degree": 0},
            {"n_gs": 50, "width": 100, "height": 66, "tile_size": 32, "sh_degree": 1},
        ]
        for tc in test_cases:
            with self.subTest(**tc):
                splats = _gen_splats(tc["n_gs"], tc["sh_degree"])
                cam = _gen_camera(tc["width"], tc["height"])
                rasterizer = Rasterizer()
                render_colors, render_depths, _ = rasterizer.rasterization(
                    cam=cam, size=(tc["width"], tc["height"]),
                    tile_size=tc["tile_size"], active_sh_degree=tc["sh_degree"],
                    splats=splats, camera_model="pinhole",
                )
                _assert_shape(self, render_colors, (1, tc["height"], tc["width"], 3), "colors")
                _assert_shape(self, render_depths, (1, tc["height"], tc["width"], 1), "depths")
                del splats, cam, render_colors, render_depths
                gc.collect()
                torch.npu.empty_cache()

    def test_non_aligned_padded_dims(self):
        rasterizer = Rasterizer()
        splats = _gen_splats(10, 0)
        cam = _gen_camera(48, 48)
        rasterizer.rasterization(
            cam=cam, size=(48, 48), tile_size=32,
            active_sh_degree=0, splats=splats, camera_model="pinhole",
        )
        self.assertEqual(rasterizer.padded_width, 64)
        self.assertEqual(rasterizer.padded_height, 64)


class TestRasterizerValidation(TestCase):

    def test_invalid_means_shape(self):
        rasterizer = Rasterizer()
        splats = {
            "means": torch.randn(10, 2, device='npu'),
            "quats": torch.randn(10, 4, device='npu'),
            "scales": torch.zeros(10, 3, device='npu'),
            "opacities": torch.zeros(10, device='npu'),
            "sh0": torch.zeros(10, 1, 3, device='npu'),
            "shN": torch.zeros(10, 0, 3, device='npu'),
        }
        cam = _gen_camera(32, 32)
        with self.assertRaises(ValueError):
            rasterizer.rasterization(
                cam=cam, size=(32, 32), tile_size=32,
                active_sh_degree=0, splats=splats, camera_model="pinhole",
            )

    def test_invalid_quats_shape(self):
        rasterizer = Rasterizer()
        splats = {
            "means": torch.randn(10, 3, device='npu'),
            "quats": torch.randn(10, 3, device='npu'),
            "scales": torch.zeros(10, 3, device='npu'),
            "opacities": torch.zeros(10, device='npu'),
            "sh0": torch.zeros(10, 1, 3, device='npu'),
            "shN": torch.zeros(10, 0, 3, device='npu'),
        }
        cam = _gen_camera(32, 32)
        with self.assertRaises(ValueError):
            rasterizer.rasterization(
                cam=cam, size=(32, 32), tile_size=32,
                active_sh_degree=0, splats=splats, camera_model="pinhole",
            )

    def test_invalid_render_mode(self):
        rasterizer = Rasterizer()
        splats = _gen_splats(10, 0)
        camtoworlds = torch.eye(4, device='npu').unsqueeze(0)
        camtoworlds[0, 2, 3] = 4.0
        Ks = torch.tensor([[[300.0, 0, 16.0], [0, 300.0, 16.0], [0, 0, 1]]], device='npu')
        cam = (camtoworlds, Ks, "INVALID")
        with self.assertRaises(ValueError):
            rasterizer.rasterization(
                cam=cam, size=(32, 32), tile_size=32,
                active_sh_degree=0, splats=splats, camera_model="pinhole",
            )

def _gen_multi_camera(width, height, n_cams=2, device='npu'):
    camtoworlds = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(n_cams, -1, -1).clone()
    for i in range(n_cams):
        camtoworlds[i, 2, 3] = 4.0 + i * 0.5

    fx = fy = 50.0
    cx, cy = width / 2.0, height / 2.0
    Ks = torch.tensor([[[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]]], dtype=torch.float32, device=device).expand(n_cams, -1, -1).clone()

    render_mode = "RGB"
    return (camtoworlds, Ks, render_mode)


class TestRasterizerMultiCamera(TestCase):

    def setUp(self):
        torch.manual_seed(123)

    def test_multi_camera_output_shape(self):
        n_cams = 2
        n_gs = 20
        width, height, tile_size, sh_degree = 32, 32, 32, 0

        splats = _gen_splats(n_gs, sh_degree)
        cam = _gen_multi_camera(width, height, n_cams)
        rasterizer = Rasterizer()
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam, size=(width, height), tile_size=tile_size,
            active_sh_degree=sh_degree, splats=splats, camera_model="pinhole",
        )
        _assert_shape(self, render_colors, (n_cams, height, width, 3), "colors")
        _assert_shape(self, render_depths, (n_cams, height, width, 1), "depths")
        self.assertEqual(meta["n_cameras"], n_cams)
        del splats, cam, render_colors, render_depths, meta
        gc.collect()
        torch.npu.empty_cache()

    def test_multi_camera_different_positions(self):
        n_cams = 2
        n_gs = 50
        width, height, tile_size, sh_degree = 64, 64, 32, 0

        splats = _gen_splats(n_gs, sh_degree)

        camtoworlds = torch.eye(4, dtype=torch.float32, device='npu').unsqueeze(0).expand(n_cams, -1, -1).clone()
        camtoworlds[0, 2, 3] = 4.0
        camtoworlds[1, 0, 3] = 2.0
        camtoworlds[1, 2, 3] = 4.0

        fx = fy = 50.0
        cx, cy = width / 2.0, height / 2.0
        Ks = torch.tensor([[[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]]], dtype=torch.float32, device='npu').expand(n_cams, -1, -1).clone()
        cam = (camtoworlds, Ks, "RGB")

        rasterizer = Rasterizer()
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam, size=(width, height), tile_size=tile_size,
            active_sh_degree=sh_degree, splats=splats, camera_model="pinhole",
        )
        _assert_shape(self, render_colors, (n_cams, height, width, 3), "colors")
        _assert_shape(self, render_depths, (n_cams, height, width, 1), "depths")
        self.assertEqual(meta["n_cameras"], n_cams)
        del splats, cam, render_colors, render_depths, meta
        gc.collect()
        torch.npu.empty_cache()

    def test_multi_camera_gradients_match_individual_views(self):
        means = torch.tensor([[-0.1, 0.0, 2.0], [0.1, 0.0, 2.0]],
                             dtype=torch.float32, device='npu', requires_grad=True)
        shN = torch.tensor([[[0.2, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]],
                           dtype=torch.float32, device='npu').expand(2, -1, -1).clone().requires_grad_()
        splats = {
            "means": means,
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='npu').expand(2, -1).clone().requires_grad_(),
            "scales": torch.tensor([[-2.0, -2.2, -1.9]], device='npu').expand(2, -1).clone().requires_grad_(),
            "opacities": torch.full((2,), 1.0, device='npu', requires_grad=True),
            "sh0": torch.full((2, 1, 3), 0.2, device='npu', requires_grad=True),
            "shN": shN,
        }
        camtoworlds = torch.eye(4, device='npu').unsqueeze(0).repeat(2, 1, 1)
        camtoworlds[1, 0, 3] = 0.3
        Ks = torch.tensor([[[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]]],
                          device='npu').repeat(2, 1, 1)
        grad_inputs = tuple(splats[key] for key in ("means", "quats", "scales", "opacities", "sh0", "shN"))

        def render(camera_matrices, intrinsics):
            return Rasterizer().rasterization(
                cam=(camera_matrices, intrinsics, "RGB"),
                size=(32, 32), tile_size=32, active_sh_degree=1,
                splats=splats, camera_model="pinhole")

        batch_colors, batch_depths, batch_info = render(camtoworlds, Ks)
        batch_loss = batch_colors.sum() + 0.01 * batch_depths.sum()
        batch_loss = batch_loss + 1e-4 * batch_info["means2d"].sum()
        batch_grads = torch.autograd.grad(batch_loss, grad_inputs)

        single_results = [render(camtoworlds[i:i + 1], Ks[i:i + 1]) for i in range(2)]
        single_loss = sum(colors.sum() + 0.01 * depths.sum() + 1e-4 * info["means2d"].sum()
                          for colors, depths, info in single_results)
        single_grads = torch.autograd.grad(single_loss, grad_inputs)

        self.assertGreater(batch_colors.detach().abs().sum().item(), 0)
        self.assertGreater(sum(grad.detach().abs().sum().item() for grad in batch_grads), 0)
        for camera_index, (colors, depths, _) in enumerate(single_results):
            torch.testing.assert_close(batch_colors[camera_index], colors[0], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(batch_depths[camera_index], depths[0], rtol=1e-2, atol=1e-3)
        for batch_grad, single_grad in zip(batch_grads, single_grads):
            torch.testing.assert_close(batch_grad, single_grad, rtol=1e-2, atol=1e-3)

class TestRasterizerMinimalGaussians(TestCase):

    def setUp(self):
        torch.manual_seed(7)

    def test_visible_gaussian_ids(self):
        if not getattr(gsplat_c, "use_arch35", False):
            self.skipTest("A5 dense_indices output is required")
        splats = _gen_splats(3, 0)
        splats["means"] = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, -3.0],
                                         [0.1, 0.0, 3.0]], device='npu')
        camtoworlds = torch.eye(4, device='npu').unsqueeze(0)
        ks = torch.tensor([[[50.0, 0.0, 16.0], [0.0, 50.0, 16.0],
                            [0.0, 0.0, 1.0]]], device='npu')
        _, _, meta = Rasterizer().rasterization(
            cam=(camtoworlds, ks, "RGB"), size=(32, 32), tile_size=32,
            active_sh_degree=0, splats=splats)
        self.assertEqual(meta["gaussian_ids"].tolist(), [0, 2])

    def test_single_gaussian(self):
        rasterizer = Rasterizer()
        splats = _gen_splats(1, 0)
        cam = _gen_camera(32, 32)
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam, size=(32, 32), tile_size=32,
            active_sh_degree=0, splats=splats, camera_model="pinhole",
        )
        _assert_shape(self, render_colors, (1, 32, 32, 3))
        _assert_shape(self, render_depths, (1, 32, 32, 1))
        self.assertEqual(meta["n_cameras"], 1)

    def test_gaussian_behind_camera(self):
        rasterizer = Rasterizer()
        means = torch.tensor([[0.0, 0.0, -100.0]], device='npu')
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='npu')
        scales = torch.log(torch.ones(1, 3, device='npu') * 0.001)
        opacities = torch.logit(torch.tensor([0.01], device='npu'))
        splats = {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": opacities,
            "sh0": torch.zeros(1, 1, 3, device='npu'),
            "shN": torch.zeros(1, 0, 3, device='npu'),
        }
        cam = _gen_camera(32, 32)
        render_colors, render_depths, meta = rasterizer.rasterization(
            cam=cam, size=(32, 32), tile_size=32,
            active_sh_degree=0, splats=splats, camera_model="pinhole",
        )
        self.assertEqual(render_colors.shape[0], 1)
        self.assertEqual(render_depths.shape[0], 1)


if __name__ == "__main__":
    run_tests()
