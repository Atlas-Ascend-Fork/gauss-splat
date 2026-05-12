"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import math
import unittest
from collections import namedtuple

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import Rasterizer

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
    means = (torch.rand(n_gs, 3, dtype=torch.float32) - 0.5) * 4.0
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

    fx = fy = 300.0
    cx, cy = width / 2.0, height / 2.0
    Ks = torch.tensor([[[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]]], dtype=torch.float32, device=device)

    render_mode = "RGB"
    return (camtoworlds, Ks, render_mode)


def _golden_rasterization(splats, cam, size, tile_size, active_sh_degree, camera_model):
    import torch.nn.functional as F

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
    means2d, depths, conics, opacities_proj, radius, covars2d, colors_proj, cnt = \
        projection_three_dims_gaussian_fused(
            means.reshape(B, N, 3), colors, None,
            quats.reshape(B, N, 4), scales.reshape(B, N, 3), opacities.reshape(B, N),
            viewmats.reshape(B, C, 4, 4).contiguous(), Ks.reshape(B, C, 3, 3),
            width, height, 0.3, 0.2)

    padded_h = math.ceil(height / tile_size) * tile_size
    padded_w = math.ceil(width / tile_size) * tile_size
    tile_grid = torch.stack(torch.meshgrid(
        torch.arange(0, padded_h, tile_size),
        torch.arange(0, padded_w, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(means.device)

    from gauss_splat import flash_gaussian_build_mask, gaussian_sort, calc_render, get_render_schedule_cpp
    import acl

    with torch.no_grad():
        sorted_gs_ids = []
        tile_offsets = []
        for c in range(C):
            mask = flash_gaussian_build_mask(
                means2d[:, c:c+1, :, :],
                opacities_proj[:, c:c+1, :].unsqueeze(0),
                conics[:, c:c+1, :, :],
                covars2d[:, c:c+1, :, :],
                cnt[:, c:c+1].unsqueeze(0),
                tile_grid.float(), width, height, tile_size)
            sid, toff = gaussian_sort(mask[0, 0], depths[0, c])
            sorted_gs_ids.append(sid)
            tile_offsets.append(toff)

    pix_coord = torch.stack(torch.meshgrid(
        torch.arange(padded_w), torch.arange(padded_h), indexing='xy'), dim=-1).to(means.device)
    ts = tile_size
    pix_coords = pix_coord.reshape(padded_h // ts, ts, padded_w // ts, ts, 2) \
        .permute(0, 2, 1, 3, 4).reshape(padded_h // ts * padded_w // ts, ts * ts, 2) \
        .permute(0, 2, 1).to(torch.float32).contiguous()

    render_colors_list = []
    render_depths_list = []
    for c in range(C):
        cf_means2 = means2d[0, c]
        cf_colors3 = colors_proj[0, c]
        cf_opacity = opacities_proj[0, c]
        inv_x_0 = conics[0, c, 0, :]
        inv_x_1 = conics[0, c, 1, :]
        inv_x_2 = conics[0, c, 2, :]
        cf_depths = depths[0, c:c+1]

        nums = torch.cat([tile_offsets[c][:1], tile_offsets[c][1:] - tile_offsets[c][:-1]])
        lb_sched = get_render_schedule_cpp(nums.cpu().to(torch.int64),
            acl.get_device_capability(0, 1)[0]).clone().detach().to(torch.int64).npu()

        cf_render_colors, cf_render_depths = calc_render(
            cf_means2, inv_x_0, inv_x_1, inv_x_2, cf_opacity,
            cf_colors3, cf_depths, pix_coords, lb_sched, sorted_gs_ids[c])

        nh = padded_h // ts
        nw = padded_w // ts
        rc = cf_render_colors.permute(1, 2, 0).reshape(nh, nw, ts, ts, -1) \
             .transpose(1, 2).reshape(nh * ts, nw * ts, -1).permute(2, 0, 1)[:, :height, :width]
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
        self.test_results = self._gen_results()

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

    def _gen_results(self):
        results = []
        for tc in self.test_cases:
            splats, cam, size, tile_size, sh_degree = self._gen_inputs(tc)
            golden_colors, golden_depths = self._golden_exec(splats, cam, size, tile_size, sh_degree)
            npu_colors, npu_depths, meta = self._npu_exec(splats, cam, size, tile_size, sh_degree)
            results.append({
                "golden_colors": golden_colors,
                "golden_depths": golden_depths,
                "npu_colors": npu_colors,
                "npu_depths": npu_depths,
                "meta": meta,
            })
        return results

    def test_rasterization_output_shape(self):
        for i, (tc, res) in enumerate(zip(self.test_cases, self.test_results)):
            width, height = tc["width"], tc["height"]
            C = 1
            npu_c = res["npu_colors"]
            npu_d = res["npu_depths"]
            _assert_shape(self, npu_c, (C, height, width, 3),
                          f"Case {i}: colors")
            _assert_shape(self, npu_d, (C, height, width, 1),
                          f"Case {i}: depths")

    def test_rasterization_colors_match(self):
        for i, res in enumerate(self.test_results):
            diff = (res["golden_colors"].cpu() - res["npu_colors"].cpu()).abs()
            self.assertLess(diff.max().item(), 0.02,
                f"Case {i}: max color diff {diff.max().item():.6f} exceeds tolerance 0.02")

    def test_rasterization_depths_match(self):
        for i, res in enumerate(self.test_results):
            self.assertRtolEqual(
                res["golden_depths"].cpu().numpy(),
                res["npu_depths"].cpu().numpy(),
            )

    def test_rasterization_meta(self):
        for i, (tc, res) in enumerate(zip(self.test_cases, self.test_results)):
            meta = res["meta"]
            self.assertEqual(meta["width"], tc["width"])
            self.assertEqual(meta["height"], tc["height"])
            self.assertEqual(meta["n_cameras"], 1)
            self.assertIn("means2d", meta)
            self.assertIn("radii", meta)


class TestRasterizerMultiCamera(TestCase):

    def setUp(self):
        self.test_cases = [
            {"n_gs": 10, "width": 32, "height": 32, "tile_size": 32, "sh_degree": 0, "n_cams": 2},
            {"n_gs": 50, "width": 64, "height": 64, "tile_size": 32, "sh_degree": 1, "n_cams": 3},
        ]
        self.test_results = self._gen_results()

    def _gen_splats(self, n_gs, sh_degree, device='npu'):
        means = (torch.rand(n_gs, 3, dtype=torch.float32) - 0.5) * 4.0
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

    def _gen_multi_camera(self, width, height, n_cams, device='npu'):
        camtoworlds = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(n_cams, 1, 1)
        for i in range(n_cams):
            camtoworlds[i, 2, 3] = 4.0 + i * 0.5
            camtoworlds[i, 0, 3] = i * 0.1
        fx = fy = 300.0
        cx, cy = width / 2.0, height / 2.0
        Ks = torch.tensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]],
                          dtype=torch.float32, device=device).repeat(n_cams, 1, 1)
        render_mode = "RGB"
        return (camtoworlds, Ks, render_mode)

    def _gen_results(self):
        results = []
        for tc in self.test_cases:
            splats = self._gen_splats(tc["n_gs"], tc["sh_degree"])
            cam = self._gen_multi_camera(tc["width"], tc["height"], tc["n_cams"])
            rasterizer = Rasterizer()
            render_colors, render_depths, meta = rasterizer.rasterization(
                cam=cam,
                size=(tc["width"], tc["height"]),
                tile_size=tc["tile_size"],
                active_sh_degree=tc["sh_degree"],
                splats=splats,
                camera_model="pinhole",
            )
            results.append({
                "npu_colors": render_colors.detach().float(),
                "npu_depths": render_depths.detach().float(),
                "meta": meta,
            })
        return results

    def test_multi_camera_output_shape(self):
        for i, (tc, res) in enumerate(zip(self.test_cases, self.test_results)):
            npu_c = res["npu_colors"]
            npu_d = res["npu_depths"]
            _assert_shape(self, npu_c, (tc["n_cams"], tc["height"], tc["width"], 3),
                          f"Case {i}: colors")
            _assert_shape(self, npu_d, (tc["n_cams"], tc["height"], tc["width"], 1),
                          f"Case {i}: depths")

    def test_multi_camera_meta(self):
        for i, (tc, res) in enumerate(zip(self.test_cases, self.test_results)):
            meta = res["meta"]
            self.assertEqual(meta["width"], tc["width"])
            self.assertEqual(meta["height"], tc["height"])
            self.assertEqual(meta["n_cameras"], tc["n_cams"])


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
        self.assertIsNotNone(rasterizer.tile_grid, "tile_grid should be cached after first call")
        self.assertIsNotNone(rasterizer.pix_coord, "pix_coord should be cached after first call")
        self.assertEqual(rasterizer.padded_width, 32)
        self.assertEqual(rasterizer.padded_height, 32)

        cached_tile_grid = rasterizer.tile_grid
        cached_pix_coord = rasterizer.pix_coord

        splats2 = _gen_splats(n_gs, sh_degree)
        rasterizer.rasterization(
            cam=cam, size=(width, height), tile_size=tile_size,
            active_sh_degree=sh_degree, splats=splats2, camera_model="pinhole",
        )
        self.assertTrue(torch.equal(rasterizer.tile_grid, cached_tile_grid),
                        "tile_grid should be reused on second call")
        self.assertTrue(torch.equal(rasterizer.pix_coord, cached_pix_coord),
                        "pix_coord should be reused on second call")

    def test_fresh_rasterizer_no_cache(self):
        rasterizer = Rasterizer()
        self.assertIsNone(rasterizer.tile_grid)
        self.assertIsNone(rasterizer.pix_coord)
        self.assertIsNone(rasterizer.padded_width)
        self.assertIsNone(rasterizer.padded_height)


class TestRasterizerNonAlignedSize(TestCase):

    def setUp(self):
        self.test_cases = [
            {"n_gs": 30, "width": 48, "height": 48, "tile_size": 32, "sh_degree": 0},
            {"n_gs": 50, "width": 100, "height": 66, "tile_size": 32, "sh_degree": 1},
        ]
        self.test_results = self._gen_results()

    def _gen_results(self):
        results = []
        for tc in self.test_cases:
            splats = _gen_splats(tc["n_gs"], tc["sh_degree"])
            cam = _gen_camera(tc["width"], tc["height"])
            rasterizer = Rasterizer()
            render_colors, render_depths, meta = rasterizer.rasterization(
                cam=cam, size=(tc["width"], tc["height"]),
                tile_size=tc["tile_size"], active_sh_degree=tc["sh_degree"],
                splats=splats, camera_model="pinhole",
            )
            results.append({
                "npu_colors": render_colors.detach().float(),
                "npu_depths": render_depths.detach().float(),
                "meta": meta,
            })
        return results

    def test_non_aligned_output_shape(self):
        for i, (tc, res) in enumerate(zip(self.test_cases, self.test_results)):
            npu_c = res["npu_colors"]
            npu_d = res["npu_depths"]
            _assert_shape(self, npu_c, (1, tc["height"], tc["width"], 3),
                          f"Case {i}: colors")
            _assert_shape(self, npu_d, (1, tc["height"], tc["width"], 1),
                          f"Case {i}: depths")

    def test_non_aligned_padded_dims(self):
        rasterizer = Rasterizer()
        tc = {"n_gs": 10, "width": 48, "height": 48, "tile_size": 32, "sh_degree": 0}
        splats = _gen_splats(tc["n_gs"], tc["sh_degree"])
        cam = _gen_camera(tc["width"], tc["height"])
        rasterizer.rasterization(
            cam=cam, size=(tc["width"], tc["height"]),
            tile_size=tc["tile_size"], active_sh_degree=tc["sh_degree"],
            splats=splats, camera_model="pinhole",
        )
        self.assertEqual(rasterizer.padded_width, 64, "48 should pad to 64 for tile_size=32")
        self.assertEqual(rasterizer.padded_height, 64, "48 should pad to 64 for tile_size=32")


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
        with self.assertRaises(AssertionError):
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
        with self.assertRaises(AssertionError):
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
        with self.assertRaises(AssertionError):
            rasterizer.rasterization(
                cam=cam, size=(32, 32), tile_size=32,
                active_sh_degree=0, splats=splats, camera_model="pinhole",
            )


class TestRasterizerMinimalGaussians(TestCase):

    def setUp(self):
        torch.manual_seed(7)

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
        self.assertEqual(meta["gaussian_ids"], None)

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
        self.assertEqual(render_colors.shape[0], 1, "Expected 1 camera in output")
        self.assertEqual(render_depths.shape[0], 1, "Expected 1 camera in output")


if __name__ == "__main__":
    run_tests()
