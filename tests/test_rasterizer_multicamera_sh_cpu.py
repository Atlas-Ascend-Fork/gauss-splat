# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
import types
import unittest
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch


REPO = Path(__file__).resolve().parents[1]


class _ProjectionOutput(NamedTuple):
    means2d: torch.Tensor
    depths: torch.Tensor
    conics: torch.Tensor
    opacities: torch.Tensor
    radius: torch.Tensor
    covars2d: torch.Tensor
    colors: torch.Tensor
    count: torch.Tensor
    proj_filter: torch.Tensor


class _Scene(NamedTuple):
    means: torch.Tensor
    camtoworlds: torch.Tensor
    intrinsics: torch.Tensor
    splats: dict


def _identity(value):
    return value


def _get_device_capability(*_args):
    return (4,)


def _spherical_harmonics_stub(observed_dirs, _degree, dirs, _coeffs):
    observed_dirs.append(dirs.detach().clone())
    return dirs[..., 0].unsqueeze(1).expand(-1, 3, -1).contiguous()


def _projection_stub(*args):
    means = args[0]
    colors = args[1]
    opacities = args[5]
    viewmats = args[6]
    batch, cameras = means.shape[0], viewmats.shape[1]
    count = means.shape[1]
    projected_colors = colors.unsqueeze(1).expand(batch, cameras, 3, count).contiguous()
    return _ProjectionOutput(
        torch.zeros(batch, cameras, 2, count),
        torch.ones(batch, cameras, count),
        torch.zeros(batch, cameras, 3, count),
        opacities.unsqueeze(1).expand(batch, cameras, count).contiguous(),
        torch.ones(batch, cameras, 2, count),
        torch.ones(batch, cameras, 3, count),
        projected_colors,
        torch.full((batch, cameras), count, dtype=torch.int32),
        torch.ones(batch, cameras, count, dtype=torch.uint8),
    )


def _build_mask_stub(*args):
    means2d = args[0]
    tile_grid = args[6]
    batch, cameras = means2d.shape[:2]
    tiles = tile_grid.shape[0]
    return (
        torch.zeros(batch, cameras, tiles, 1),
        torch.zeros(batch, cameras, tiles, 1, dtype=torch.int32),
        torch.zeros(batch, cameras, tiles, 1),
        torch.zeros(batch, cameras, tiles, 1, dtype=torch.int32),
    )


def _render_schedule_stub(tile_sums, _vector_num):
    return torch.zeros(tile_sums.shape[0], tile_sums.shape[1], 1, dtype=torch.int64)


def _sort_gaussians_stub(*_args):
    return torch.empty(0, dtype=torch.int32)


def _calc_render_stub(*args):
    colors = args[5]
    pix_coords = args[7]
    tile_count, pixel_count = pix_coords.shape[0], pix_coords.shape[2]
    color = colors.mean(dim=-1).view(3, 1, 1).expand(3, tile_count, pixel_count).contiguous()
    return color, torch.zeros(1, tile_count, pixel_count)


def _install_operator_stubs(rendering, observed_dirs):
    rendering.spherical_harmonics = partial(_spherical_harmonics_stub, observed_dirs)
    rendering.projection_three_dims_gaussian_fused = _projection_stub
    rendering.flash_gaussian_build_mask = _build_mask_stub
    rendering.get_render_schedule_cpp = _render_schedule_stub
    rendering.gaussian_sort = _sort_gaussians_stub
    rendering.calc_render = _calc_render_stub


def _load_rendering_module():
    names = [
        "acl",
        "gauss_splat",
        "gauss_splat.ops",
        "gauss_splat.ops.spherical_harmonics",
        "gauss_splat.ops.projection_three_dims_gaussian_fused",
        "gauss_splat.ops.flash_gaussian_build_mask",
        "gauss_splat.ops.gaussian_sort",
        "gauss_splat.ops.calc_render",
        "gauss_splat.ops.get_render_schedule",
        "gauss_splat.ops.rendering",
    ]
    previous_modules = {name: sys.modules.get(name) for name in names}
    tensor_has_npu = hasattr(torch.Tensor, "npu")

    if not tensor_has_npu:
        torch.Tensor.npu = _identity

    package = types.ModuleType("gauss_splat")
    package.__path__ = [str(REPO / "gauss_splat")]
    ops_package = types.ModuleType("gauss_splat.ops")
    ops_package.__path__ = [str(REPO / "gauss_splat" / "ops")]
    sys.modules["gauss_splat"] = package
    sys.modules["gauss_splat.ops"] = ops_package

    acl = types.ModuleType("acl")
    acl.get_device_capability = _get_device_capability
    sys.modules["acl"] = acl

    stubs = {
        "spherical_harmonics": {"spherical_harmonics": None},
        "projection_three_dims_gaussian_fused": {"projection_three_dims_gaussian_fused": None},
        "flash_gaussian_build_mask": {"flash_gaussian_build_mask": None},
        "gaussian_sort": {"gaussian_sort": None},
        "calc_render": {"calc_render": None},
        "get_render_schedule": {"get_render_schedule_cpp": None},
    }
    for module_name, attrs in stubs.items():
        module = types.ModuleType(f"gauss_splat.ops.{module_name}")
        for name, value in attrs.items():
            setattr(module, name, value)
        sys.modules[module.__name__] = module

    spec = importlib.util.spec_from_file_location(
        "gauss_splat.ops.rendering", REPO / "gauss_splat" / "ops" / "rendering.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    def restore_modules():
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        if not tensor_has_npu:
            delattr(torch.Tensor, "npu")

    return module, restore_modules


def _make_scene():
    means = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.25, 2.0]])
    camtoworlds = torch.eye(4).repeat(2, 1, 1)
    camtoworlds[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
    camtoworlds[1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    splats = {
        "means": means,
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "scales": torch.zeros(2, 3),
        "opacities": torch.zeros(2),
        "sh0": torch.zeros(2, 1, 3),
        "shN": torch.zeros(2, 3, 3),
    }
    return _Scene(means, camtoworlds, intrinsics, splats)


def _render(rendering, camtoworlds, intrinsics, splats):
    return rendering.Rasterizer().rasterization(
        cam=(camtoworlds, intrinsics, "RGB"),
        size=(4, 4),
        tile_size=4,
        active_sh_degree=1,
        splats=splats,
    )


class TestRasterizerMultiCameraShCPU(unittest.TestCase):
    def test_sh_colors_follow_each_camera_view_direction(self):
        rendering, restore_modules = _load_rendering_module()
        self.addCleanup(restore_modules)
        observed_dirs = []
        _install_operator_stubs(rendering, observed_dirs)
        scene = _make_scene()

        batched_colors, _, batched_meta = _render(
            rendering, scene.camtoworlds, scene.intrinsics, scene.splats
        )
        single_colors = [
            _render(
                rendering,
                scene.camtoworlds.narrow(0, camera_index, 1),
                scene.intrinsics.narrow(0, camera_index, 1),
                scene.splats,
            )[0][0]
            for camera_index in range(2)
        ]
        directions = scene.means.unsqueeze(0) - scene.camtoworlds[:, None, :3, 3]
        directions = torch.nn.functional.normalize(directions, dim=-1)

        self.assertEqual(tuple(batched_colors.shape), (2, 4, 4, 3))
        self.assertEqual(batched_meta["n_cameras"], 2)
        self.assertEqual(tuple(batched_meta["means2d"].shape), (1, 2, 2, 2))
        self.assertEqual(tuple(batched_meta["radii"].shape), (1, 2, 2, 2))
        self.assertEqual(tuple(batched_meta["proj_filter"].shape), (1, 2, 2))
        self.assertFalse(torch.allclose(directions[0], directions[1]))
        torch.testing.assert_close(batched_colors[0], single_colors[0])
        torch.testing.assert_close(batched_colors[1], single_colors[1])
        self.assertFalse(torch.allclose(single_colors[0], single_colors[1]))
        self.assertEqual(tuple(observed_dirs[0].shape), (2, 2, 3))
        torch.testing.assert_close(observed_dirs[0], directions)


if __name__ == "__main__":
    unittest.main()
