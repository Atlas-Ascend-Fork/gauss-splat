# Copyright (c) 2026 KaranocaVe. All rights reserved.

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


def load_rasterizer_class():
    source_root = Path(__file__).resolve().parents[1]
    package_name = "_gauss_splat_geometry_cache_test"
    for name, path in (
        (package_name, source_root / "gauss_splat"),
        (f"{package_name}.ops", source_root / "gauss_splat" / "ops"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    functions = {
        "spherical_harmonics": "spherical_harmonics",
        "projection_three_dims_gaussian_fused": "projection_three_dims_gaussian_fused",
        "flash_gaussian_build_mask": "flash_gaussian_build_mask",
        "gaussian_sort": "gaussian_sort",
        "calc_render": "calc_render",
        "get_render_schedule": "get_render_schedule_cpp",
    }
    for module_name, function_name in functions.items():
        module = types.ModuleType(f"{package_name}.ops.{module_name}")
        setattr(module, function_name, lambda *args, **kwargs: None)
        sys.modules[module.__name__] = module

    acl_was_loaded = "acl" in sys.modules
    previous_acl = sys.modules.get("acl")
    sys.modules["acl"] = types.SimpleNamespace(get_device_capability=lambda *_: [1])
    try:
        module_name = f"{package_name}.ops.rendering"
        spec = importlib.util.spec_from_file_location(module_name, source_root / "gauss_splat" / "ops" / "rendering.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if acl_was_loaded:
            sys.modules["acl"] = previous_acl
        else:
            del sys.modules["acl"]

    return module.Rasterizer


class RasterizerGeometryCacheTest(unittest.TestCase):
    def test_geometry_cache_refreshes_for_new_size_and_tile_size(self):
        rasterizer_class = load_rasterizer_class()
        rasterizer = rasterizer_class()
        calls = []

        def mock_pipeline(gs, camera, size, sh_degree=None, tile_size=32):
            tile_render = torch.zeros((rasterizer.tile_grid.shape[0], rasterizer.tile_size * rasterizer.tile_size, 3))
            render_colors = rasterizer.tile2image(tile_render, size[1], size[0])
            render_colors = render_colors.permute(1, 2, 0).unsqueeze(0)
            calls.append(
                {
                    "tile_grid": rasterizer.tile_grid,
                    "pix_coord": rasterizer.pix_coord,
                    "tile_size": rasterizer.tile_size,
                    "padded_height": rasterizer.padded_height,
                    "padded_width": rasterizer.padded_width,
                    "render_shape": tuple(render_colors.shape),
                }
            )
            return render_colors, None, {}

        splats = {
            "means": torch.zeros((1, 3)),
            "quats": torch.zeros((1, 4)),
            "scales": torch.zeros((1, 3)),
            "opacities": torch.zeros((1,)),
            "sh0": torch.zeros((1, 1, 3)),
            "shN": torch.zeros((1, 15, 3)),
        }
        camera = (torch.eye(4).unsqueeze(0), torch.eye(3).unsqueeze(0), "RGB")

        with mock.patch.object(rasterizer, "_rasterization", new=mock_pipeline):
            rasterizer.rasterization(camera, (65, 40), 32, 0, splats)
            rasterizer.rasterization(camera, (65, 40), 32, 0, splats)
            rasterizer.rasterization(camera, (129, 70), 16, 0, splats)

        self.assertEqual(calls[0]["render_shape"], (1, 40, 65, 3))
        self.assertEqual(calls[2]["render_shape"], (1, 70, 129, 3))
        self.assertIs(calls[0]["tile_grid"], calls[1]["tile_grid"])
        self.assertEqual(tuple(calls[0]["tile_grid"].shape), (6, 2))
        self.assertIsNot(calls[1]["tile_grid"], calls[2]["tile_grid"])
        self.assertEqual(tuple(calls[2]["tile_grid"].shape), (45, 2))
        self.assertEqual(tuple(calls[2]["pix_coord"].shape), (80, 144, 2))
        self.assertEqual(calls[2]["tile_size"], 16)
        self.assertEqual(calls[2]["padded_height"], 80)
        self.assertEqual(calls[2]["padded_width"], 144)


if __name__ == "__main__":
    unittest.main()
