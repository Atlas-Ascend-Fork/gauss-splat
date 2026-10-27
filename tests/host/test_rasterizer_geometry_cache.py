# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import math
import types
import unittest
from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class Array:
    def __init__(self, values, device="cpu"):
        self.values = np.asarray(values)
        self.device = device
        self.shape = self.values.shape

    def view(self, *shape):
        return Array(self.values.reshape(shape), self.device)

    def reshape(self, *shape):
        return self.view(*shape)

    def transpose(self, dim0, dim1):
        return Array(self.values.swapaxes(dim0, dim1), self.device)

    def permute(self, *dims):
        return Array(self.values.transpose(dims), self.device)

    def __getitem__(self, index):
        return Array(self.values[index], self.device)

    def to(self, device):
        return Array(self.values, device)


def load_rasterizer():
    path = ROOT / "gauss_splat" / "ops" / "rendering.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rasterizer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Rasterizer")
    fake_torch = types.SimpleNamespace(
        arange=np.arange,
        meshgrid=np.meshgrid,
        stack=lambda arrays, dim: Array(np.stack(arrays, axis=dim)),
        exp=np.exp,
        sigmoid=lambda values: 1 / (1 + np.exp(-values)),
        cat=lambda arrays, dim: np.concatenate(arrays, axis=dim),
        linalg=types.SimpleNamespace(inv=np.linalg.inv),
    )
    namespace = {
        "Dict": Dict, "Optional": Optional, "Tuple": Tuple, "Tensor": object,
        "RastCtx": object, "math": math, "torch": fake_torch,
    }
    exec(compile(ast.Module(body=[rasterizer], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Rasterizer"]()


def render(rasterizer, size, tile_size, device="cpu"):
    splats = {
        "means": Array(np.zeros((1, 3)), device),
        "quats": np.zeros((1, 4)),
        "scales": np.zeros((1, 3)),
        "opacities": np.zeros(1),
        "sh0": np.zeros((1, 1, 3)),
        "shN": np.zeros((1, 0, 3)),
    }
    camera = (np.eye(4)[None], np.eye(3)[None], "RGB")

    def mock_pipeline(*args, **kwargs):
        width, height = kwargs["size"]
        tile_pixels = rasterizer.tile_size ** 2
        tiles = Array(np.zeros((rasterizer.tile_grid.shape[0], tile_pixels, 3)))
        image = rasterizer.tile2image(tiles, height, width)
        return image, None, {}

    with mock.patch.object(rasterizer, "_rasterization", side_effect=mock_pipeline):
        image, _, _ = rasterizer.rasterization(camera, size, tile_size, 0, splats)
    return image.shape


class RasterizerGeometryCacheTest(unittest.TestCase):
    def test_reuses_grid_for_same_padded_geometry(self):
        rasterizer = load_rasterizer()
        self.assertEqual(render(rasterizer, (65, 40), 32), (3, 40, 65))
        tile_grid, pix_coord = rasterizer.tile_grid, rasterizer.pix_coord
        self.assertEqual(tile_grid.shape, (6, 2))
        self.assertEqual(pix_coord.shape, (64, 96, 2))

        render(rasterizer, (65, 40), 32)
        self.assertIs(rasterizer.tile_grid, tile_grid)
        self.assertIs(rasterizer.pix_coord, pix_coord)

        self.assertEqual(render(rasterizer, (66, 40), 32), (3, 40, 66))
        self.assertIs(rasterizer.tile_grid, tile_grid)
        self.assertIs(rasterizer.pix_coord, pix_coord)

    def test_rebuilds_grid_for_changed_geometry(self):
        rasterizer = load_rasterizer()
        render(rasterizer, (65, 40), 32)
        previous_grid = rasterizer.tile_grid

        for size, tile_size, device, grid_shape, pixel_shape in (
            ((129, 40), 32, "cpu", (10, 2), (64, 160, 2)),
            ((129, 70), 32, "cpu", (15, 2), (96, 160, 2)),
            ((129, 70), 16, "cpu", (45, 2), (80, 144, 2)),
            ((129, 70), 16, "npu:0", (45, 2), (80, 144, 2)),
        ):
            with self.subTest(size=size, tile_size=tile_size, device=device):
                self.assertEqual(render(rasterizer, size, tile_size, device), (3, size[1], size[0]))
                self.assertIsNot(rasterizer.tile_grid, previous_grid)
                self.assertEqual(rasterizer.tile_grid.shape, grid_shape)
                self.assertEqual(rasterizer.pix_coord.shape, pixel_shape)
                self.assertEqual(rasterizer.tile_grid.device, device)
                previous_grid = rasterizer.tile_grid

        rasterizer.tile_grid = None
        render(rasterizer, (129, 70), 16, "npu:0")
        self.assertIsNotNone(rasterizer.tile_grid)
        self.assertIsNot(rasterizer.tile_grid, previous_grid)


if __name__ == "__main__":
    unittest.main()
