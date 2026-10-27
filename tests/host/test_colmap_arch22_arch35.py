# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


class TensorStub:
    def __init__(self, array):
        self.array = array

    def float(self):
        return self

    def bool(self):
        return self


def load_dataset(arch, image):
    path = ROOT / "examples" / arch / "datasets" / "colmap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dataset_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Dataset")
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "Parser": object,
        "np": np,
        "imageio": types.SimpleNamespace(imread=lambda _: image.copy()),
        "torch": types.SimpleNamespace(from_numpy=TensorStub),
    }
    exec(compile(ast.Module(body=[dataset_node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Dataset"]


def load_parser(arch, manager):
    path = ROOT / "examples" / arch / "datasets" / "colmap.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parser_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Parser")
    path_stub = types.SimpleNamespace(join=lambda *parts: "/".join(parts), exists=lambda _: True)
    namespace = {
        "np": np,
        "os": types.SimpleNamespace(path=path_stub),
        "SceneManager": lambda _: manager,
    }
    exec(compile(ast.Module(body=[parser_node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Parser"]


def parser_stub(mask):
    return types.SimpleNamespace(
        image_names=["image.png"], image_paths=["image.png"], camera_ids=[0], test_every=1,
        Ks_dict={0: np.array([[2., 0., 1.], [0., 2., 1.], [0., 0., 1.]])},
        params_dict={0: np.empty(0)}, camtoworlds=np.eye(4)[None], mask_dict={0: mask},
        point_indices={}, points=np.empty((0, 3), dtype=np.float32),
    )


class ColmapDatasetTest(unittest.TestCase):
    def test_unknown_camera_after_supported_camera_raises(self):
        image = lambda camera_id: types.SimpleNamespace(
            R=lambda: np.eye(3), tvec=np.zeros(3), camera_id=camera_id,
        )
        camera = lambda model: types.SimpleNamespace(
            camera_type=model, fx=1., fy=1., cx=1., cy=1., width=3, height=3,
        )
        manager = types.SimpleNamespace(
            load_cameras=lambda: None, load_images=lambda: None, load_points3D=lambda: None,
            images={0: image(0), 1: image(1)}, cameras={0: camera(0), 1: camera(999)},
        )
        for arch in ("arch22", "arch35"):
            parser = load_parser(arch, manager)
            with self.subTest(arch=arch), self.assertRaisesRegex(AssertionError, "Only perspective and fisheye"):
                parser("scene")

    def test_crop_keeps_mask_and_reaches_lower_right(self):
        image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        mask = np.arange(9).reshape(3, 3) % 2 == 0
        for arch in ("arch22", "arch35"):
            dataset = load_dataset(arch, image)(parser_stub(mask), split="test", patch_size=2)
            with self.subTest(arch=arch), patch.object(np.random, "randint", return_value=1) as randint:
                data = dataset[0]
                self.assertEqual(randint.call_count, 2)
                self.assertTrue(all(call.args == (0, 2) for call in randint.call_args_list))
                np.testing.assert_array_equal(data["image"].array, image[1:, 1:])
                np.testing.assert_array_equal(data["mask"].array, mask[1:, 1:])
                np.testing.assert_array_equal(data["K"].array[:2, 2], [0., 0.])

    def test_depths_allow_image_without_sparse_observations(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        for arch in ("arch22", "arch35"):
            dataset = load_dataset(arch, image)(parser_stub(None), split="test", load_depths=True)
            with self.subTest(arch=arch):
                data = dataset[0]
                self.assertEqual(data["points"].array.shape, (0, 2))
                self.assertEqual(data["depths"].array.shape, (0,))


if __name__ == "__main__":
    unittest.main()
