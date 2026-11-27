# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import struct
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


class TensorStub:
    def __init__(self, array):
        self.array = array

    def float(self):
        return self

    def bool(self):
        return self


def load_dataset(image):
    path = ROOT / "examples" / "datasets" / "colmap.py"
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


def load_parser(manager):
    path = ROOT / "examples" / "datasets" / "colmap.py"
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
        parser = load_parser(manager)
        with self.subTest(), self.assertRaisesRegex(AssertionError, "Only perspective and fisheye"):
            parser("scene")

    def test_crop_keeps_mask_and_reaches_lower_right(self):
        image = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        mask = np.arange(9).reshape(3, 3) % 2 == 0
        dataset = load_dataset(image)(parser_stub(mask), split="test", patch_size=2)
        with self.subTest(), patch.object(np.random, "randint", return_value=1) as randint:
            data = dataset[0]
            self.assertEqual(randint.call_count, 2)
            self.assertTrue(all(call.args == (0, 2) for call in randint.call_args_list))
            np.testing.assert_array_equal(data["image"].array, image[1:, 1:])
            np.testing.assert_array_equal(data["mask"].array, mask[1:, 1:])
            np.testing.assert_array_equal(data["K"].array[:2, 2], [0., 0.])

    def test_depths_allow_image_without_sparse_observations(self):
        image = np.zeros((3, 3, 3), dtype=np.uint8)
        dataset = load_dataset(image)(parser_stub(None), split="test", load_depths=True)
        with self.subTest():
            data = dataset[0]
            self.assertEqual(data["points"].array.shape, (0, 2))
            self.assertEqual(data["depths"].array.shape, (0,))


def _complete_colmap_binary_scene(tmp_path, model, size, intrinsics, distortion):
    from examples.datasets.colmap import Parser

    width, height = size
    fx, fy, cx, cy = intrinsics
    root = tmp_path / "colmap_scene"
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = root / "images"
    images.mkdir()
    if model in (0, 2):
        params = [fx, cx, cy] + list(distortion)
    else:
        params = [fx, fy, cx, cy] + list(distortion)
    camera = struct.pack("<QIiQQ", 1, 1, model, width, height)
    (sparse / "cameras.bin").write_bytes(camera + struct.pack("<" + "d" * len(params), *params))
    record = struct.pack("<QI4d3dI", 1, 1, 1, 0, 0, 0, 0, 0, 0, 1)
    record += b"image.png" + bytes(1) + struct.pack("<QddQ", 1, cx, cy, 1)
    (sparse / "images.bin").write_bytes(record)
    points = struct.pack("<QQ3d3BdQII", 1, 1, 0, 0, 2, 128, 64, 32, 0, 1, 1, 0)
    (sparse / "points3D.bin").write_bytes(points)
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    Image.fromarray(image).save(images / "image.png")
    parser = Parser(str(root), normalize=False, test_every=1)
    return parser, image


def _assert_complete_parser_sample(parser, image, maps, intrinsics, roi, mask):
    from examples.datasets.colmap import Dataset

    x, y, width, height = roi
    expected_image = image
    if maps is not None:
        mapx, mapy = maps
        np.testing.assert_allclose(parser.mapx_dict[1], mapx, rtol=0, atol=1e-6)
        np.testing.assert_allclose(parser.mapy_dict[1], mapy, rtol=0, atol=1e-6)
        assert list(parser.roi_undist_dict[1]) == list(roi)
        expected_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        expected_image = expected_image[y:y + height, x:x + width]
    np.testing.assert_allclose(parser.Ks_dict[1], intrinsics, rtol=0, atol=1e-6)
    assert parser.imsize_dict[1] == (width, height)
    sample = Dataset(parser, split="test", load_depths=True)[0]
    np.testing.assert_array_equal(sample["image"].numpy(), expected_image.astype(np.float32))
    np.testing.assert_allclose(sample["K"].numpy(), intrinsics, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(sample["depths"].numpy(), [2.0])
    if mask is not None:
        np.testing.assert_array_equal(parser.mask_dict[1], mask)
        np.testing.assert_array_equal(sample["mask"].numpy(), mask)
    else:
        assert "mask" not in sample


@pytest.mark.parametrize("size", [(13, 9), (9, 7)])
@pytest.mark.parametrize("off_center", [False, True])
@pytest.mark.parametrize("distortion", [(0.0, 0.0, 0.0, 0.0), (0.08, -0.01, 0.001, -0.0001)])
def test_complete_parser_fisheye_projection(tmp_path, size, off_center, distortion):
    width, height = size
    cx, cy = (width * 0.3, height * 0.35) if off_center else ((width - 1) / 2, (height - 1) / 2)
    fx, fy = 5.0, 6.0
    parser, image = _complete_colmap_binary_scene(
        tmp_path, 5, size, (fx, fy, cx, cy), distortion
    )
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    rays = np.stack(((xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1)
    coefficients = np.array(distortion, dtype=np.float32).astype(np.float64)
    projected, _ = cv2.fisheye.projectPoints(
        rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intrinsics, coefficients
    )
    projected = projected.reshape(height, width, 2).astype(np.float32)
    radius = np.linalg.norm(rays[..., :2], axis=-1)
    theta = np.arctan(radius)
    polynomial = sum(coefficient * theta ** (2 * index + 2) for index, coefficient in enumerate(coefficients))
    scale = np.divide(theta * (1 + polynomial), radius, out=np.ones_like(radius), where=radius > 0)
    analytic = np.stack((fx * rays[..., 0] * scale + cx, fy * rays[..., 1] * scale + cy), axis=-1)
    np.testing.assert_allclose(projected, analytic, rtol=0, atol=1e-6)
    mapx, mapy = projected[..., 0], projected[..., 1]
    valid = (mapx > 0) & (mapy > 0) & (mapx < width - 1) & (mapy < height - 1)
    rows, columns = np.nonzero(valid)
    x, y = columns.min(), rows.min()
    w, h = columns.max() + 1 - x, rows.max() + 1 - y
    cropped_intrinsics = intrinsics.copy()
    cropped_intrinsics[0, 2] -= x
    cropped_intrinsics[1, 2] -= y
    _assert_complete_parser_sample(
        parser, image, (mapx, mapy), cropped_intrinsics, (x, y, w, h), valid[y:y + h, x:x + w]
    )


@pytest.mark.parametrize("model", [0, 1, 2, 4])
def test_complete_parser_perspective_control(tmp_path, model):
    width, height = 13, 9
    fx, fy, cx, cy = 5.0, (5.0 if model in (0, 2) else 6.0), 4.0, 3.0
    distortion = {0: (), 1: (), 2: (0.01,), 4: (0.01, -0.001, 0.0, 0.0)}[model]
    parser, image = _complete_colmap_binary_scene(
        tmp_path, model, (width, height), (fx, fy, cx, cy), distortion
    )
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    maps, roi = None, (0, 0, width, height)
    if distortion:
        params = np.array([distortion[0], 0, 0, 0] if model == 2 else distortion, dtype=np.float32)
        new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(
            intrinsics, params, (width, height), 0
        )
        maps = cv2.initUndistortRectifyMap(
            intrinsics, params, None, new_intrinsics, (width, height), cv2.CV_32FC1
        )
        intrinsics = new_intrinsics
    _assert_complete_parser_sample(parser, image, maps, intrinsics, roi, None)

if __name__ == "__main__":
    unittest.main()
