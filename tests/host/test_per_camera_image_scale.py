# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import struct
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def colmap():
    import importlib

    module = importlib.import_module("examples.datasets.colmap")
    assert Path(module.__file__).resolve() == ROOT / "examples/datasets/colmap.py"
    return module


def write_scene(root, names, second_size):
    sparse = root / "sparse"
    sparse.mkdir()
    (root / "images").mkdir()
    cameras = struct.pack("<Q", 2)
    images = struct.pack("<Q", 2)
    sizes = [(4, 6), second_size]
    for index, (name, size) in enumerate(zip(names, sizes), start=1):
        imageio.imwrite(root / "images" / name, np.zeros((*size, 3), dtype=np.uint8))
        cameras += struct.pack("<IiQQ4d", index, 1, 12, 8, 8, 8, 6, 4)
        images += struct.pack("<I4d3dI", index, 1, 0, 0, 0, index - 1, 0, 0, index)
        images += name.encode() + b"\0" + struct.pack("<QddQ", 1, 1, 1, 1)
    (sparse / "cameras.bin").write_bytes(cameras)
    (sparse / "images.bin").write_bytes(images)
    point = struct.pack("<QQ3d3BdQ", 1, 1, 0, 0, 4, 255, 255, 255, 0.1, 2)
    (sparse / "points3D.bin").write_bytes(point + struct.pack("<4I", 1, 0, 2, 0))
    return sizes


@pytest.mark.parametrize("names", [("a.png", "b.png"), ("z.png", "a.png")])
@pytest.mark.parametrize("second_size", [(4, 6), (8, 12), (2, 12)])
def test_each_camera_uses_its_own_image_scale(tmp_path, names, second_size, colmap):
    sizes = write_scene(tmp_path, names, second_size)
    parsed = colmap.Parser(str(tmp_path), normalize=False)
    assert parsed.image_names == sorted(names)
    for camera_id, (height, width) in enumerate(sizes, start=1):
        assert parsed.imsize_dict[camera_id] == (width, height)
        np.testing.assert_allclose(
            parsed.Ks_dict[camera_id][0], [8 * width / 12, 0, width / 2]
        )
        np.testing.assert_allclose(
            parsed.Ks_dict[camera_id][1], [0, height, height / 2]
        )
