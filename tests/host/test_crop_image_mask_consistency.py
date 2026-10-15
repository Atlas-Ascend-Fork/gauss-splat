# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_objects(relative, names, **namespace):
    """Execute unmodified production definitions, omitting unrelated imports."""
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in nodes} == set(names)
    namespace.setdefault("np", np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def make_dataset(
    tmp_path, patch_size=None, load_depths=False, with_mask=False, point_indices=None
):
    height, width = 4, 5
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    path = tmp_path / "camera.png"
    imageio.imwrite(path, image)
    mask = (
        np.arange(height * width).reshape(height, width) % 2 == 0 if with_mask else None
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    parser = SimpleNamespace(
        image_names=[path.name],
        image_paths=[str(path)],
        camera_ids=[1],
        test_every=1,
        Ks_dict={1: np.eye(3)},
        camtoworlds=np.eye(4)[None],
        params_dict={1: np.zeros(4) if with_mask else np.empty(0)},
        mask_dict={1: mask},
        mapx_dict={1: grid_x},
        mapy_dict={1: grid_y},
        roi_undist_dict={1: (0, 0, width, height)},
        points=np.array([[1.0, 1.0, 1.0]]),
        point_indices={} if point_indices is None else point_indices,
    )
    namespace = load_objects(
        "examples/datasets/colmap.py",
        ["Dataset"],
        Parser=object,
        Any=Any,
        Dict=Dict,
        Optional=Optional,
        imageio=imageio,
        torch=torch,
        cv2=cv2,
    )
    dataset = namespace.get("Dataset")(
        parser, split="val", patch_size=patch_size, load_depths=load_depths
    )
    return dataset, image, mask


def test_patch_crops_mask_and_includes_last_legal_origin(tmp_path, monkeypatch):
    dataset, image, mask = make_dataset(tmp_path, patch_size=2, with_mask=True)
    bounds = []

    def choose_last(low, high):
        bounds.append((low, high))
        return high - 1

    monkeypatch.setattr(np.random, "randint", choose_last)
    result = dataset[0]
    assert bounds == [(0, 4), (0, 3)]
    np.testing.assert_array_equal(result.get("image").numpy(), image[2:4, 3:5])
    np.testing.assert_array_equal(result.get("mask").numpy(), mask[2:4, 3:5])
    np.testing.assert_allclose(result.get("K").numpy()[:2, 2], [-3, -2])
    np.testing.assert_array_equal(dataset.parser.mask_dict[1], mask)


def test_uncropped_dataset_keeps_full_mask(tmp_path):
    dataset, image, mask = make_dataset(tmp_path, with_mask=True)
    result = dataset[0]
    np.testing.assert_array_equal(result.get("image").numpy(), image)
    np.testing.assert_array_equal(result.get("mask").numpy(), mask)
