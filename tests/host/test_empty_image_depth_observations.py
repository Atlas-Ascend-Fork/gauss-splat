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


def test_image_without_sparse_observations_has_empty_depths(tmp_path):
    dataset, _, _ = make_dataset(tmp_path, load_depths=True)
    result = dataset[0]
    assert tuple(result.get("points").shape) == (0, 2)
    assert tuple(result.get("depths").shape) == (0,)


def test_observed_point_still_projects_to_image(tmp_path):
    dataset, _, _ = make_dataset(
        tmp_path, load_depths=True, point_indices={"camera.png": np.array([0])}
    )
    result = dataset[0]
    torch.testing.assert_close(result.get("points"), torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(result.get("depths"), torch.tensor([1.0]))
