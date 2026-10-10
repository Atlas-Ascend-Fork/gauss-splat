# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import os
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image
from tqdm import tqdm

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


@pytest.mark.parametrize("folder", ["camera1/nested", ""])
def test_recursive_image_resize_creates_matching_directories(tmp_path, folder):
    namespace = load_objects(
        "examples/datasets/colmap.py",
        ["_get_rel_paths", "_resize_image_folder"],
        os=os,
        List=List,
        imageio=imageio,
        Image=Image,
        tqdm=tqdm,
    )
    source = tmp_path / "images"
    (source / folder).mkdir(parents=True)
    imageio.imwrite(
        source / folder / "frame.jpg", np.full((8, 12, 3), 64, dtype=np.uint8)
    )
    output = tmp_path / "images_2"
    resize = namespace.get("_resize_image_folder")
    assert resize(str(source), str(output), 2) == str(output)
    result = output / folder / "frame.png"
    assert imageio.imread(result).shape == (4, 6, 3)
    original = result.read_bytes()
    resize(str(source), str(output), 2)
    assert result.read_bytes() == original
