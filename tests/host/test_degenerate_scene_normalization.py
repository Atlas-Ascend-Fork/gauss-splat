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

import numpy as np
import pytest

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


@pytest.mark.parametrize("kind", ["one", "coincident", "cancelled-up"])
def test_undefined_similarity_reports_input_error(kind):
    function = load_objects("examples/datasets/normalize.py", ["similarity_from_cameras"]).get(
        "similarity_from_cameras"
    )
    cameras = np.repeat(np.eye(4)[None], 1 if kind == "one" else 2, axis=0)
    if kind == "cancelled-up":
        cameras[:, 0, 3] = [-1, 1]
        cameras[1, :3, :3] = np.diag([1.0, -1.0, -1.0])
    with pytest.raises(ValueError, match="Camera"):
        function(cameras, center_method="poses")


def test_regular_camera_layout_remains_finite():
    function = load_objects("examples/datasets/normalize.py", ["similarity_from_cameras"]).get(
        "similarity_from_cameras"
    )
    cameras = np.repeat(np.eye(4)[None], 2, axis=0)
    cameras[:, 0, 3] = [-1, 1]
    np.testing.assert_allclose(function(cameras, center_method="poses"), np.eye(4))
