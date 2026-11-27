# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from math import exp
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.autograd import Variable

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


@pytest.mark.parametrize(
    "first,second", [((1, 7), (3, 7)), ((3, 7), (3, 11)), ((3, 7), (3, 7))]
)
def test_ssim_cached_result_matches_fresh_configuration(first, second):
    ssim = load_objects(
        "examples/rasterization/utils.py",
        ["SSIM"],
        torch=torch,
        F=F,
        Variable=Variable,
        exp=exp,
    ).get("SSIM")
    torch.manual_seed(10)
    first_image = torch.rand(1, first[0], 16, 16)
    ssim.ssim(first_image, first_image, window_size=first[1])
    old_window = ssim.window_cache
    left, right = torch.rand(1, second[0], 16, 16), torch.rand(1, second[0], 16, 16)
    cached = ssim.ssim(left, right, window_size=second[1])
    if first == second:
        assert ssim.window_cache is old_window
    ssim.window_cache = None
    fresh = ssim.ssim(left, right, window_size=second[1])
    torch.testing.assert_close(cached, fresh)
