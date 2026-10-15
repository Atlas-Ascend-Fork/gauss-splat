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

import numpy as np
import pytest
import torch
from gsplat.strategy.default import DefaultStrategy

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


class HostStrategy(DefaultStrategy):
    """Expose the inherited update hook for its Host regression."""

    def update_state(self, *args, **kwargs):
        self._update_state(*args, **kwargs)


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_gaussian_ids_take_max_radius_across_views(packed, reverse):
    strategy = HostStrategy()
    state = strategy.initialize_state()
    params = {"means": torch.nn.Parameter(torch.zeros(1, 3))}
    radii = torch.tensor([[10.0, 10.0], [2.0, 2.0]])
    if reverse:
        radii = radii.flip(0)
    gradient = torch.ones((2, 2) if packed else (2, 1, 2))
    info = {
        "width": 100,
        "height": 100,
        "n_cameras": 2,
        "gaussian_ids": torch.zeros(2, dtype=torch.long),
        "radii": radii if packed else radii[:, None],
        strategy.key_for_gradient: SimpleNamespace(grad=gradient),
    }
    strategy.update_state(params, state, info, packed=packed)
    torch.testing.assert_close(state.get("radii"), torch.tensor([0.1]))
    torch.testing.assert_close(state.get("count"), torch.tensor([2.0]))
    info["radii"] = torch.ones_like(info.get("radii"))
    strategy.update_state(params, state, info, packed=packed)
    torch.testing.assert_close(state.get("radii"), torch.tensor([0.1]))
