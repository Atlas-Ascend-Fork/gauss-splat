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
import torch
from gsplat.strategy.ops import duplicate, split

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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("operation", [duplicate, split])
def test_densification_preserves_parameter_and_adam_state_dtype(dtype, operation):
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype)),
            "scales": torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype)),
            "quats": torch.nn.Parameter(
                torch.tensor([[1.0, 0, 0, 0]] * 2, dtype=dtype)
            ),
            "opacities": torch.nn.Parameter(torch.zeros(2, dtype=dtype)),
        }
    )
    optimizers = {
        name: torch.optim.Adam([parameter], eps=1e-3)
        for name, parameter in params.items()
    }
    for name, parameter in params.items():
        parameter.grad = torch.ones_like(parameter)
        optimizers[name].step()
    operation(params, optimizers, {}, torch.tensor([True, False]))
    for name, parameter in params.items():
        assert parameter.dtype == dtype
        optimizer = optimizers[name]
        assert optimizer.param_groups[0].get("params")[0] is parameter
        for key in ("exp_avg", "exp_avg_sq"):
            value = optimizer.state[parameter].get(key)
            assert value.dtype == dtype
            assert value.shape == parameter.shape
