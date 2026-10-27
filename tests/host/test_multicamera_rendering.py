# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
import unittest
from pathlib import Path
from typing import Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]


class TensorStub:
    def __init__(self, shape, requires_grad=False, camera_index=None):
        self.shape = shape
        self.requires_grad = requires_grad
        self.camera_index = camera_index

    def dim(self):
        return len(self.shape)

    def __getitem__(self, index):
        return TensorStub((index.stop - index.start, *self.shape[1:]),
                          self.requires_grad, index.start)


def concatenate(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return TensorStub(tuple(shape), any(tensor.requires_grad for tensor in tensors))


def load_rasterization_method():
    path = ROOT / "gauss_splat" / "ops" / "rendering.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rasterizer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Rasterizer")
    method = next(node for node in rasterizer.body if isinstance(node, ast.FunctionDef) and node.name == "_rasterization")
    namespace = {
        "Dict": Dict, "Optional": Optional, "Tuple": Tuple, "Tensor": object, "RastCtx": object,
        "torch": type("TorchStub", (), {"cat": staticmethod(concatenate)}),
        "_validate_inputs": lambda *args: None,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_rasterization"]


class MultiCameraGradientTest(unittest.TestCase):
    def test_dispatches_training_views_with_separate_contexts(self):
        gs = (TensorStub((5, 3), True), TensorStub((5, 4)), TensorStub((5, 3)),
              TensorStub((5,)), TensorStub((5, 16, 3), True))
        camera = (TensorStub((2, 4, 4)), TensorStub((2, 3, 3)), "pinhole", "RGB")
        calls = []

        class RecordingRasterizer:
            def _rasterization(self, _gs, view_camera, *args):
                calls.append((view_camera[0].camera_index, args[-1]))
                info = {
                    "means2d": TensorStub((1, 1, 2, 5), True),
                    "radii": TensorStub((1, 1, 2, 5)),
                    "proj_filter": TensorStub((1, 1, 5)),
                }
                return TensorStub((1, 64, 64, 3), True), TensorStub((1, 64, 64, 1), True), info

        colors, depths, info = load_rasterization_method()(
            RecordingRasterizer(), gs, camera, (64, 64), sh_degree=3,
            camtoworlds=TensorStub((2, 4, 4)))
        self.assertEqual([camera_index for camera_index, _ in calls], [0, 1])
        self.assertIsNot(calls[0][1], calls[1][1])
        self.assertEqual(colors.shape, (2, 64, 64, 3))
        self.assertEqual(depths.shape, (2, 64, 64, 1))
        self.assertEqual(info["means2d"].shape, (1, 2, 2, 5))
        self.assertTrue(colors.requires_grad)
        self.assertTrue(info["means2d"].requires_grad)


if __name__ == "__main__":
    unittest.main()
