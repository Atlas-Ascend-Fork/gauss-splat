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

try:
    import torch
except ImportError:
    torch = None
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]

class TensorStub:
    def __init__(self, shape, requires_grad=False, camera_index=None):
        self.shape = shape
        self.requires_grad = requires_grad
        self.camera_index = camera_index

    def dim(self):
        return len(self.shape)

    def __getitem__(self, index):
        if isinstance(index, int):
            return TensorStub(self.shape[1:], self.requires_grad, index)
        start, stop, step = index.indices(self.shape[0])
        return TensorStub((len(range(start, stop, step)), *self.shape[1:]),
                          self.requires_grad, start)

    def contiguous(self):
        return self


def concatenate(tensors, dim=0):
    shape = list(tensors[0].shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return TensorStub(tuple(shape), any(tensor.requires_grad for tensor in tensors))


def load_rasterization_method():
    path = ROOT / "gauss_splat" / "ops" / "rendering.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rasterizer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Rasterizer")
    method = next(node for node in rasterizer.body if isinstance(node, ast.FunctionDef) and node.name == "_rasterization")
    data_types = [node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name in ("GaussianData", "CameraData", "ConfigData")]
    validators = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name in ("assert_check", "_validate_inputs")]
    namespace = {
        "Dict": Dict, "Optional": Optional, "Tuple": Tuple, "Tensor": object, "RastCtx": object,
        "NamedTuple": NamedTuple,
        "torch": type("TorchStub", (), {"cat": staticmethod(concatenate),
                                         "exp": staticmethod(lambda value: value),
                                         "sigmoid": staticmethod(lambda value: value)}),
    }
    exec(compile(ast.Module(body=data_types + validators + [method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class MultiCameraDispatchTest(unittest.TestCase):
    def test_dispatches_training_views_with_separate_contexts(self):
        self.check_dispatch()

    def test_shared_sh_and_per_view_alpha_are_preserved(self):
        self.check_dispatch(with_alpha=True)

    def test_shared_zero_degree_sh(self):
        self.check_dispatch(sh_degree=0)

    def check_dispatch(self, with_alpha=False, sh_degree=3):
        ns = load_rasterization_method()
        dc_shape = (1, 3, 5)
        sh_shape = ((sh_degree + 1) ** 2 - 1, 3, 5)
        gs = ns["GaussianData"](
            TensorStub((3, 5), True), TensorStub((4, 5)), TensorStub((3, 5)),
            TensorStub((5,)), TensorStub(dc_shape, True), TensorStub(sh_shape, True))
        camera = ns["CameraData"](64, 64, TensorStub((2, 4, 4)), TensorStub((2, 3, 3)))
        config = ns["ConfigData"](sh_degree=sh_degree, channel_first=True)
        calls = []

        class RecordingRasterizer:
            def __init__(self):
                self.calls = calls

            def _rasterization(self, view_gs, view_camera, view_config, ctx):
                self.calls.append((view_camera.w2c.camera_index, ctx, view_gs, view_config))
                info = {
                    "means2d": TensorStub((1, 1, 2, 5), True),
                    "radii": TensorStub((1, 1, 2, 5)),
                    "proj_filter": TensorStub((1, 1, 5)),
                }
                alpha = TensorStub((1, 64, 64, 1), True) if with_alpha else None
                return TensorStub((1, 64, 64, 3), True), TensorStub((1, 64, 64, 1), True), alpha, info

        colors, depths, alpha, info = ns["_rasterization"](
            RecordingRasterizer(), gs, camera, config)
        self.assertEqual([call[0] for call in calls], [0, 1])
        self.assertIsNot(calls[0][1], calls[1][1])
        for _, _, view_gs, view_config in calls:
            self.assertIs(view_config, config)
            self.assertIs(view_gs.scales, gs.scales)
            self.assertIs(view_gs.opacities, gs.opacities)
            self.assertEqual(view_gs.features_dc.shape, dc_shape)
            self.assertEqual(view_gs.features_sh.shape, sh_shape)
            self.assertIs(view_gs, gs)
            self.assertIs(view_gs.features_dc, gs.features_dc)
            self.assertIs(view_gs.features_sh, gs.features_sh)
        self.assertEqual(gs.features_dc.shape, dc_shape)
        self.assertEqual(camera.w2c.shape, (2, 4, 4))
        if with_alpha:
            self.assertEqual(alpha.shape, (2, 64, 64, 1))
        else:
            self.assertIsNone(alpha)
        self.assertEqual(colors.shape, (2, 64, 64, 3))
        self.assertEqual(depths.shape, (2, 64, 64, 1))
        self.assertEqual(info["means2d"].shape, (1, 2, 2, 5))
        self.assertTrue(colors.requires_grad)
        self.assertTrue(info["means2d"].requires_grad)


def load_strategy(name):
    path = ROOT / "examples" / "gsplat" / "strategy" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.decorator_list = []
    cls.bases = []
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in ("step_pre_backward", "_update_state", "_get_n")]
    for node in ast.walk(cls):
        if isinstance(node, ast.arg):
            node.annotation = None
        elif isinstance(node, ast.FunctionDef):
            node.returns = None
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    strategy = namespace[cls.name]()
    strategy.key_for_gradient = "means2d"
    strategy.absgrad = False
    strategy._is_channel_first = False
    return strategy


@unittest.skipIf(torch is None, "CPU PyTorch is required")
class MultiCameraAutogradTest(unittest.TestCase):
    def test_view_gradients_and_gaussian_attribution(self):
        for name in ("adc", "default"):
            for explicit_ids in (True, False):
                with self.subTest(strategy=name, explicit_ids=explicit_ids):
                    self.check_gradients(name, explicit_ids)

    def check_gradients(self, name, explicit_ids):
        ns = load_rasterization_method()
        ns["torch"] = torch
        means = torch.ones(3, 4, requires_grad=True)
        gs = ns["GaussianData"](means, torch.ones(4, 4), torch.ones(3, 4),
                                 torch.ones(4), torch.ones(1, 3, 4), torch.ones(0, 3, 4))
        views = torch.eye(4).repeat(2, 1, 1)
        views[1, 0, 3] = 1
        camera = ns["CameraData"](2, 2, views, torch.eye(3).repeat(2, 1, 1))
        config = ns["ConfigData"](sh_degree=0)
        originals = []

        class Renderer:
            def _rasterization(self, view_gs, view_camera, config, ctx):
                index = int(view_camera.w2c[0, 0, 3].item())
                m = (view_gs.means[:2] * (index + 1))[None, None]
                originals.append(m)
                radii = torch.zeros_like(m)
                radii[..., :2] = 1
                ids = torch.tensor([0, 2] if index == 0 else [1, 2])
                info = dict(means2d=m, radii=radii, width=2, height=2,
                            n_cameras=1, n_gaussians=4,
                            gaussian_ids=ids if explicit_ids else None,
                            proj_filter=torch.tensor([[[5 if index == 0 else 6]]], dtype=torch.uint8))
                image = (m * (index + 1)).sum().reshape(1, 1, 1, 1)
                return image, image, None, info

        images, _, _, info = ns["_rasterization"](Renderer(), gs, camera, config)
        strategy = load_strategy(name)
        params = {"means": torch.zeros(4, 3)}
        state = dict(grad2d=None, count=None, radii=None)
        strategy.step_pre_backward(params, {}, state, 0, info)
        images.sum().backward()
        for index, original in enumerate(originals):
            self.assertIs(info["view_infos"][index]["means2d"], original)
            torch.testing.assert_close(original.grad, torch.full_like(original, index + 1))
        torch.testing.assert_close(means.grad[:2], torch.full_like(means[:2], 5))
        strategy._update_state(params, state, info)
        torch.testing.assert_close(state["count"], torch.tensor([1., 1., 2., 0.]))
        torch.testing.assert_close(state["grad2d"], torch.tensor([2., 4., 6., 0.]) * (2 ** 0.5))
        torch.testing.assert_close(state["radii"], torch.tensor([.5, .5, .5, 0.]))


if __name__ == "__main__":
    unittest.main()
