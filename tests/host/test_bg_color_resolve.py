# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""bg_color 训练侧隐式传参（splats 字典）解析校验的 CPU 单测。"""

import unittest
from unittest import mock

try:
    import torch
    from gauss_splat.ops.calc_render import _validate_bg_color
    from gauss_splat.ops.rendering import _resolve_bg_color
    IMPORTS_OK = True
except Exception:  # pragma: no cover - 无 CANN/扩展时跳过
    IMPORTS_OK = False


@unittest.skipUnless(IMPORTS_OK, "gauss_splat extension is not importable")
class TestResolveBgColor(unittest.TestCase):
    def test_disabled(self):
        self.assertIsNone(_resolve_bg_color({}, "cpu"))

    def test_tuple_and_list(self):
        bg = _resolve_bg_color({"bg_color": (0.1, 0.2, 0.3)}, "cpu")
        self.assertEqual(tuple(bg.shape), (3,))
        self.assertEqual(bg.dtype, torch.float32)
        self.assertAlmostEqual(float(bg[0]), 0.1, places=6)

        bg = _resolve_bg_color({"bg_color": [1.0, 0.0, 0.5]}, "cpu")
        self.assertEqual(tuple(bg.shape), (3,))

    def test_tensor_and_parameter(self):
        src = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        bg = _resolve_bg_color({"bg_color": src}, "cpu")
        self.assertEqual(bg.dtype, torch.float32)
        self.assertFalse(bg.requires_grad)

        param = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]))
        bg = _resolve_bg_color({"bg_color": param}, param.device)
        self.assertTrue(bg.requires_grad)
        bg.sum().backward()
        self.assertIsNotNone(param.grad)

    def test_invalid_numel(self):
        with self.assertRaises(ValueError):
            _resolve_bg_color({"bg_color": (0.1, 0.2)}, "cpu")
        with self.assertRaises(ValueError):
            _resolve_bg_color({"bg_color": (0.1, 0.2, 0.3, 0.4)}, "cpu")


@unittest.skipUnless(IMPORTS_OK, "gauss_splat extension is not importable")
class TestValidateBgColor(unittest.TestCase):
    def test_none(self):
        _validate_bg_color(None)

    def test_valid_tensor(self):
        _validate_bg_color(torch.zeros(3))

    def test_non_tensor(self):
        with self.assertRaises(ValueError):
            _validate_bg_color((0.0, 0.0, 0.0))

    def test_invalid_dtype(self):
        with self.assertRaises(ValueError):
            _validate_bg_color(torch.zeros(3, dtype=torch.float64))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            _validate_bg_color(torch.zeros(4))


@unittest.skipUnless(IMPORTS_OK, "gauss_splat extension is not importable")
class TestRasterizationBgWiring(unittest.TestCase):
    """验证中间层 Rasterizer.rasterization 从 splats 取值并透传给渲染实现。"""

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)

    @staticmethod
    def _splats(n=4, **extra):
        data = {
            "means": torch.zeros(3, n, dtype=torch.float32),
            "quats": torch.zeros(4, n, dtype=torch.float32),
            "scales": torch.zeros(3, n, dtype=torch.float32),
            "opacities": torch.zeros(n, dtype=torch.float32),
            "sh": torch.zeros(1, 3, n, dtype=torch.float32),
        }
        data.update(extra)
        return data

    def test_forwards_bg_from_splats(self):
        captured = self._run(self._splats(bg_color=(0.2, 0.4, 0.6)))
        self.assertIsNotNone(captured["bg"])
        self.assertEqual(tuple(captured["bg"].shape), (3,))

    def test_disabled_when_absent(self):
        captured = self._run(self._splats())
        self.assertIsNone(captured["bg"])

    def test_splats_not_mutated(self):
        splats = self._splats(bg_color=(1.0, 1.0, 1.0))
        self._run(splats)
        self.assertIn("bg_color", splats)
        self.assertEqual(tuple(splats["bg_color"]), (1.0, 1.0, 1.0))

    def _run(self, splats):
        from gauss_splat.ops.rendering import Rasterizer

        captured = {}

        def fake(self, gs, camera, size, sh_degree=None, tile_size=32,
                 camtoworlds=None, rast_ctx=None, bg_color=None):
            captured["bg"] = bg_color
            return (torch.zeros(1, 3, 8, 8), torch.zeros(1, 8, 8), {})

        with mock.patch.object(Rasterizer, "_rasterization", fake):
            rasterizer = Rasterizer()
            cam = (torch.eye(4, dtype=torch.float32).unsqueeze(0),
                   torch.eye(3, dtype=torch.float32).unsqueeze(0), "RGB")
            rasterizer.rasterization(cam=cam, size=(8, 8), tile_size=32,
                                     active_sh_degree=0, splats=splats)
        return captured


if __name__ == "__main__":
    unittest.main()
