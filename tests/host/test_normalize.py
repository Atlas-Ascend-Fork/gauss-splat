# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_normalize():
    path = ROOT / "examples" / "datasets" / "normalize.py"
    spec = importlib.util.spec_from_file_location("normalize", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CameraNormalizationTest(unittest.TestCase):
    def test_degenerate_cameras_raise(self):
        normalize = load_normalize()
        cameras = np.repeat(np.eye(4)[None], 2, axis=0)
        with self.subTest(case="coincident"):
            with self.assertRaisesRegex(ValueError, "Camera"):
                normalize.similarity_from_cameras(cameras, center_method="poses")
        cameras[1, :3, :3] = np.diag([1.0, -1.0, -1.0])
        with self.subTest(case="cancelled-up"):
            with self.assertRaisesRegex(ValueError, "Camera"):
                normalize.similarity_from_cameras(cameras, center_method="poses")

    def test_antiparallel_up_is_rotation(self):
        normalize = load_normalize()
        cameras = np.repeat(np.eye(4)[None], 2, axis=0)
        cameras[:, 0, 3] = [-1, 1]
        cameras[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
        transform = normalize.similarity_from_cameras(cameras, center_method="poses")
        result = normalize.transform_cameras(transform, cameras)
        with self.subTest():
            np.testing.assert_allclose(np.linalg.det(result[:, :3, :3]), [1, 1])
            np.testing.assert_allclose(
                result[:, :3, :3] @ np.array([0, -1, 0]),
                [[0, -1, 0], [0, -1, 0]],
            )


if __name__ == "__main__":
    unittest.main()
