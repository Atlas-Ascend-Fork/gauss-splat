# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import sys
import tempfile
import types
import unittest
from collections import OrderedDict, namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DATASETS = EXAMPLES / "datasets"


RectifyMapCall = namedtuple(
    "RectifyMapCall",
    "intrinsics distortion rectification new_intrinsics size map_type",
)


def make_rectify_maps(*arguments):
    call = RectifyMapCall(*arguments)
    width, height = call.size
    return (
        np.full((height, width), 101.0, dtype=np.float32),
        np.full((height, width), 102.0, dtype=np.float32),
    )


def make_cv2():
    cv2 = Mock(spec=[
        "CV_32FC1", "INTER_LINEAR", "perspective_params",
        "getOptimalNewCameraMatrix", "initUndistortRectifyMap",
    ])
    cv2.CV_32FC1 = 5
    cv2.INTER_LINEAR = 1
    cv2.perspective_params = []

    def optimal_camera_matrix(intrinsics, params, size, alpha):
        cv2.perspective_params.append(tuple(float(value) for value in params))
        return intrinsics.copy(), (0, 0, size[0], size[1])

    cv2.configure_mock(**{
        "getOptimalNewCameraMatrix.side_effect": optimal_camera_matrix,
        "initUndistortRectifyMap.side_effect": make_rectify_maps,
    })
    return cv2


class FakeCamera:
    def __init__(self, camera_type, focal_length, distortion):
        self.camera_type = camera_type
        self.width = 12
        self.height = 10
        self.fx = focal_length
        self.fy = focal_length
        self.cx = 5.5
        self.cy = 4.5
        self.k1, self.k2, self.p1, self.p2 = distortion
        self.k3 = distortion[2]
        self.k4 = distortion[3]


def make_image(camera_id, name):
    image = Mock(spec=["camera_id", "name", "tvec", "R"])
    image.configure_mock(**{
        "camera_id": camera_id,
        "name": name,
        "tvec": np.zeros(3),
        "R.return_value": np.eye(3),
    })
    return image


def read_test_image(path):
    return np.zeros((10, 12, 3), dtype=np.uint8)


def load_colmap_module():
    """Load the parser with optional example dependencies replaced by small stubs."""
    cv2 = make_cv2()
    imageio = types.ModuleType("imageio")
    imageio.__path__ = []
    imageio_v2 = types.ModuleType("imageio.v2")
    imageio_v2.imread = read_test_image
    imageio.v2 = imageio_v2
    pycolmap = types.ModuleType("pycolmap")
    pycolmap.SceneManager = object
    torch = types.ModuleType("torch")

    package_name = "_gauss_splat_colmap_test"
    examples_package = types.ModuleType(package_name)
    examples_package.__path__ = [str(EXAMPLES)]
    datasets_package = types.ModuleType(package_name + ".datasets")
    datasets_package.__path__ = [str(DATASETS)]
    sys.modules[package_name] = examples_package
    sys.modules[package_name + ".datasets"] = datasets_package

    with patch.dict(
        sys.modules,
        {"cv2": cv2, "imageio": imageio, "imageio.v2": imageio_v2, "pycolmap": pycolmap, "torch": torch},
    ):
        spec = importlib.util.spec_from_file_location(package_name + ".datasets.colmap", DATASETS / "colmap.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    module.cv2 = cv2
    return module, cv2


class TestColmapMixedCameraModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.colmap, cls.cv2 = load_colmap_module()
        cls.temp_root = ROOT / ".tmp"
        cls.created_temp_root = not cls.temp_root.exists()
        cls.temp_root.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.created_temp_root:
            try:
                cls.temp_root.rmdir()
            except OSError:
                pass

    def parse(self, camera_order, camera_types=None):
        if camera_types is None:
            camera_types = {11: "OPENCV", 22: "OPENCV_FISHEYE"}
        cameras = {
            11: FakeCamera(camera_types[11], 7.0, (0.10, -0.02, 0.001, -0.002)),
            22: FakeCamera(camera_types[22], 6.0, (0.05, 0.01, -0.001, 0.0005)),
        }

        def scene_manager(path):
            images = OrderedDict(
                (index, make_image(camera_id, name)) for index, (camera_id, name) in enumerate(camera_order, start=1)
            )
            manager = Mock(spec=[
                "images", "cameras", "points3D", "point3D_errors", "point3D_colors",
                "point3D_id_to_images", "point3D_id_to_point3D_idx", "name_to_image_id",
                "load_cameras", "load_images", "load_points3D",
            ])
            manager.configure_mock(**{
                "images": images,
                "cameras": cameras,
                "points3D": np.empty((0, 3), dtype=np.float32),
                "point3D_errors": np.empty((0,), dtype=np.float32),
                "point3D_colors": np.empty((0, 3), dtype=np.uint8),
                "point3D_id_to_images": {},
                "point3D_id_to_point3D_idx": {},
                "name_to_image_id": {image.name: image_id for image_id, image in images.items()},
                "load_cameras.return_value": None,
                "load_images.return_value": None,
                "load_points3D.return_value": None,
            })
            return manager

        self.colmap.SceneManager = scene_manager
        self.cv2.perspective_params.clear()
        with tempfile.TemporaryDirectory(prefix="colmap-mixed-", dir=self.temp_root) as data_dir:
            data_path = Path(data_dir)
            (data_path / "sparse" / "0").mkdir(parents=True)
            images_path = data_path / "images"
            images_path.mkdir()
            for _, name in camera_order:
                (images_path / name).touch()
            parser = self.colmap.Parser(data_dir)
        return parser, tuple(self.cv2.perspective_params)

    def test_per_camera_undistortion_is_independent_of_image_order(self):
        perspective_first, perspective_first_calls = self.parse(((11, "a_opencv.png"), (22, "b_fisheye.png")))
        fisheye_first, fisheye_first_calls = self.parse(((22, "b_fisheye.png"), (11, "a_opencv.png")))

        expected_opencv_params = (
            0.10000000149011612,
            -0.019999999552965164,
            0.0010000000474974513,
            -0.0020000000949949026,
        )
        for calls in (perspective_first_calls, fisheye_first_calls):
            self.assertEqual(calls, (expected_opencv_params,))

        for parser in (perspective_first, fisheye_first):
            np.testing.assert_array_equal(parser.mapx_dict[11], np.full((10, 12), 101.0, dtype=np.float32))
            np.testing.assert_array_equal(parser.mapy_dict[11], np.full((10, 12), 102.0, dtype=np.float32))
            self.assertGreater(parser.roi_undist_dict[22][2], 0)
            self.assertGreater(parser.roi_undist_dict[22][3], 0)
            self.assertNotEqual(parser.roi_undist_dict[22], (0, 0, 12, 10))

        for camera_id in (11, 22):
            np.testing.assert_array_equal(perspective_first.mapx_dict[camera_id], fisheye_first.mapx_dict[camera_id])
            np.testing.assert_array_equal(perspective_first.mapy_dict[camera_id], fisheye_first.mapy_dict[camera_id])
            np.testing.assert_array_equal(
                perspective_first.intrinsics_dict[camera_id],
                fisheye_first.intrinsics_dict[camera_id],
            )
            self.assertEqual(perspective_first.roi_undist_dict[camera_id], fisheye_first.roi_undist_dict[camera_id])

    def test_same_model_control_is_stable_when_image_order_changes(self):
        camera_types = {11: "OPENCV", 22: "OPENCV"}
        first, first_calls = self.parse(((11, "a_opencv.png"), (22, "b_opencv.png")), camera_types)
        second, second_calls = self.parse(((22, "b_opencv.png"), (11, "a_opencv.png")), camera_types)

        self.assertEqual(len(first_calls), 2)
        self.assertEqual(sorted(first_calls), sorted(second_calls))
        for camera_id in camera_types:
            np.testing.assert_array_equal(first.mapx_dict[camera_id], second.mapx_dict[camera_id])
            np.testing.assert_array_equal(first.mapy_dict[camera_id], second.mapy_dict[camera_id])
            np.testing.assert_array_equal(first.intrinsics_dict[camera_id], second.intrinsics_dict[camera_id])
            self.assertEqual(first.roi_undist_dict[camera_id], second.roi_undist_dict[camera_id])


if __name__ == "__main__":
    unittest.main()
