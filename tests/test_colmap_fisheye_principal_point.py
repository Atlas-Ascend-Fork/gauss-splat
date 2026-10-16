# Copyright (c) 2026 KaranocaVe. All rights reserved.

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "examples.datasets.colmap"


def _identity_rotation():
    return np.eye(3)


def _no_op(*_args):
    return None


def _passthrough(values):
    return values


def _synthetic_path_exists(path):
    return path.endswith("/sparse") or path.endswith("/images")


class Camera:
    camera_type = 5
    width = 12
    height = 8
    fx = 4.0
    fy = 4.0
    k1 = 0.0
    k2 = 0.0
    k3 = 0.0
    k4 = 0.0

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy


class Image:
    camera_id = 1
    name = "frame.png"
    tvec = np.zeros(3)

    # Preserve the rotation accessor name from the pycolmap API.
    R = staticmethod(_identity_rotation)


class SceneManager:
    fixture = None

    def __init__(self, _path):
        self.__dict__.update(self.fixture)

    def load_cameras(self):
        pass

    def load_images(self):
        pass

    # Preserve the point-loader name from the pycolmap API.
    load_points3D = staticmethod(_no_op)


def _load_parser_module():
    fake_torch = types.ModuleType("torch")
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CV_32FC1 = 5
    fake_imageio = types.ModuleType("imageio")
    fake_imageio.__path__ = []
    fake_imageio_v2 = types.ModuleType("imageio.v2")
    fake_imageio.v2 = fake_imageio_v2
    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = _passthrough
    fake_pycolmap = types.ModuleType("pycolmap")
    fake_pycolmap.SceneManager = SceneManager
    fake_normalize = types.ModuleType("examples.datasets.normalize")
    fake_normalize.align_principal_axes = _no_op
    fake_normalize.similarity_from_cameras = _no_op
    fake_normalize.transform_cameras = _no_op
    fake_normalize.transform_points = _no_op

    with mock.patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "cv2": fake_cv2,
            "imageio": fake_imageio,
            "imageio.v2": fake_imageio_v2,
            "pycolmap": fake_pycolmap,
            "tqdm": fake_tqdm,
            "examples.datasets.normalize": fake_normalize,
        },
    ):
        module_path = REPO_ROOT / "examples" / "datasets" / "colmap.py"
        spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module, fake_imageio_v2


class FisheyePrincipalPointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, cls.imageio = _load_parser_module()

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(MODULE_NAME, None)

    def test_zero_distortion_preserves_pixels_for_off_center_principal_point(self):
        parser = self._parse(cx=3.0, cy=2.0)
        expected_y, expected_x = np.mgrid[0:8, 0:12]

        np.testing.assert_array_equal(parser.mapx_dict[1], expected_x)
        np.testing.assert_array_equal(parser.mapy_dict[1], expected_y)

    def test_centered_principal_point_is_identity_control(self):
        parser = self._parse(cx=6.0, cy=4.0)
        expected_y, expected_x = np.mgrid[0:8, 0:12]

        np.testing.assert_array_equal(parser.mapx_dict[1], expected_x)
        np.testing.assert_array_equal(parser.mapy_dict[1], expected_y)

    def _parse(self, cx, cy):
        SceneManager.fixture = {
            "images": {1: Image()},
            "cameras": {1: Camera(cx, cy)},
            "points3D": np.empty((0, 3)),
            "point3D_errors": np.empty((0,)),
            "point3D_colors": np.empty((0, 3)),
            "name_to_image_id": {"frame.png": 1},
            "point3D_id_to_images": {},
            "point3D_id_to_point3D_idx": {},
        }
        self.imageio.imread = mock.Mock(
            return_value=np.zeros((8, 12, 3), dtype=np.uint8)
        )
        with mock.patch.object(self.module, "_get_rel_paths", return_value=["frame.png"]):
            with mock.patch.object(
                self.module.os.path, "exists", side_effect=_synthetic_path_exists
            ):
                return self.module.Parser(
                    data_dir="/synthetic-colmap", factor=1, normalize=False, test_every=8
                )


if __name__ == "__main__":
    unittest.main()
