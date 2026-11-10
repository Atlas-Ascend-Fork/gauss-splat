# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/utils.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import math
import struct
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_SH0_TO_DC = 0.2820947917738781


def _transpose_if_channel_first(array: np.ndarray, channels: int) -> np.ndarray:
    if array.ndim == 2 and array.shape[0] == channels and array.shape[1] != channels:
        return array.T
    return array


def _splats_to_n_first(numpy_data: dict) -> dict:
    """Accept both N-first [N,D]/[N,K,3] and channel-first [D,N]/[K,3,N]."""
    numpy_data["means"] = _transpose_if_channel_first(numpy_data["means"], 3)
    numpy_data["scales"] = _transpose_if_channel_first(numpy_data["scales"], 3)
    numpy_data["quats"] = _transpose_if_channel_first(numpy_data["quats"], 4)
    sh = numpy_data["sh"]
    if sh.ndim == 3 and sh.shape[-1] != 3:
        numpy_data["sh"] = sh.transpose(2, 0, 1)  # [K,3,N] -> [N,K,3]
    return numpy_data


def _split_sh_coeffs(sh: np.ndarray, num_points: int):
    sh0 = sh[:, :1, :].transpose(0, 2, 1).reshape(num_points, -1)
    sh_n = sh[:, 1:, :].transpose(0, 2, 1).reshape(num_points, -1)
    return sh0, sh_n


def _invalid_mask(numpy_data: dict, sh0: np.ndarray, sh_n: np.ndarray) -> np.ndarray:
    mask = np.zeros(numpy_data["opacities"].shape[0], dtype=bool)
    for array in (numpy_data["means"], numpy_data["scales"], numpy_data["quats"], sh0, sh_n):
        mask |= np.isnan(array).any(axis=1) | np.isinf(array).any(axis=1)
    opacities = numpy_data["opacities"]
    mask |= np.isnan(opacities) | np.isinf(opacities)
    return mask


def _write_ply_header(handle, num_points: int, numpy_data: dict,
                      sh0: np.ndarray, sh_n: np.ndarray, colors: Tensor) -> None:
    handle.write(b"ply\n")
    handle.write(b"format binary_little_endian 1.0\n")
    handle.write(f"element vertex {num_points}\n".encode())
    handle.write(b"property float x\n")
    handle.write(b"property float y\n")
    handle.write(b"property float z\n")
    handle.write(b"property float nx\n")
    handle.write(b"property float ny\n")
    handle.write(b"property float nz\n")

    if colors is not None:
        for j in range(colors.shape[1]):
            handle.write(f"property float f_dc_{j}\n".encode())
    else:
        for i, coeffs in enumerate([sh0, sh_n]):
            prefix = "f_dc" if i == 0 else "f_rest"
            for j in range(coeffs.shape[1]):
                handle.write(f"property float {prefix}_{j}\n".encode())

    handle.write(b"property float opacity\n")

    for i in range(numpy_data["scales"].shape[1]):
        handle.write(f"property float scale_{i}\n".encode())
    for i in range(numpy_data["quats"].shape[1]):
        handle.write(f"property float rot_{i}\n".encode())

    handle.write(b"end_header\n")


def _write_vertex(handle, index: int, numpy_data: dict, sh0: np.ndarray,
                  sh_n: np.ndarray, color_values: np.ndarray) -> None:
    means = numpy_data["means"]
    handle.write(struct.pack("<fff", *means[index]))  # x, y, z
    handle.write(struct.pack("<fff", 0, 0, 0))  # nx, ny, nz (zeros)

    if color_values is not None:
        for j in range(color_values.shape[1]):
            f_dc = (color_values[index, j] - 0.5) / _SH0_TO_DC
            handle.write(struct.pack("<f", f_dc))
    else:
        for coeffs in (sh0, sh_n):
            for j in range(coeffs.shape[1]):
                handle.write(struct.pack("<f", coeffs[index, j]))

    handle.write(struct.pack("<f", numpy_data["opacities"][index]))  # opacity

    for values in (numpy_data["scales"], numpy_data["quats"]):
        for j in range(values.shape[1]):
            handle.write(struct.pack("<f", values[index, j]))


def _write_ply(path: str, numpy_data: dict, sh0: np.ndarray,
               sh_n: np.ndarray, colors: Tensor) -> None:
    num_points = numpy_data["means"].shape[0]
    color_values = None if colors is None else colors.detach().cpu().numpy()
    with open(path, "wb") as handle:
        _write_ply_header(handle, num_points, numpy_data, sh0, sh_n, colors)
        for i in range(num_points):
            _write_vertex(handle, i, numpy_data, sh0, sh_n, color_values)


def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    warnings.warn(
        "save_ply() is deprecated and may be removed in a future release. "
        "Please use the new export_splats() function instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(f"Saving ply to {dir}")
    numpy_data = _splats_to_n_first({k: v.detach().cpu().numpy() for k, v in splats.items()})

    num_points = numpy_data["means"].shape[0]
    sh0, sh_n = _split_sh_coeffs(numpy_data["sh"], num_points)
    invalid_mask = _invalid_mask(numpy_data, sh0, sh_n)

    # Only point-level parameters are indexed by gaussian count and need masking;
    # any global parameters must be kept as-is.
    for key in ("means", "scales", "quats", "opacities", "sh"):
        if key in numpy_data:
            numpy_data[key] = numpy_data[key][~invalid_mask]
    sh0 = sh0[~invalid_mask]
    sh_n = sh_n[~invalid_mask]

    _write_ply(dir, numpy_data, sh0, sh_n, colors)
