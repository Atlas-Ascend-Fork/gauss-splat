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


def save_ply(splats: torch.nn.ParameterDict, dir: str, colors: torch.Tensor = None):
    warnings.warn(
        "save_ply() is deprecated and may be removed in a future release. "
        "Please use the new export_splats() function instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Convert all tensors to numpy arrays in one go
    print(f"Saving ply to {dir}")
    numpy_data = {k: v.detach().cpu().numpy() for k, v in splats.items()}

    means = numpy_data["means"]
    scales = numpy_data["scales"]
    quats = numpy_data["quats"]
    opacities = numpy_data["opacities"]

    sh0 = numpy_data["sh0"].transpose(0, 2, 1).reshape(means.shape[0], -1)
    shN = numpy_data["shN"].transpose(0, 2, 1).reshape(means.shape[0], -1)

    # Create a mask to identify rows with NaN or Inf in any of the numpy_data arrays
    invalid_mask = (
        np.isnan(means).any(axis=1)
        | np.isinf(means).any(axis=1)
        | np.isnan(scales).any(axis=1)
        | np.isinf(scales).any(axis=1)
        | np.isnan(quats).any(axis=1)
        | np.isinf(quats).any(axis=1)
        | np.isnan(opacities)
        | np.isinf(opacities)
        | np.isnan(sh0).any(axis=1)
        | np.isinf(sh0).any(axis=1)
        | np.isnan(shN).any(axis=1)
        | np.isinf(shN).any(axis=1)
    )

    # Filter out rows with NaNs or Infs from all data arrays
    means = means[~invalid_mask]
    scales = scales[~invalid_mask]
    quats = quats[~invalid_mask]
    opacities = opacities[~invalid_mask]
    sh0 = sh0[~invalid_mask]
    shN = shN[~invalid_mask]

    num_points = means.shape[0]

    with open(dir, "wb") as f:
        # Write PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {num_points}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property float nx\n")
        f.write(b"property float ny\n")
        f.write(b"property float nz\n")

        if colors is not None:
            for j in range(colors.shape[1]):
                f.write(f"property float f_dc_{j}\n".encode())
        else:
            for i, data in enumerate([sh0, shN]):
                prefix = "f_dc" if i == 0 else "f_rest"
                for j in range(data.shape[1]):
                    f.write(f"property float {prefix}_{j}\n".encode())

        f.write(b"property float opacity\n")

        for i in range(scales.shape[1]):
            f.write(f"property float scale_{i}\n".encode())
        for i in range(quats.shape[1]):
            f.write(f"property float rot_{i}\n".encode())

        f.write(b"end_header\n")

        # Write vertex data
        for i in range(num_points):
            f.write(struct.pack("<fff", *means[i]))  # x, y, z
            f.write(struct.pack("<fff", 0, 0, 0))  # nx, ny, nz (zeros)

            if colors is not None:
                color = colors.detach().cpu().numpy()
                for j in range(color.shape[1]):
                    f_dc = (color[i, j] - 0.5) / 0.2820947917738781
                    f.write(struct.pack("<f", f_dc))
            else:
                for data in [sh0, shN]:
                    for j in range(data.shape[1]):
                        f.write(struct.pack("<f", data[i, j]))

            f.write(struct.pack("<f", opacities[i]))  # opacity

            for data in [scales, quats]:
                for j in range(data.shape[1]):
                    f.write(struct.pack("<f", data[i, j]))