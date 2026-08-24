# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Adapted from
# https://github.com/nerfstudio-project/gsplat
# Licensed under the Apache License, Version 2.0 (the "License");	 
# you may not use this file except in compliance with the License.	 
# You may obtain a copy of the License at	 
#	 
#     http://www.apache.org/licenses/LICENSE-2.0	 
# 
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License.

import unittest
from collections import namedtuple
import struct
import math
from typing import Optional, Tuple
from typing_extensions import Literal, assert_never

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch import Tensor
from torch_npu.testing.testcase import TestCase, run_tests

from gauss_splat import projection_three_dims_gaussian_fused

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=np.inf)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

ExecResults = namedtuple('ExecResults', ['means2d_culling', 'depths_culling', \
              'conics_culling', 'opacities_culling', 'radius_culling', 'covars2d_culling', 'cnt'])
Inputs = namedtuple('Inputs', ['means', 'colors', 'covars', 'opacities', 'viewmats', 'ks'])
camera_model_dict = {0: 'PINHOLE', 1: 'ORTHO', 2: 'FISHEYE'}


def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats, dim=-1)
    rotmat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return rotmat.reshape(quats.shape[:-1] + (3, 3))


def _quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.quat_scale_to_covar_preci()`."""
    batch_dims = quats.shape[:-1]
    assert quats.shape == batch_dims + (4,), quats.shape
    assert scales.shape == batch_dims + (3,), scales.shape
    rotmat = _quat_to_rotmat(quats)  # [..., 3, 3]

    if compute_covar:
        matrix_r = rotmat * scales[..., None, :]  # [..., 3, 3]
        covars = torch.einsum("...ij,...kj -> ...ik", matrix_r, matrix_r)  # [..., 3, 3]
        if triu:
            covars = covars.reshape(batch_dims + (9,))  # [..., 9]
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]
    if compute_preci:
        preci_matrix = rotmat * (1 / scales[..., None, :])  # [..., 3, 3]
        precis = torch.einsum("...ij,...kj -> ...ik", preci_matrix, preci_matrix)  # [..., 3, 3]
        if triu:
            precis = precis.reshape(batch_dims + (9,))  # [..., 9]
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]

    return covars if compute_covar else None, precis if compute_preci else None


def _persp_proj(
    means: Tensor,  # [..., camera_nums, gaussian_nums, 3]
    covars: Tensor,  # [..., camera_nums, gaussian_nums, 3, 3]
    ks: Tensor,  # [..., camera_nums, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., camera_nums, gaussian_nums, 3].
        covars: Gaussian covariances in camera coordinate system. [..., camera_nums, gaussian_nums, 3, 3].
        ks: Camera intrinsics. [..., camera_nums, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., camera_nums, gaussian_nums, 2].
        - **cov2d**: Projected covariances. [..., camera_nums, gaussian_nums, 2, 2].
    """
    batch_dims = means.shape[:-3]
    camera_nums, gaussian_nums = means.shape[-3:-1]
    assert means.shape == batch_dims + (camera_nums, gaussian_nums, 3), means.shape
    assert covars.shape == batch_dims + (camera_nums, gaussian_nums, 3, 3), covars.shape
    assert ks.shape == batch_dims + (camera_nums, 3, 3), ks.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [..., camera_nums, gaussian_nums]
    tz2 = tz**2  # [..., camera_nums, gaussian_nums]

    fx = ks[..., 0, 0, None]  # [..., camera_nums, 1]
    fy = ks[..., 1, 1, None]  # [..., camera_nums, 1]
    cx = ks[..., 0, 2, None]  # [..., camera_nums, 1]
    cy = ks[..., 1, 2, None]  # [..., camera_nums, 1]
    tan_fovx = 0.5 * width / fx  # [..., camera_nums, 1]
    tan_fovy = 0.5 * height / fy  # [..., camera_nums, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    zero_o = torch.zeros(batch_dims + (camera_nums, gaussian_nums), device=means.device, dtype=means.dtype)
    covars_j = torch.stack(
        [fx / tz, zero_o, -fx * tx / tz2, zero_o, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(batch_dims + (camera_nums, gaussian_nums, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", covars_j, covars, covars_j.transpose(-1, -2))
    means2d = torch.einsum(
        "...ij,...nj->...ni", ks[..., :2, :3], means
    )  # [..., camera_nums, gaussian_nums, 2]
    means2d = means2d / tz[..., None]  # [..., camera_nums, gaussian_nums, 2]
    return means2d, cov2d  # [..., camera_nums, gaussian_nums, 2], [..., camera_nums, gaussian_nums, 2, 2]


def _fisheye_proj(
    means: Tensor,  # [..., camera_nums, gaussian_nums, 3]
    covars: Tensor,  # [..., camera_nums, gaussian_nums, 3, 3]
    ks: Tensor,  # [..., camera_nums, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of fisheye projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., camera_nums, gaussian_nums, 3].
        covars: Gaussian covariances in camera coordinate system. [..., camera_nums, gaussian_nums, 3, 3].
        ks: Camera intrinsics. [..., camera_nums, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., camera_nums, gaussian_nums, 2].
        - **cov2d**: Projected covariances. [..., camera_nums, gaussian_nums, 2, 2].
    """
    batch_dims = means.shape[:-3]
    camera_nums, gaussian_nums = means.shape[-3:-1]
    assert means.shape == batch_dims + (camera_nums, gaussian_nums, 3), means.shape
    assert covars.shape == batch_dims + (camera_nums, gaussian_nums, 3, 3), covars.shape
    assert ks.shape == batch_dims + (camera_nums, 3, 3), ks.shape

    x, y, z = torch.unbind(means, dim=-1)  # [..., camera_nums, gaussian_nums]

    fx = ks[..., 0, 0, None]  # [..., camera_nums, 1]
    fy = ks[..., 1, 1, None]  # [..., camera_nums, 1]
    cx = ks[..., 0, 2, None]  # [..., camera_nums, 1]
    cy = ks[..., 1, 2, None]  # [..., camera_nums, 1]

    eps = 0.0000001
    xy_len = (x**2 + y**2) ** 0.5 + eps
    theta = torch.atan2(xy_len, z + eps)
    means2d = torch.stack(
        [
            x * fx * theta / xy_len + cx,
            y * fy * theta / xy_len + cy,
        ],
        dim=-1,
    )  # [..., camera_nums, gaussian_nums, 2]

    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = torch.atan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / (x2y2)
    covars_j = torch.stack(
        [
            fx * (x2 * a + y2 * b),
            fx * xy * (a - b),
            -fx * x * x2y2z2_inv,
            fy * xy * (a - b),
            fy * (y2 * a + x2 * b),
            -fy * y * x2y2z2_inv,
        ],
        dim=-1,
    ).reshape(batch_dims + (camera_nums, gaussian_nums, 2, 3))

    cov2d = torch.einsum("...ij,...jk,...kl->...il", covars_j, covars, covars_j.transpose(-1, -2))
    return means2d, cov2d  # [..., camera_nums, gaussian_nums, 2], [..., camera_nums, gaussian_nums, 2, 2]


def _ortho_proj(
    means: Tensor,  # [..., camera_nums, gaussian_nums, 3]
    covars: Tensor,  # [..., camera_nums, gaussian_nums, 3, 3]
    ks: Tensor,  # [..., camera_nums, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of orthographic projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [..., camera_nums, gaussian_nums, 3].
        covars: Gaussian covariances in camera coordinate system. [..., camera_nums, gaussian_nums, 3, 3].
        ks: Camera intrinsics. [..., camera_nums, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [..., camera_nums, gaussian_nums, 2].
        - **cov2d**: Projected covariances. [..., camera_nums, gaussian_nums, 2, 2].
    """
    batch_dims = means.shape[:-3]
    camera_nums, gaussian_nums = means.shape[-3:-1]
    assert means.shape == batch_dims + (camera_nums, gaussian_nums, 3), means.shape
    assert covars.shape == batch_dims + (camera_nums, gaussian_nums, 3, 3), covars.shape
    assert ks.shape == batch_dims + (camera_nums, 3, 3), ks.shape

    fx = ks[..., 0, 0, None]  # [..., camera_nums, 1]
    fy = ks[..., 1, 1, None]  # [..., camera_nums, 1]

    zero_o = torch.zeros(batch_dims + (camera_nums, 1), device=means.device, dtype=means.dtype)
    covars_j = (
        torch.stack([fx, zero_o, zero_o, zero_o, fy, zero_o], dim=-1)
        .reshape(batch_dims + (camera_nums, 1, 2, 3))
        .repeat([1] * len(batch_dims) + [1, gaussian_nums, 1, 1])
    )

    cov2d = torch.einsum("...ij,...jk,...kl->...il", covars_j, covars, covars_j.transpose(-1, -2))
    means2d = (
        means[..., :2] * ks[..., None, [0, 1], [0, 1]] + ks[..., None, [0, 1], [2, 2]]
    )  # [..., camera_nums, gaussian_nums, 2]
    return means2d, cov2d  # [..., camera_nums, gaussian_nums, 2], [..., camera_nums, gaussian_nums, 2, 2]


def _world_to_cam(
    means: Tensor,  # [..., gaussian_nums, 3]
    covars: Tensor,  # [..., gaussian_nums, 3, 3]
    viewmats: Tensor,  # [..., camera_nums, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [..., gaussian_nums, 3].
        covars: Gaussian covariances in world coordinate system. [..., gaussian_nums, 3, 3].
        viewmats: world to camera transformation matrices. [..., camera_nums, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [..., camera_nums, gaussian_nums, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [..., camera_nums, gaussian_nums, 3, 3].
    """
    batch_dims = means.shape[:-2]
    gaussian_nums = means.shape[-2]
    camera_nums = viewmats.shape[-3]
    assert means.shape == batch_dims + (gaussian_nums, 3), means.shape
    assert covars.shape == batch_dims + (gaussian_nums, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (camera_nums, 4, 4), viewmats.shape

    rotmat = viewmats[..., :3, :3]  # [..., camera_nums, 3, 3]
    t = viewmats[..., :3, 3]  # [..., camera_nums, 3]
    means_c = (
        torch.einsum("...cij,...nj->...cni", rotmat, means) + t[..., None, :]
    )  # [..., camera_nums, gaussian_nums, 3]
    covars_c = torch.einsum(
        "...cij,...njk,...clk->...cnil", rotmat, covars, rotmat
    )  # [..., camera_nums, gaussian_nums, 3, 3]
    return means_c, covars_c


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def _gaussian_filter(means, colors, det, opacities, means2d, depths, radius_in,
                     conics, covars2d, compensations, width, height, near_plane, far_plane):
    batch_dims, camera_nums, gaussian_nums = det.shape
    if compensations is not None:
        opacities = opacities.unsqueeze(1) * compensations
    else:
        opacities = opacities.unsqueeze(1).repeat(1, camera_nums, 1)
    means = means.float().permute(0, 2, 1).contiguous()
    means2d = means2d.float().permute(0, 1, 3, 2).contiguous()
    radius = radius_in.float().permute(0, 1, 3, 2).contiguous()
    radius_out = radius_in.float().permute(0, 1, 3, 2).contiguous()
    conics = conics.float().permute(0, 1, 3, 2).contiguous()
    colors = colors.float().permute(0, 2, 1).contiguous()
    covars2d = covars2d.float().permute(0, 1, 3, 2).contiguous()
    det = det.float()
    opacities = opacities.float()
    depths = depths.float()
    if compensations is not None:
        compensations = compensations.float()
    
    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0
    inside = (
        (means2d[..., 0] + radius[..., 0] > 0)
        & (means2d[..., 0] - radius[..., 0] < width)
        & (means2d[..., 1] + radius[..., 1] > 0)
        & (means2d[..., 1] - radius[..., 1] < height)
    )
    radius[~inside] = 0.0
    radii = radius.int()
    proj_filter = torch.logical_and(inside, valid)
    means_culling = torch.zeros_like(conics)
    radius_culling = torch.zeros_like(radius)
    means2d_culling = torch.zeros_like(means2d)
    depths_culling = torch.zeros_like(depths)
    opacities_culling = torch.zeros_like(depths)
    conics_culling = torch.zeros_like(conics)
    colors_culling = torch.zeros_like(conics)
    covars2d_culling = torch.zeros_like(covars2d)

    for b in range(batch_dims):
        for c in range(camera_nums):
            radius_culling[b, c, :proj_filter[b, c].sum()] = radius_out[b, c, proj_filter[b, c]]
            means_culling[b, c, :proj_filter[b, c].sum()] = means[b, proj_filter[b, c]]
            means2d_culling[b, c, :proj_filter[b, c].sum()] = means2d[b, c, proj_filter[b, c]]
            depths_culling[b, c, :proj_filter[b, c].sum()] = depths[b, c, proj_filter[b, c]]
            conics_culling[b, c, :proj_filter[b, c].sum()] = conics[b, c, proj_filter[b, c]]
            colors_culling[b, c, :proj_filter[b, c].sum()] = colors[b, proj_filter[b, c]]
            covars2d_culling[b, c, :proj_filter[b, c].sum()] = covars2d[b, c, proj_filter[b, c]]
            opacities_culling[b, c, :proj_filter[b, c].sum()] = opacities[b, c, proj_filter[b, c]]

    cnt = proj_filter.sum(-1)

    filter_bool = proj_filter.bool()
    remainder = gaussian_nums % 8
    if remainder != 0:
        pad_size = 8 - remainder
        filter_bool = F.pad(proj_filter, (0, pad_size), mode='constant', value=False)
    matrix_r = (gaussian_nums + 7) // 8
    filter_reshaped = filter_bool.reshape(batch_dims, camera_nums, matrix_r, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], 
                         dtype=torch.uint8, device=proj_filter.device)
    filter_uint8 = (filter_reshaped.to(torch.uint8) * powers).sum(dim=-1, dtype=torch.uint8)
    
    means_culling = means_culling.permute(0, 1, 3, 2).contiguous()
    radius_culling = radius_culling.permute(0, 1, 3, 2).contiguous()
    means2d_culling = means2d_culling.permute(0, 1, 3, 2).contiguous()
    conics_culling = conics_culling.permute(0, 1, 3, 2).contiguous()
    colors_culling = colors_culling.permute(0, 1, 3, 2).contiguous()
    covars2d_culling = covars2d_culling.permute(0, 1, 3, 2).contiguous()

    return means_culling.float(), colors_culling.float(), means2d_culling.float(), \
           depths_culling.float(), radius_culling.float(), covars2d_culling.float(), \
           conics_culling.float(), opacities_culling.float(), filter_uint8, cnt.to(torch.int32)


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def _fully_fused_projection(
    means: Tensor,  # [..., gaussian_nums, 3]
    colors: Tensor,  # [..., gaussian_nums, 3]
    opacities: Tensor,  # [..., gaussian_nums, 3]
    covars: Tensor,  # [..., gaussian_nums, 3, 3]
    viewmats: Tensor,  # [..., camera_nums, 4, 4]
    ks: Tensor,  # [..., camera_nums, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    batch_dims = means.shape[:-2]
    gaussian_nums = means.shape[-2]
    camera_nums = viewmats.shape[-3]
    assert means.shape == batch_dims + (gaussian_nums, 3), means.shape
    assert covars.shape == batch_dims + (gaussian_nums, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (camera_nums, 4, 4), viewmats.shape
    assert ks.shape == batch_dims + (camera_nums, 3, 3), ks.shape

    assert (
        camera_model != "ftheta"
    ), "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"

    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    if camera_model == "ortho":
        means2d, covars2d = _ortho_proj(means_c, covars_c, ks, width, height)
    elif camera_model == "fisheye":
        means2d, covars2d = _fisheye_proj(means_c, covars_c, ks, width, height)
    elif camera_model == "pinhole":
        means2d, covars2d = _persp_proj(means_c, covars_c, ks, width, height)
    else:
        assert_never(camera_model)

    det_orig = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    if calc_compensations:
        compensations = torch.sqrt(torch.clamp(det_orig / det, min=0.0))
    else:
        compensations = None

    conics = torch.stack(
        [
            covars2d[..., 1, 1] / det,
            -(covars2d[..., 0, 1] + covars2d[..., 1, 0]) / 2.0 / det,
            covars2d[..., 0, 0] / det,
        ],
        dim=-1,
    )  # [..., camera_nums, gaussian_nums, 3]

    depths = means_c[..., 2]  # [..., camera_nums, gaussian_nums]

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2
    tmp = torch.sqrt(torch.clamp(b**2 - det, min=0.01))
    v1 = b + tmp
    r1 = 3.33 * torch.sqrt(v1)
    radius_x = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 0, 0]), r1))
    radius_y = torch.ceil(torch.minimum(3.33 * torch.sqrt(covars2d[..., 1, 1]), r1))

    radius = torch.stack([radius_x, radius_y], dim=-1)  # [..., camera_nums, gaussian_nums, 2]
    covars2d = torch.stack([covars2d[..., 0, 0], covars2d[..., 0, 1], covars2d[..., 1, 1]], dim=-1)
    
    means = means.float().permute(0, 2, 1).contiguous()
    means2d = means2d.float().permute(0, 1, 3, 2).contiguous()
    radius = radius.float().permute(0, 1, 3, 2).contiguous()
    conics = conics.float().permute(0, 1, 3, 2).contiguous()
    colors = colors.float().permute(0, 2, 1).contiguous()
    covars2d = covars2d.float().permute(0, 1, 3, 2).contiguous()
    means_culling, colors_culling, means2d_culling, \
    depths_culling, radius_culling, covars2d_culling, \
    conics_culling, opacities_culling, proj_filter, cnt = _gaussian_filter(means, \
        colors, det, opacities, means2d, depths, radius, conics, covars2d, \
        compensations, width, height, near_plane, far_plane)
    
    return means2d_culling, depths_culling, conics_culling, opacities_culling, radius_culling, covars2d_culling, cnt


class TestProjection3DGSForward(TestCase):
    def setUp(self):
        self.test_cases = [
            [1, 1, 1, 64, 64, 0.3, 0.01, 1e10, True, "pinhole"],
            [1, 8, 8, 64, 64, 0.3, 0.01, 1e10, False, "pinhole"],
            [1, 15, 2048, 64, 64, 0.3, 0.01, 1e10, True, "pinhole"],
        ]
        self.test_results = self.gen_results()

    def gen_inputs(self, shape):
        batch_size, camera_num, gaussian_num = shape

        means = torch.rand(batch_size, gaussian_num, 3).float()
        covars = torch.rand(batch_size, gaussian_num, 3, 3).float()
        colors = torch.rand(batch_size, gaussian_num, 3).float()
        opacities = torch.rand(batch_size, gaussian_num).float()
        viewmats = torch.zeros(batch_size, camera_num, 4, 4).float()
        rotmat = torch.rand(batch_size, camera_num, 3, 3).float()
        rotmat = torch.matmul(rotmat, rotmat.transpose(-2, -1)).float()
        t = torch.rand(batch_size, camera_num, 3).float()
        viewmats[:, :, :3, :3] = rotmat
        viewmats[:, :, :3, 3] = t
        viewmats[:, :, 3, 3] = 1
        fx, fy, s, x0, y0 = 1.611400e3, 1.611400e3, 0, 7.79500e2, 5.19500e2
        k = torch.tensor([[[[fx, s, x0], [0, fy, y0], [0, 0, 1]]]], dtype=torch.float32)
        ks = k.expand(batch_size, camera_num, 3, 3)

        return Inputs(means, colors, covars, opacities, viewmats, ks)

    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            inputs = self.gen_inputs(test_case[:3])
            width, height, eps2d, near_plane, far_plane, calc_compensations, camera_model = test_case[3:]
            cpu_results = self.cpu_to_exec(inputs, width, height, eps2d, \
                          near_plane, far_plane, calc_compensations, camera_model)
            npu_results = self.npu_to_exec(inputs, width, height, eps2d, \
                          near_plane, far_plane, calc_compensations, camera_model)
            test_results.append((cpu_results, npu_results))
        return test_results

    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def cpu_to_exec(self, inputs, width, height, eps2d, near_plane, far_plane, calc_compensations, camera_model):
        means = inputs.means.npu()
        colors = inputs.colors.npu()
        covars = inputs.covars.npu()
        opacities = inputs.opacities.npu()
        viewmats = inputs.viewmats.npu()
        ks = inputs.ks.npu()
        means2d_culling, depths_culling, conics_culling, opacities_culling, \
        radius_culling, covars2d_culling, cnt = _fully_fused_projection(means, colors, \
            opacities, covars, viewmats, ks, width, height, eps2d, near_plane, \
            far_plane, calc_compensations, camera_model)

        return ExecResults(
            means2d_culling=means2d_culling.detach().float(),
            depths_culling=depths_culling.detach().float(),
            conics_culling=conics_culling.detach().float(),
            opacities_culling=opacities_culling.detach().float(),
            radius_culling=radius_culling.detach().float(),
            covars2d_culling=covars2d_culling.detach().float(),
            cnt=cnt.detach().int()
        )

    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def npu_to_exec(self, inputs, width, height, eps2d, near_plane, far_plane, calc_compensations, camera_model):
        means = inputs.means.npu()
        colors = inputs.colors.npu()
        covars = inputs.covars.npu()
        opacities = inputs.opacities.npu()
        viewmats = inputs.viewmats.npu()
        ks = inputs.ks.npu()
        means2d_culling, depths_culling, conics_culling, opacities_culling, \
        radius_culling, covars2d_culling, colors_culling, cnt, _proj_filter = \
            projection_three_dims_gaussian_fused(means, colors, covars, \
            None, None, opacities, viewmats, ks, width, height, eps2d, \
            near_plane, far_plane, calc_compensations, camera_model)
        return ExecResults(
            means2d_culling=means2d_culling.detach().float(),
            depths_culling=depths_culling.detach().float(),
            conics_culling=conics_culling.detach().float(),
            opacities_culling=opacities_culling.detach().float(),
            radius_culling=radius_culling.detach().float(),
            covars2d_culling=covars2d_culling.detach().float(),
            cnt=cnt.detach().int()
        )

    def test_projection_3dgs_forward(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.means2d_culling.cpu().numpy(), npu_results.means2d_culling.cpu().numpy())
            self.assertRtolEqual(cpu_results.depths_culling.cpu().numpy(), npu_results.depths_culling.cpu().numpy())

if __name__ == "__main__":
    run_tests()