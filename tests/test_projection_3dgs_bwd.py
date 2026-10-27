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
import struct
import math
from typing import Optional, Tuple
from collections import namedtuple
from typing_extensions import Literal, assert_never

import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch_npu.testing.testcase import TestCase, run_tests
from torch import Tensor

from gauss_splat.gsplat_c import fully_fused_projection_bwd
from gauss_splat.ops.rast_ctx import RastCtx

torch.npu.set_device('npu:0')
torch.set_printoptions(sci_mode=False)
np.set_printoptions(threshold=np.inf)

option = {}
option['ACL_OP_DEBUG_LEVEL'] = 1
torch.npu.set_option(option)

ExecResults = namedtuple('ExecResults', ['v_p_w', 'v_quats', 'v_scales', 'v_r', 'v_colors', 'v_opacities'])
Inputs = namedtuple('Inputs', ['means', 'quats', 'scales', 'conics', 'viewmats', 'ks', 'v_means2d', \
                    'v_depths', 'v_conics', 'v_colors_culling', 'v_opacities_culling', 'proj_filter', 'compensations'])


def inverse_vjp(minv, v_minv):
    return -minv @ v_minv @ minv


def covar_w2c(r, covar_w):
    covars_c = torch.einsum(
        "...cij,...njk,...clk->...cnil", r, covar_w, r
    )  # [..., c, n, 3, 3]
    return covars_c


def covar_w2c_vjp(r, covar_w, v_covar_c, v_r):
    v_r = v_r + torch.einsum("...cnij,...cjk,...nlk->...cil", v_covar_c, r, covar_w) +\
          torch.einsum("...cnji,...cjk,...nkl->...cil", v_covar_c, r, covar_w)
    v_covar_w = torch.einsum("...cij,...cnik,...ckl->...njl", r, v_covar_c, r)

    return v_r, v_covar_w


def pos_w2c(r, t, p_w):
    p_c = (
        torch.einsum("...cij,...nj->...cni", r, p_w) + t[..., None, :]
    )  # [..., c, n, 3]
    return p_c


def pos_w2c_vjp(r, t, p_w, v_p_c):
    v_r = torch.einsum('...cni,...nj->...cij', v_p_c, p_w)
    v_t = torch.einsum('...cni->...ci', v_p_c)
    v_p_w = torch.einsum("...cji,...cnj->...ni", r.transpose(-2, -1), v_p_c)  # (n, 3)
    
    return v_r, v_t, v_p_w


def _quat_to_rotmat(quats: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats = F.normalize(quats, p=2, dim=-1)    
    w, x, y, z = torch.unbind(quats, dim=-1)
    r = torch.stack(
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
    return r.reshape(quats.shape[:-1] + (3, 3))


def _quat_to_rotmat_vjp(quats: Tensor, v_r: Tensor) -> Tensor:
    """Convert quaternion to rotation matrix."""
    quats_n = F.normalize(quats, p=2, dim=-1)
    w, x, y, z = torch.unbind(quats_n, dim=-1)
    vr12_vr21 = v_r[..., 2, 1] - v_r[..., 1, 2]
    vr20_vr02 = v_r[..., 0, 2] - v_r[..., 2, 0]
    vr01_vr10 = v_r[..., 1, 0] - v_r[..., 0, 1]
    vr11_add_vr22 = v_r[..., 1, 1] + v_r[..., 2, 2]
    vr00_add_vr22 = v_r[..., 0, 0] + v_r[..., 2, 2]
    vr00_add_vr11 = v_r[..., 0, 0] + v_r[..., 1, 1]
    vr01_add_vr10 = v_r[..., 1, 0] + v_r[..., 0, 1]
    vr02_add_vr20 = v_r[..., 2, 0] + v_r[..., 0, 2]
    vr12_add_vr21 = v_r[..., 2, 1] + v_r[..., 1, 2]
    v_quat_n = torch.stack(
        [
            2.0 * (x * vr12_vr21 + y * vr20_vr02 + z * vr01_vr10),
            2.0 * (-2 * x * vr11_add_vr22 + y * vr01_add_vr10 + z * vr02_add_vr20 + w * vr12_vr21),
            2.0 * (x * vr01_add_vr10 - 2.0 * y * vr00_add_vr22 + z * vr12_add_vr21 + w * vr20_vr02),
            2.0 * (x * vr02_add_vr20 + y * vr12_add_vr21 - 2.0 * z * vr00_add_vr11 + w * vr01_vr10)
        ],
        dim=-1,
    )

    v_quat = (v_quat_n - torch.einsum('...ni,...ni->...n', v_quat_n, quats_n)[..., None] * quats_n) *\
             torch.rsqrt((quats * quats).sum(-1, keepdim=True))
    return v_quat


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
    r = _quat_to_rotmat(quats)  # [..., 3, 3]
    if compute_covar:
        m = r * scales[..., None, :]  # [..., 3, 3]
        covars = torch.einsum("...ij,...kj -> ...ik", m, m)  # [..., 3, 3]
        if triu:
            covars = covars.reshape(batch_dims + (9,))  # [..., 9]
            covars = (
                covars[..., [0, 1, 2, 4, 5, 8]] + covars[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]
    if compute_preci:
        p = r * (1 / scales[..., None, :])  # [..., 3, 3]
        precis = torch.einsum("...ij,...kj -> ...ik", p, p)  # [..., 3, 3]
        if triu:
            precis = precis.reshape(batch_dims + (9,))  # [..., 9]
            precis = (
                precis[..., [0, 1, 2, 4, 5, 8]] + precis[..., [0, 3, 6, 4, 7, 8]]
            ) / 2.0  # [..., 6]

    return covars if compute_covar else None, precis if compute_preci else None


def _quat_scale_to_covar_vjp(quats, scales, r, v_covar):
    m = r * scales[..., None, :]  # [..., 3, 3]
    v_m = (v_covar + v_covar.transpose(-2, -1)) @ m
    v_r = v_m * scales[..., None, :]
    v_quats = _quat_to_rotmat_vjp(quats, v_r)
    v_scales = torch.einsum("...ij,...ij->...j", r, v_m)
    return v_quats, v_scales


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def _persp_proj_vjp(
        means,
        cov3d,
        ks,
        width,
        height,
        v_cov2d,
        v_mean2d
):
    batch_dims = means.shape[:-3]
    c, n = means.shape[-3:-1]
    assert means.shape == batch_dims + (c, n, 3), means.shape
    assert cov3d.shape == batch_dims + (c, n, 3, 3), cov3d.shape
    assert ks.shape == batch_dims + (c, 3, 3), ks.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [..., c, n]
    tz2 = tz**2  # [..., c, n]

    fx = ks[..., 0, 0, None]  # [..., c, 1]
    fy = ks[..., 1, 1, None]  # [..., c, 1]
    cx = ks[..., 0, 2, None]  # [..., c, 1]
    cy = ks[..., 1, 2, None]  # [..., c, 1]
    tan_fovx = 0.5 * width / fx  # [..., c, 1]
    tan_fovy = 0.5 * height / fy  # [..., c, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy

    # for clipping mask
    x_clipping_mask = (means[..., 0] / tz <= lim_x_pos) & (means[..., 0] / tz >= -lim_x_neg)
    y_clipping_mask = (means[..., 1] / tz <= lim_y_pos) & (means[..., 1] / tz >= -lim_y_neg)
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)
    zero_o = torch.zeros(batch_dims + (c, n), device=means.device, dtype=means.dtype)

    j = torch.stack(
        [fx / tz, zero_o, -fx * tx / tz2, zero_o, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(batch_dims + (c, n, 2, 3))

    v_cov3d = j.transpose(-2, -1) @ v_cov2d @ j

    v_mean3d = torch.stack(
        [fx / tz * v_mean2d[..., 0], 
        fy / tz * v_mean2d[..., 1], 
        -(fx * means[..., 0] * v_mean2d[..., 0] + fy * means[..., 1] * v_mean2d[..., 1]) / tz2], dim=-1)

    tz3 = tz2 * tz
    v_j = v_cov2d @ j @ cov3d.transpose(-2, -1) + v_cov2d.transpose(-2, -1) @ j @ cov3d

    v_mean3d[..., 0] = v_mean3d[..., 0] - (fx / tz2 * v_j[..., 0, 2]) * x_clipping_mask
    v_mean3d[..., 2] = v_mean3d[..., 2] - (fx / tz3 * v_j[..., 0, 2] * tx) * (~x_clipping_mask)

    v_mean3d[..., 1] = v_mean3d[..., 1] - (fy / tz2 * v_j[..., 1, 2]) * y_clipping_mask
    v_mean3d[..., 2] = v_mean3d[..., 2] - (fy / tz3 * v_j[..., 1, 2] * ty) * (~y_clipping_mask)

    v_mean3d[..., 2] = v_mean3d[..., 2] - fx / tz2 * v_j[..., 0, 0] - fy / tz2 * v_j[..., 1, 1] + \
            2.0 * fx * tx / tz3 * v_j[..., 0, 2] + \
            2.0 * fy * ty / tz3 * v_j[..., 1, 2]
    
    return v_mean3d, v_cov3d


# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def _fully_fused_projection_bwd(means, quats, scales, conics, viewmats, ks, v_means2d,\
        v_depths, v_conics, v_colors_culling, v_opacities_culling, proj_filter, compensations, width, height):
    
    means = means.permute(0, 2, 1).contiguous().float()
    quats = quats.permute(0, 2, 1).contiguous().float()
    scales = scales.permute(0, 2, 1).contiguous().float()
    conics = conics.permute(0, 1, 3, 2).contiguous().float()
    v_means2d_culling = v_means2d.permute(0, 1, 3, 2).contiguous().float()
    v_depths_culling = v_depths.float()
    v_conics_culling = v_conics.permute(0, 1, 3, 2).contiguous().float()
    v_colors_culling = v_colors_culling.permute(0, 1, 3, 2).contiguous().float()
    b, c, n = v_opacities_culling.shape

    v_conics = torch.zeros_like(v_conics_culling)
    v_means2d = torch.zeros_like(v_means2d_culling)
    v_depths = torch.zeros_like(v_depths_culling)
    v_colors = torch.zeros_like(v_colors_culling)
    v_opacities = torch.zeros_like(v_opacities_culling)
    
    bit_mask = (1 << torch.arange(8, dtype=torch.uint8))
    proj_filter = (proj_filter.unsqueeze(-1).bitwise_and(bit_mask) != 0).reshape(b, c, -1)
    proj_filter = proj_filter[:, :, :n]
    
    for batch_idx in range(b):
        for camera_idx in range(c):
            cnt = proj_filter[batch_idx, camera_idx].sum()
            v_conics[batch_idx, camera_idx, proj_filter[batch_idx, camera_idx]] = v_conics_culling[batch_idx, camera_idx, :cnt]
            v_means2d[batch_idx, camera_idx, proj_filter[batch_idx, camera_idx]] = v_means2d_culling[batch_idx, camera_idx, :cnt]
            v_depths[batch_idx, camera_idx, proj_filter[batch_idx, camera_idx]] = v_depths_culling[batch_idx, camera_idx, :cnt]
            v_colors[batch_idx, camera_idx, proj_filter[batch_idx, camera_idx]] = v_colors_culling[batch_idx, camera_idx, :cnt]
            v_opacities[batch_idx, camera_idx, proj_filter[batch_idx, camera_idx]] = v_opacities_culling[batch_idx, camera_idx, :cnt]
    
    covar2d_inv = torch.stack([conics[..., 0], conics[..., 1], conics[..., 1],\
                            conics[..., 2]], dim=-1).reshape(conics.shape[:-1] + (2, 2))
    v_covar2d_inv = torch.stack([v_conics[..., 0], v_conics[..., 1] * 0.5, v_conics[..., 1] * 0.5,\
                                v_conics[..., 2]], dim=-1).reshape(conics.shape[:-1] + (2, 2))
    
    v_covar2d = inverse_vjp(covar2d_inv, v_covar2d_inv)

    r = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3]
    covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
    mean_c = pos_w2c(r, t, means)
    covar_c = covar_w2c(r, covars)

    v_mean_c, v_covar_c = _persp_proj_vjp(mean_c, covar_c, ks, width, height, v_covar2d, v_means2d)
    v_mean_c[..., 2] = v_mean_c[..., 2] + v_depths
    
    v_r, v_t, v_p_w = pos_w2c_vjp(r, t, means, v_mean_c)
    v_r, v_covar = covar_w2c_vjp(r, covars, v_covar_c, v_r) # v_r [b,c,3,3], v_covar[b,n,3,3]

    rotmat = _quat_to_rotmat(quats)
    v_quats, v_scales = _quat_scale_to_covar_vjp(quats, scales, rotmat, v_covar)
    
    if compensations is not None:
        v_opacities = v_opacities * compensations
    v_colors = v_colors.sum(dim=1).permute(0, 2, 1).contiguous()
    v_opacities = v_opacities.sum(dim=1)

    return v_p_w, v_quats, v_scales, v_r, v_colors, v_opacities


class TestProjection3DGSBackward(TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.test_cases = [
            [1, 1, 10000, 64, 64],
            [1, 8, 80000, 64, 64],
            [1, 16, 160000, 64, 64]
            ]
        self.test_results = self.gen_results()

    def gen_inputs(self, shape):
        batch_size, camera_num, gaussian_num = shape

        means = torch.rand(batch_size, 3, gaussian_num).float()
        quats = torch.rand(batch_size, 4, gaussian_num).float()
        scales = torch.rand(batch_size, 3, gaussian_num).float()
        conics = torch.rand(batch_size, camera_num, 3, gaussian_num).float()
        viewmats = torch.zeros(batch_size, camera_num, 4, 4).float()
        r = torch.rand(batch_size, camera_num, 3, 3).float()
        r = torch.matmul(r, r.transpose(-2, -1)).float()
        t = torch.rand(batch_size, camera_num, 3).float()
        viewmats[:, :, :3, :3] = r
        viewmats[:, :, :3, 3] = t
        viewmats[:, :, 3, 3] = 1
        fx, fy, s, x0, y0 = 1.611400e3, 1.611400e3, 0, 7.79500e2, 5.19500e2
        k = torch.tensor([[[[fx, s, x0], [0, fy, y0], [0, 0, 1]]]], dtype=torch.float32)
        ks = k.expand(batch_size, camera_num, 3, 3)
        v_means2d = torch.rand(batch_size, camera_num, 2, gaussian_num).float()
        v_depths = torch.rand(batch_size, camera_num, gaussian_num).float()
        v_conics = torch.rand(batch_size, camera_num, 3, gaussian_num).float()
        v_colors_culling = torch.rand(batch_size, camera_num, 3, gaussian_num).float()
        v_opacities_culling = torch.rand(batch_size, camera_num, gaussian_num).float()
        proj_filter = torch.randint(0, 256, (batch_size, camera_num, (gaussian_num + 7) // 8), dtype=torch.uint8)
        compensations = torch.rand(batch_size, camera_num, gaussian_num).float()
        
        return Inputs(means, quats, scales, conics, viewmats, ks, v_means2d, v_depths,\
                      v_conics, v_colors_culling, v_opacities_culling, proj_filter, compensations), \
               Inputs(means, quats, scales, conics, viewmats, ks, v_means2d, v_depths,\
                      v_conics, v_colors_culling, v_opacities_culling, proj_filter, None)

    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            inputs1, inputs2 = self.gen_inputs(test_case[:3])
            width, height = test_case[3:]
            cpu_results1 = self.cpu_to_exec(inputs1, width, height)
            npu_results1 = self.npu_to_exec(inputs1, width, height)
            test_results.append((cpu_results1, npu_results1))
            
            cpu_results2 = self.cpu_to_exec(inputs2, width, height)
            npu_results2 = self.npu_to_exec(inputs2, width, height)
            test_results.append((cpu_results2, npu_results2))
        return test_results

    def cpu_to_exec(self, inputs, width, height):
        means = inputs.means
        quats = inputs.quats
        scales = inputs.scales
        conics = inputs.conics
        viewmats = inputs.viewmats
        ks = inputs.ks
        v_means2d = inputs.v_means2d
        v_depths = inputs.v_depths
        v_conics = inputs.v_conics
        v_colors_culling = inputs.v_colors_culling
        v_opacities_culling = inputs.v_opacities_culling
        proj_filter = inputs.proj_filter
        compensations = inputs.compensations

        v_p_w, v_quats, v_scales, v_r, v_colors, v_opacities = _fully_fused_projection_bwd(means,\
            quats, scales, conics, viewmats, ks, v_means2d, v_depths, v_conics, v_colors_culling,\
            v_opacities_culling, proj_filter, compensations, width, height)

        return ExecResults(
            v_p_w=v_p_w.detach().float(),
            v_quats=v_quats.detach().float(),
            v_scales=v_scales.detach().float(),
            v_r=v_r.detach().float(),
            v_colors=v_colors.detach().float(),
            v_opacities=v_opacities.detach().float(),
        )


    def npu_to_exec(self, inputs, width, height):
        means = inputs.means.npu()
        quats = inputs.quats.npu()
        scales = inputs.scales.npu()
        conics = inputs.conics.npu()
        viewmats = inputs.viewmats.npu()
        ks = inputs.ks.npu()
        v_means2d = inputs.v_means2d.npu()
        v_depths = inputs.v_depths.npu()
        v_conics = inputs.v_conics.npu()
        v_colors_culling = inputs.v_colors_culling.npu()
        v_opacities_culling = inputs.v_opacities_culling.npu()
        proj_filter = inputs.proj_filter.npu()
        if inputs.compensations is not None:
            compensations = inputs.compensations.npu()
        else:
            compensations = None
        b, _, n = means.shape
        c = viewmats.shape[1]
        rast_ctx = RastCtx()
        rast_ctx.proj_backward_culling = False
        rast_ctx.prealloc_if_need(b, c, n, means.device)
        outputs = rast_ctx.outputs_for_proj_backward()
        for output in outputs:
            output.zero_()
        v_p_w, v_quats, v_scales, v_r, v_colors, v_opacities, *_ = fully_fused_projection_bwd(
            means, quats, scales, conics, viewmats, ks, v_means2d, v_depths, v_conics,
            v_colors_culling, v_opacities_culling, proj_filter, compensations,
            None, None, None, None, None, None, None,
            width, height, 0, 0, outputs)
        return ExecResults(
            v_p_w=v_p_w.detach().float(),
            v_quats=v_quats.detach().float(),
            v_scales=v_scales.detach().float(),
            v_r=v_r.detach().float(),
            v_colors=v_colors.detach().float(),
            v_opacities=v_opacities.detach().float(),
        )

    def test_projection_3dgs_bwd(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.v_p_w.numpy(), npu_results.v_p_w.cpu().numpy(), prec=1.e-2)
            self.assertRtolEqual(cpu_results.v_quats.numpy(), npu_results.v_quats.cpu().numpy(), prec=2.e-1)
            self.assertRtolEqual(cpu_results.v_scales.numpy(), npu_results.v_scales.cpu().numpy(), prec=5.e-1)
            self.assertRtolEqual(cpu_results.v_colors.numpy(), npu_results.v_colors.cpu().numpy(), prec=1.e-3)
            self.assertRtolEqual(cpu_results.v_opacities.numpy(), npu_results.v_opacities.cpu().numpy(), prec=1.e-3)

class TestProjectionAutogradEndToEnd(unittest.TestCase):
    """Exercise public forward -> filter -> loss.backward with real NPU ops."""

    def check_projection(self, n, compensation=False, all_culled=False):
        from gauss_splat import projection_three_dims_gaussian_fused
        from test_gaussian_filter import _gaussian_filter
        from test_projection_3dgs_forward import _world_to_cam, _persp_proj

        torch.manual_seed(2026)
        b, c, width, height = 2, 2, 64, 64
        means = torch.rand(b, n, 3) - 0.5
        means[..., 2] = torch.rand(b, n) + 3
        # Holes and differing camera counts exercise compact-index row strides.
        means[:, 1::5, 2] = -2
        means[:, 2::7, 0] = 30
        if all_culled:
            means[..., 2] = -2
        quats = torch.rand(b, n, 4) + 0.2
        scales = torch.rand(b, n, 3) * 0.15 + 0.1
        colors = torch.rand(b, 3, n)
        opacities = torch.rand(b, n) * 0.5 + 0.4
        leaves = [t.requires_grad_() for t in (means, quats, scales, colors, opacities)]
        device_leaves = [t.detach().npu().requires_grad_() for t in leaves]
        views = torch.eye(4).repeat(b, c, 1, 1)
        views[:, 1, 0, 3] = 1.5
        views[1, :, 1, 3] = 0.2
        ks = torch.tensor([[40., 0., 32.], [0., 40., 32.], [0., 0., 1.]]).repeat(b, c, 1, 1)

        # Independent differentiable CPU reference; no native/NPU calls.
        covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False)
        means_c, covars_c = _world_to_cam(means, covars, views)
        means2d, covars2d = _persp_proj(means_c, covars_c, ks, width, height)
        det_orig = torch.linalg.det(covars2d)
        covars2d = covars2d + torch.eye(2) * 0.3
        det = torch.linalg.det(covars2d).clamp_min(1e-10)
        conics = torch.stack((covars2d[..., 1, 1], -covars2d[..., 0, 1],
                              covars2d[..., 0, 0]), -1) / det[..., None]
        half_trace = covars2d.diagonal(dim1=-2, dim2=-1).sum(-1) / 2
        bound = 3.33 * (half_trace + (half_trace.square() - det).clamp_min(0.01).sqrt()).sqrt()
        radius = torch.minimum(3.33 * covars2d.diagonal(dim1=-2, dim2=-1).sqrt(),
                               bound[..., None]).ceil()
        packed_covars = torch.stack((covars2d[..., 0, 0], covars2d[..., 0, 1],
                                     covars2d[..., 1, 1]), -1)
        comp = (det_orig / det).clamp_min(0).sqrt() if compensation else None
        # Current backward treats compensation as a saved opacity multiplier,
        # without differentiating compensation through covariance.
        comp = comp.detach() if comp is not None else None
        expected_filter = _gaussian_filter(
            means.transpose(1, 2), colors, det, opacities,
            means2d.transpose(2, 3), means_c[..., 2], radius.transpose(2, 3),
            conics.transpose(2, 3), packed_covars.transpose(2, 3), comp,
            width, height, 0.01, 100.)
        expected = tuple(expected_filter[i] for i in (2, 3, 6, 7, 4, 5, 1, 9, 8))
        mn, qn, sn, cn, on = device_leaves
        actual = projection_three_dims_gaussian_fused(
            mn, cn, None, qn, sn, on, views.npu(), ks.npu(),
            torch.ones((b, c, n), device=mn.device), torch.ones((1,), device=mn.device),
            None, width, height, 0.3, 0.01, 100., compensation, 'pinhole')
        # Compact culling buffers only define entries [0, cnt); the operator
        # leaves the tail unspecified, so mask it before comparing/backprop.
        cnt = actual[7]
        def _mask_compact(value):
            keep = torch.arange(value.shape[-1], device=value.device) < cnt[..., None]
            if value.dim() == 4:
                keep = keep[:, :, None, :]
            return torch.where(keep, value, torch.zeros_like(value))
        masked = [_mask_compact(actual[i]) for i in range(7)]
        for i, (result, reference) in enumerate(zip(masked + [actual[7]], expected[:8])):
            torch.testing.assert_close(result.cpu(), reference, atol=1e-4, rtol=1e-4,
                                       msg=lambda message, i=i: f'forward output {i}: {message}')
        if all_culled:
            self.assertEqual(actual[7].sum().item(), 0)
        else:
            self.assertTrue(bool((actual[7] > 0).all()))
            self.assertTrue(bool((actual[7] < n).all()))
        # Only outputs supported by the public backward contract enter loss.
        cpu_loss, npu_loss = 0, 0
        for i in (0, 1, 2, 3, 6):
            grad = torch.randn_like(expected[i])
            cpu_loss = cpu_loss + (expected[i] * grad).sum()
            npu_loss = npu_loss + (masked[i] * grad.npu()).sum()
        cpu_loss.backward()
        npu_loss.backward()
        for name, cpu, npu in zip(('means', 'quats', 'scales', 'colors', 'opacities'),
                                   leaves, device_leaves):
            self.assertIsNotNone(npu.grad, name)
            torch.testing.assert_close(npu.grad.cpu(), cpu.grad, atol=2e-4, rtol=2e-3,
                                       msg=lambda message: f'{name} gradient: {message}')

    def test_public_autograd_tail(self):
        self.check_projection(65)

    def test_public_autograd_multiple_tiles(self):
        self.check_projection(769)

    def test_public_autograd_compensation(self):
        self.check_projection(65, compensation=True)

    def test_public_autograd_all_culled(self):
        self.check_projection(65, all_culled=True)


if __name__ == "__main__":
    run_tests()
