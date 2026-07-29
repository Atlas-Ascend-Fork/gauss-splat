# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

import acl

from .spherical_harmonics import spherical_harmonics
from .projection_three_dims_gaussian_fused import projection_three_dims_gaussian_fused
from .flash_gaussian_build_mask import flash_gaussian_build_mask
from .gaussian_sort import gaussian_sort
from .calc_render import calc_render
from .get_render_schedule import get_render_schedule_cpp


def assert_check(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_inputs(
    gs: Tuple,
    camera: Tuple,
    sh_degree: Optional[int],
    N: int,
    C: int
) -> None:
    means, quats, scales, opacities, colors = gs
    viewmats, Ks, camera_model, render_mode = camera
    assert_check(means.shape == (N, 3), f"Invalid shape for means: {means.shape}")
    assert_check(quats.shape == (N, 4), f"Invalid shape for quats: {quats.shape}")
    assert_check(scales.shape == (N, 3), f"Invalid shape for scales: {scales.shape}")
    assert_check(opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}")
    assert_check(viewmats.shape == (C, 4, 4), f"Invalid shape for viewmats: {viewmats.shape}")
    assert_check(Ks.shape == (C, 3, 3), f"Invalid shape for Ks: {Ks.shape}")
    assert_check(camera_model in ["pinhole", "ortho", "fisheye"], f"Invalid camera_model: {camera_model}")
    assert_check(render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], f"Invalid render_mode: {render_mode}")

    if sh_degree is None:
        assert_check((colors.dim() == 2 and colors.shape[0] == N) or
                    (colors.dim() == 3 and colors.shape[:2] == (C, N)),
                    f"Invalid shape for colors: {colors.shape}")
    else:
        assert_check((colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or
                    (colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3),
                    f"Invalid shape for colors: {colors.shape}")

        assert_check((sh_degree + 1) ** 2 <= colors.shape[-2], f"Invalid sh_degree for colors shape: {colors.shape}")


class Rasterizer:
    def __init__(self) -> None:
        self.tile_size = 32
        self.pix_coord = None
        self.padded_width = None
        self.padded_height = None
        self.tile_grid = None
        self.tilenum_height = None
        self.tilenum_width = None

    def tile2image(self, rendered_image, height, width):
        ts = self.tile_size
        nh = self.tilenum_height
        nw = self.tilenum_width
        rendered_image = rendered_image.reshape(nh, nw, ts, ts, -1).transpose(1, 2).reshape(nh * ts, nw * ts, -1)
        return rendered_image.permute(2, 0, 1)[:, :height, :width]

    def get_render_input(self, gs, depths, _cam_view):
        means2d, colors, opacities, conics = gs
        cf_means2 = means2d[0, _cam_view]
        cf_colors3 = colors[0, _cam_view]
        cf_opacity = opacities[0, _cam_view]

        inv_x_0 = conics[0, _cam_view, 0, :]
        inv_x_1 = conics[0, _cam_view, 1, :]
        inv_x_2 = conics[0, _cam_view, 2, :]

        cf_depths = depths[0, _cam_view].unsqueeze(0)
        ts = self.tile_size
        height = self.padded_height
        width = self.padded_width
        pix_coords = self.pix_coord.reshape(height // ts, ts, width // ts, ts, 2) \
            .permute(0, 2, 1, 3, 4).reshape(height // ts * width // ts, ts * ts, 2) \
            .permute(0, 2, 1).to(torch.float32).contiguous()
        cf_gs = cf_means2, cf_colors3, cf_opacity, inv_x_0, inv_x_1, inv_x_2, cf_depths
        return (cf_gs, pix_coords)

    def rasterization(
        self,
        cam: Tuple,
        size: Tuple,
        tile_size: int,
        active_sh_degree: int,
        splats: dict,
        camera_model: str = "pinhole",
    ) -> Tuple[Tensor, Tensor, Dict]:
        width, height = size
        if self.tile_grid is None:
            self.tile_size = tile_size
            self.tilenum_height = math.ceil(height / tile_size)
            self.tilenum_width = math.ceil(width / tile_size)
            self.padded_height = self.tilenum_height * tile_size
            self.padded_width = self.tilenum_width * tile_size
            device = splats["means"].device
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size),
                torch.arange(0, self.padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width),
                torch.arange(self.padded_height), indexing='xy'), dim=-1).to(device)

        means = splats["means"]
        quats = splats["quats"]
        scales = torch.exp(splats["scales"])
        opacities = torch.sigmoid(splats["opacities"])

        colors = torch.cat([splats["sh0"], splats["shN"]], 1)

        camtoworlds, Ks, render_mode = cam
        viewmats = torch.linalg.inv(camtoworlds)
        render_colors, render_depth, info = self._rasterization(
            gs=(means, quats, scales, opacities, colors),
            camera=(viewmats, Ks, camera_model, render_mode),
            size=size,
            sh_degree=active_sh_degree,
            tile_size=tile_size,
        )
        return render_colors, render_depth, info

    def _rasterization(
        self,
        gs,
        camera,
        size: Tuple,
        sh_degree: Optional[int] = None,
        tile_size: int = 32,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means, quats, scales, opacities, colors = gs
        viewmats, Ks, camera_model, render_mode = camera

        N = means.shape[0]
        C = viewmats.shape[0]
        B = 1
        _validate_inputs(gs, camera, sh_degree, N, C)
        width, height = size

        camtoworlds = torch.inverse(viewmats)
        if colors.dim() == 3:
            shs = colors.expand(C, -1, -1, -1)
        else:
            shs = colors

        rays_o = camtoworlds[0, :3, 3]
        rays_d = means - rays_o
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        k = (sh_degree + 1) ** 2
        colors = spherical_harmonics(sh_degree, rays_d.reshape(B, N, 3), shs[0, :, :k, :].reshape(B, N, k, 3))
        colors = (colors + 0.5).clip(min=0.0)

        means2d, depths, conics, opacities, radius, covars2d, colors, cnt = projection_three_dims_gaussian_fused(
            means.reshape(B, N, 3),
            colors,
            None,
            quats.reshape(B, N, 4),
            scales.reshape(B, N, 3),
            opacities.reshape(B, N),
            viewmats.reshape(B, C, 4, 4).contiguous(),
            Ks.reshape(B, C, 3, 3),
            width,
            height,
            0.3,
            0.2
        )

        with torch.no_grad():
            tile_sums, tile_offsets, tile_depths, tile_gauss_ids = (
                flash_gaussian_build_mask(
                    means2d,
                    opacities.unsqueeze(2),
                    conics,
                    covars2d,
                    depths.unsqueeze(2),
                    cnt.unsqueeze(2),
                    self.tile_grid.float(),
                    width,
                    height,
                    tile_size,
                )
            )
            sorted_cnts = tile_offsets.squeeze(-1)[:, :, -1]
            sorted_cnts_flatten = sorted_cnts.flatten()
            sorted_offset = torch.cumsum(sorted_cnts_flatten, dim=0)
            tile_sums_3d = tile_sums.squeeze(-1)
            tile_sums_cpu = tile_sums_3d.cpu().to(torch.int64)

            vector_num = acl.get_device_capability(0, 1)[0]
            lb_sched_tensor_cpu = get_render_schedule_cpp(tile_sums_cpu, vector_num)
            lb_sched_tensor = lb_sched_tensor_cpu.npu()
            max_tile_gauss = torch.amax(tile_sums).item()

            sorted_gs_ids = gaussian_sort(
                lb_sched_tensor,
                tile_sums,
                tile_depths,
                tile_gauss_ids,
                sorted_offset,
                max_tile_gauss,
            )

        render_colors = []
        render_depths = []
        for _batch_id in range(0, B):
            for _cam_view in range(0, C):
                gs_tuple = means2d, colors, opacities, conics
                cf_gs, pix_coords = self.get_render_input(gs_tuple, depths, _cam_view)
                (
                    cf_means2,
                    cf_colors3,
                    cf_opacity,
                    inv_x_0,
                    inv_x_1,
                    inv_x_2,
                    cf_depths,
                ) = cf_gs
                view_idx = _batch_id * C + _cam_view
                start_index = sorted_offset[view_idx - 1] if view_idx > 0 else 0
                end_index = sorted_offset[view_idx]
                sorted_gs_id = sorted_gs_ids[start_index:end_index]
                lb_sched = lb_sched_tensor[_batch_id, _cam_view, :]

                cf_render_colors, cf_render_depths = calc_render(
                    cf_means2,
                    inv_x_0,
                    inv_x_1,
                    inv_x_2,
                    cf_opacity,
                    cf_colors3,
                    cf_depths,
                    pix_coords,
                    lb_sched,
                    sorted_gs_id,
                )
                cf_render_colors = self.tile2image(
                    cf_render_colors.permute(1, 2, 0), height, width
                )
                cf_render_depths = self.tile2image(
                    cf_render_depths.permute(1, 2, 0), height, width
                )

                render_colors.append(cf_render_colors.permute(1, 2, 0))
                render_depths.append(cf_render_depths.permute(1, 2, 0))
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)

        meta = {
            "gaussian_ids": None,
            "means2d": means2d,
            "radii": radius,
            "width": width,
            "height": height,
            "n_cameras": C,
        }
        return render_colors, render_depths, meta