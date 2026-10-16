# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Dict, NamedTuple, Optional, Tuple

import torch
from torch import Tensor

import acl

from .spherical_harmonics import spherical_harmonics
from .projection_three_dims_gaussian_fused import projection_three_dims_gaussian_fused
from .flash_gaussian_build_mask import flash_gaussian_build_mask
from .gaussian_sort import gaussian_sort
from .calc_render import calc_render
from .get_render_schedule import get_render_schedule_cpp


class _ProjectionInputs(NamedTuple):
    means: Tensor
    quats: Tensor
    scales: Tensor
    opacities: Tensor
    viewmats: Tensor
    intrinsics: Tensor
    width: int
    height: int


class _ProjectionOutputs(NamedTuple):
    means2d: Tensor
    depths: Tensor
    conics: Tensor
    opacities: Tensor
    radius: Tensor
    covars2d: Tensor
    colors: Tensor
    cnt: Tensor
    proj_filter: Tensor


class _RenderSchedule(NamedTuple):
    sorted_offset: Tensor
    lb_sched_tensor: Tensor
    sorted_gs_ids: Tensor


class _RenderView(NamedTuple):
    batch_id: int
    camera_index: int
    size: Tuple


def _compute_sh_colors(
    means: Tensor,
    camtoworlds: Tensor,
    shs: Tensor,
    sh_degree: int,
) -> Tuple[Tensor, bool]:
    camera_count = camtoworlds.shape[0]
    camera_dependent_sh = camera_count > 1 and sh_degree > 0
    sh_camera_count = camera_count if camera_dependent_sh else 1
    rays_o = camtoworlds[:sh_camera_count, :3, 3]
    rays_d = means.unsqueeze(0) - rays_o.unsqueeze(1)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    coefficient_count = (sh_degree + 1) ** 2
    colors = spherical_harmonics(
        sh_degree,
        rays_d.reshape(sh_camera_count, means.shape[0], 3),
        shs[:sh_camera_count, :, :coefficient_count, :].reshape(
            sh_camera_count, means.shape[0], coefficient_count, 3
        ),
    )
    return (colors + 0.5).clip(min=0.0), camera_dependent_sh


def _project_sh_colors(
    inputs: _ProjectionInputs,
    colors: Tensor,
    camera_dependent_sh: bool,
) -> Tuple[Tensor, ...]:
    if not camera_dependent_sh:
        return projection_three_dims_gaussian_fused(
            inputs.means,
            colors[:1],
            None,
            inputs.quats,
            inputs.scales,
            inputs.opacities,
            inputs.viewmats,
            inputs.intrinsics,
            inputs.width,
            inputs.height,
            0.3,
            0.2,
        )

    camera_results = []
    for camera_index in range(inputs.viewmats.shape[1]):
        camera_results.append(
            projection_three_dims_gaussian_fused(
                inputs.means,
                colors.narrow(0, camera_index, 1),
                None,
                inputs.quats,
                inputs.scales,
                inputs.opacities,
                inputs.viewmats.narrow(1, camera_index, 1),
                inputs.intrinsics.narrow(1, camera_index, 1),
                inputs.width,
                inputs.height,
                0.3,
                0.2,
            )
        )
    return tuple(
        torch.cat([result[index] for result in camera_results], dim=1)
        for index in range(len(camera_results[0]))
    )


def _project_rasterization(
    gs: Tuple,
    camera: Tuple,
    size: Tuple,
    sh_degree: Optional[int],
) -> _ProjectionOutputs:
    means, quats, scales, opacities, colors = gs
    viewmats, intrinsics, camera_model, render_mode = camera
    gaussian_count = means.shape[0]
    camera_count = viewmats.shape[0]
    _validate_inputs(gs, camera, sh_degree, gaussian_count, camera_count)
    width, height = size
    camtoworlds = torch.inverse(viewmats)
    shs = colors.expand(camera_count, -1, -1, -1) if colors.dim() == 3 else colors
    colors, camera_dependent_sh = _compute_sh_colors(
        means, camtoworlds, shs, sh_degree
    )
    projection_inputs = _ProjectionInputs(
        means.reshape(1, gaussian_count, 3),
        quats.reshape(1, gaussian_count, 4),
        scales.reshape(1, gaussian_count, 3),
        opacities.reshape(1, gaussian_count),
        viewmats.reshape(1, camera_count, 4, 4).contiguous(),
        intrinsics.reshape(1, camera_count, 3, 3),
        width,
        height,
    )
    return _ProjectionOutputs(
        *_project_sh_colors(projection_inputs, colors, camera_dependent_sh)
    )


def _build_render_schedule(
    projection: _ProjectionOutputs,
    tile_grid: Tensor,
    size: Tuple,
    tile_size: int,
) -> _RenderSchedule:
    width, height = size
    with torch.no_grad():
        tile_sums, tile_offsets, tile_depths, tile_gauss_ids = (
            flash_gaussian_build_mask(
                projection.means2d,
                projection.opacities.unsqueeze(2),
                projection.conics,
                projection.covars2d,
                projection.depths.unsqueeze(2),
                projection.cnt.unsqueeze(2),
                tile_grid.float(),
                width,
                height,
                tile_size,
            )
        )
        sorted_cnts = tile_offsets.squeeze(-1)[:, :, -1]
        sorted_offset = torch.cumsum(sorted_cnts.flatten(), dim=0)
        tile_sums_cpu = tile_sums.squeeze(-1).cpu().to(torch.int64)
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
    return _RenderSchedule(sorted_offset, lb_sched_tensor, sorted_gs_ids)


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
    viewmats, intrinsics, camera_model, render_mode = camera
    assert_check(means.shape == (N, 3), f"Invalid shape for means: {means.shape}")
    assert_check(quats.shape == (N, 4), f"Invalid shape for quats: {quats.shape}")
    assert_check(scales.shape == (N, 3), f"Invalid shape for scales: {scales.shape}")
    assert_check(opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}")
    assert_check(viewmats.shape == (C, 4, 4), f"Invalid shape for viewmats: {viewmats.shape}")
    assert_check(
        intrinsics.shape == (C, 3, 3),
        f"Invalid shape for intrinsics: {intrinsics.shape}",
    )
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
        self._geometry_key = None

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
        device = splats["means"].device
        geometry_key = (width, height, tile_size, device)
        if self._geometry_key != geometry_key:
            self.tile_size = tile_size
            self.tilenum_height = math.ceil(height / tile_size)
            self.tilenum_width = math.ceil(width / tile_size)
            self.padded_height = self.tilenum_height * tile_size
            self.padded_width = self.tilenum_width * tile_size
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size),
                torch.arange(0, self.padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width),
                torch.arange(self.padded_height), indexing='xy'), dim=-1).to(device)
            self._geometry_key = geometry_key

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

    def _render_camera(
        self,
        projection: _ProjectionOutputs,
        schedule: _RenderSchedule,
        view: _RenderView,
    ) -> Tuple[Tensor, Tensor]:
        width, height = view.size
        camera_count = projection.means2d.shape[1]
        gs_tuple = (
            projection.means2d,
            projection.colors,
            projection.opacities,
            projection.conics,
        )
        cf_gs, pix_coords = self.get_render_input(
            gs_tuple, projection.depths, view.camera_index
        )
        (
            cf_means2,
            cf_colors3,
            cf_opacity,
            inv_x_0,
            inv_x_1,
            inv_x_2,
            cf_depths,
        ) = cf_gs
        view_idx = view.batch_id * camera_count + view.camera_index
        start_index = (
            schedule.sorted_offset[view_idx - 1] if view_idx > 0 else 0
        )
        end_index = schedule.sorted_offset[view_idx]
        sorted_gs_id = schedule.sorted_gs_ids[start_index:end_index]
        lb_sched = schedule.lb_sched_tensor[
            view.batch_id, view.camera_index, :
        ]
        render_colors, render_depths = calc_render(
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
        render_colors = self.tile2image(render_colors.permute(1, 2, 0), height, width)
        render_depths = self.tile2image(render_depths.permute(1, 2, 0), height, width)
        return render_colors.permute(1, 2, 0), render_depths.permute(1, 2, 0)

    def _render_outputs(
        self,
        projection: _ProjectionOutputs,
        schedule: _RenderSchedule,
        size: Tuple,
    ) -> Tuple[Tensor, Tensor]:
        width, height = size
        batch_count, camera_count = projection.means2d.shape[:2]
        render_colors = []
        render_depths = []
        for batch_id in range(batch_count):
            for camera_index in range(camera_count):
                cf_render_colors, cf_render_depths = self._render_camera(
                    projection,
                    schedule,
                    _RenderView(batch_id, camera_index, size),
                )
                render_colors.append(cf_render_colors)
                render_depths.append(cf_render_depths)
        return torch.stack(render_colors), torch.stack(render_depths)

    def _rasterization(
        self,
        gs,
        camera,
        size: Tuple,
        sh_degree: Optional[int] = None,
        tile_size: int = 32,
    ) -> Tuple[Tensor, Tensor, Dict]:
        projection = _project_rasterization(gs, camera, size, sh_degree)
        schedule = _build_render_schedule(
            projection, self.tile_grid, size, tile_size
        )
        render_colors, render_depths = self._render_outputs(projection, schedule, size)
        means, *_ = gs
        width, height = size

        meta = {
            "gaussian_ids": None,
            "means2d": projection.means2d,
            "radii": projection.radius,
            "width": width,
            "height": height,
            "n_cameras": camera[0].shape[0],
            "proj_filter": projection.proj_filter,
            "n_gaussians": means.shape[0],
        }
        return render_colors, render_depths, meta
