# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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


def _validate_inputs(gs, camera, sh_degree, N, C):
    means, quats, scales, opacities, colors = gs
    viewmats, Ks, camera_model, render_mode = camera
    assert means.shape == (N, 3), f"Invalid shape for means: {means.shape}"
    assert quats.shape == (N, 4), f"Invalid shape for quats: {quats.shape}"
    assert scales.shape == (N, 3), f"Invalid shape for scales: {scales.shape}"
    assert opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}"
    assert viewmats.shape == (C, 4, 4), f"Invalid shape for viewmats: {viewmats.shape}"
    assert Ks.shape == (C, 3, 3), f"Invalid shape for Ks: {Ks.shape}"
    assert camera_model in ["pinhole", "ortho", "fisheye"], f"Invalid camera_model: {camera_model}"
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], f"Invalid render_mode: {render_mode}"

    if sh_degree is None:
        assert (colors.dim() == 2 and colors.shape[0] == N) or \
               (colors.dim() == 3 and colors.shape[:2] == (C, N)), \
               f"Invalid shape for colors: {colors.shape}"
    else:
        assert (colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3) or \
               (colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3), \
               f"Invalid shape for colors: {colors.shape}"
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], f"Invalid sh_degree for colors shape: {colors.shape}"


class Rasterizer:
    def __init__(self) -> None:
        self.tile_size = 32
        self.pix_coord = None
        self.padded_width = None
        self.padded_height = None
        self.tile_grid = None
        self.tilenum_height = None
        self.tilenum_width = None

    def tile2image(self, rendered_image, height, width, channel_dim=3):
        ts = self.tile_size
        nh = self.tilenum_height
        nw = self.tilenum_width
        rendered_image = rendered_image.reshape(nh, nw, ts, ts, -1).transpose(1, 2).reshape(nh * ts, nw * ts, -1)
        return rendered_image.permute(2, 0, 1)[:, :height, :width]

    def get_render_input(self, gs, depths, tile_offsets, _cam_view):
        means2d, colors, opacities, conics = gs
        cf_means2 = means2d[0, _cam_view]
        cf_colors3 = colors[0, _cam_view]
        cf_opacity = opacities[0, _cam_view]

        inv_x_0 = conics[0, _cam_view, 0, :]
        inv_x_1 = conics[0, _cam_view, 1, :]
        inv_x_2 = conics[0, _cam_view, 2, :]

        cf_depths = depths[0, _cam_view:_cam_view+1]
        ts = self.tile_size
        height = self.padded_height
        width = self.padded_width
        pix_coords = self.pix_coord.reshape(height // ts, ts, width // ts, ts, 2) \
            .permute(0, 2, 1, 3, 4).reshape(height // ts * width // ts, ts * ts, 2) \
            .permute(0, 2, 1).to(torch.float32).contiguous()
        nums = torch.cat([tile_offsets[_cam_view][:1], tile_offsets[_cam_view][1:] - tile_offsets[_cam_view][:-1]])
        lb_sched = get_render_schedule_cpp(nums.cpu().to(torch.int64),
            acl.get_device_capability(0, 1)[0]).clone().detach().to(torch.int64).npu()

        cf_gs = cf_means2, cf_colors3, cf_opacity, inv_x_0, inv_x_1, inv_x_2, cf_depths
        return (cf_gs, pix_coords, lb_sched)

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
            sorted_gs_ids = []
            tile_offsets = []
            for _cam_view in range(0, C):
                mask = flash_gaussian_build_mask(
                    means2d[:, _cam_view:_cam_view+1, :, :],
                    opacities[:, _cam_view:_cam_view+1, :].unsqueeze(0),
                    conics[:, _cam_view:_cam_view+1, :, :],
                    covars2d[:, _cam_view:_cam_view+1, :, :],
                    cnt[:, _cam_view:_cam_view+1].unsqueeze(0),
                    self.tile_grid.float(), width, height, tile_size)
                cf_sorted_gs_ids, cf_tile_offsets = gaussian_sort(mask[0, 0], depths[0, _cam_view])
                sorted_gs_ids.append(cf_sorted_gs_ids)
                tile_offsets.append(cf_tile_offsets)

        render_colors = []
        render_depths = []
        for _cam_view in range(0, C):
            gs = means2d, colors, opacities, conics
            cf_gs, pix_coords, lb_sched = self.get_render_input(gs, depths, tile_offsets, _cam_view)
            cf_means2, cf_colors3, cf_opacity, inv_x_0, inv_x_1, inv_x_2, cf_depths = cf_gs

            cf_render_colors, cf_render_depths = calc_render(cf_means2,
                                                            inv_x_0, inv_x_1, inv_x_2,
                                                            cf_opacity,
                                                            cf_colors3,
                                                            cf_depths,
                                                            pix_coords,
                                                            lb_sched,
                                                            sorted_gs_ids[_cam_view]
                                                            )
            cf_render_colors = self.tile2image(cf_render_colors.permute(1, 2, 0), height, width)
            cf_render_depths = self.tile2image(cf_render_depths.permute(1, 2, 0), height, width)

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