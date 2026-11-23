# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from typing import Dict, Optional, Tuple, NamedTuple

import torch
from torch import Tensor

import acl

from .spherical_harmonics import spherical_harmonics
from .sh_preprocess import sh_preprocess
from .projection_three_dims_gaussian_fused import projection_three_dims_gaussian_fused
from .flash_gaussian_build_mask import flash_gaussian_build_mask
from .gaussian_sort import gaussian_sort
from .calc_render import calc_render
from .get_render_schedule import get_render_schedule_cpp
from .rast_ctx import RastCtx


class GaussianData(NamedTuple):
    means: Tensor                        # 高斯球三维坐标
    quats: Tensor                        # 高斯球旋转四参数
    scales: Tensor                       # 高斯球缩放参数
    opacities: Tensor                    # 高斯球透明度
    features_dc: Optional[Tensor]        # 高斯球球谐DC系数
    features_sh: Optional[Tensor]        # 高斯球球谐SH高阶系数


class CameraData(NamedTuple):
    width: int                           # 渲染图像宽度
    height: int                          # 渲染图像高度
    w2c: Tensor                          # 相机外参
    ks: Tensor                           # 相机内参
    camera_model: str = "pinhole"        # 相机模型，支持：pinhole / ortho / fisheye
    exposure: Optional[Tensor] = None    # 曝光矩阵，支持：全局模式, 格网模式


class ConfigData(NamedTuple):
    sh_degree: Optional[int] = None      # 球谐系数
    tile_size: int = 32                  # 分块大小
    render_mode: str = "RGB"             # 渲染模式，支持：RGB，RGB+D
    scale_log: bool = True               # scale数据域，True: scale数据处于log域，内部将取exp操作
    opacity_logit: bool = True           # opacity数据域，True: opacity数据处于logit域，内部将取sigmoid操作
    channel_first: bool = False          # 数据布局方式, True：通道数据为第一维
    bg_color: Optional[Tensor] = None    # 背景色


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
    viewmats, ks, camera_model, render_mode = camera
    # Pure channel-first op: [D,N] / [K,3,N].
    assert_check(means.shape == (3, N), f"Invalid shape for means: {means.shape}")
    assert_check(quats.shape == (4, N), f"Invalid shape for quats: {quats.shape}")
    assert_check(scales.shape == (3, N), f"Invalid shape for scales: {scales.shape}")
    assert_check(opacities.shape == (N,), f"Invalid shape for opacities: {opacities.shape}")
    assert_check(viewmats.shape == (C, 4, 4), f"Invalid shape for viewmats: {viewmats.shape}")
    assert_check(ks.shape == (C, 3, 3), f"Invalid shape for ks: {ks.shape}")
    assert_check(camera_model in ["pinhole", "ortho", "fisheye"], f"Invalid camera_model: {camera_model}")
    assert_check(render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], f"Invalid render_mode: {render_mode}")

    if sh_degree is None:
        assert_check((colors.dim() == 2 and colors.shape[-1] == N) or
                    (colors.dim() == 3 and colors.shape[0] == C and colors.shape[-1] == N),
                    f"Invalid shape for colors: {colors.shape}")
    else:
        assert_check((colors.dim() == 3 and colors.shape[1] == 3 and colors.shape[2] == N) or
                    (colors.dim() == 4 and colors.shape[0] == C and colors.shape[2] == 3 and colors.shape[3] == N),
                    f"Invalid shape for colors: {colors.shape}")

        max_coeffs = (sh_degree + 1) ** 2
        num_coeffs = colors.shape[-3] if colors.dim() == 3 else colors.shape[1]
        assert_check(max_coeffs <= num_coeffs,
                    f"Invalid sh_degree for colors shape: {colors.shape}")


def _resolve_bg_color(splats: dict) -> Optional[Tensor]:
    """从 splats 字典解析可选背景色；None 表示关闭。"""
    device = splats["means"].device
    raw = splats.get("bg_color", None)
    if raw is None:
        return None
    if isinstance(raw, torch.Tensor):
        # 不 detach：允许 bg_color 为 nn.Parameter，由 CalcRender.backward 返回 v_bg。
        bg_color = raw.to(dtype=torch.float32, device=device).reshape(-1)
    else:
        bg_color = torch.as_tensor(raw, dtype=torch.float32, device=device).reshape(-1)
    if bg_color.numel() != 3:
        raise ValueError(
            f"splats['bg_color'] must have 3 elements, got {tuple(bg_color.shape)}")
    return bg_color.contiguous()


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
        self.rast_ctx = RastCtx()

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

    def rasterize(
        self,
        gs: GaussianData,
        camera: CameraData,
        config: ConfigData = ConfigData()
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Dict]:
        width, height = camera.width, camera.height
        device = gs.means.device
        tile_size = config.tile_size
        geometry_key = (math.ceil(width / tile_size) * tile_size,
                        math.ceil(height / tile_size) * tile_size,
                        tile_size, device)
        if self.tile_grid is None or self._geometry_key != geometry_key:
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
        return self._rasterization(gs, camera, config)

    def rasterization(
        self,
        cam: Tuple,
        size: Tuple,
        tile_size: int,
        active_sh_degree: int,
        splats: dict,
        camera_model: str = "pinhole",
    ) -> Tuple[Tensor, Tensor, Dict]:
        ### wrapper gaussian-relatived data
        colors = splats["sh"]
        gs = GaussianData(
            means=splats["means"],                          # [3, N]
            quats=splats["quats"],                          # [4, N]
            scales=splats["scales"],                        # [3, N]
            opacities=splats["opacities"],                  # [N]
            features_dc=colors[:1].contiguous(),            # [1, 3, N]
            features_sh=colors[1:].contiguous()             # [K-1, 3, N]
        )

        ### wrapper camera-relatived data
        camtoworlds, ks, render_mode = cam
        w2c = torch.linalg.inv(camtoworlds)
        camera = CameraData(
            width=size[0],
            height=size[1],
            w2c=w2c,
            ks=ks,
            camera_model=camera_model
        )

        ### wrapper config-relatived data
        config = ConfigData(
            sh_degree=active_sh_degree,
            tile_size=tile_size,
            render_mode=render_mode,
            scale_log=True,
            opacity_logit=True,
            channel_first=splats["is_channel_first"],
            bg_color=_resolve_bg_color(splats)
        )

        ### invoke enhance api
        render_colors, render_depth, render_alphas, info =\
            self.rasterize(gs, camera, config)
        return render_colors, render_depth, info

    def _rasterization(
        self,
        gs: GaussianData,
        camera: CameraData,
        config: ConfigData,
        rast_ctx: Optional[RastCtx] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Dict]:
        means, quats, scales, opacities = gs.means, gs.quats, gs.scales, gs.opacities
        scales = torch.exp(scales) if config.scale_log else scales
        opacities = torch.sigmoid(opacities) if config.opacity_logit else opacities
        colors = torch.cat([gs.features_dc, gs.features_sh], dim=0).contiguous()
        width, height, viewmats, ks, camera_model = (camera.width, camera.height, camera.w2c,
            camera.ks, camera.camera_model)
        sh_degree, tile_size, render_mode, bg_color = (config.sh_degree, config.tile_size,
            config.render_mode, config.bg_color)

        num_gs = means.shape[-1]
        C = viewmats.shape[0]
        B = 1
        valid_gs = (means, quats, scales, opacities, colors)
        valid_cam = (viewmats, ks, camera_model, render_mode)
        _validate_inputs(valid_gs, valid_cam, sh_degree, num_gs, C)

        if C > 1:
            rendered = []
            depths_per_view = []
            alphas_per_view = []
            infos = []
            for camera_index in range(C):
                if colors.dim() == 4 or (sh_degree is None and colors.dim() == 3):
                    view_colors = colors[camera_index]
                else:
                    view_colors = colors
                # Each view needs its projection buffers until its backward pass.
                gs_per_view = gs
                gs_per_view.features_dc = view_colors[0, :, :].contiguous()
                gs_per_view.features_sh = view_colors[1:, :, :].contiguous()

                cam_per_view = cam
                cam_per_view.w2c = viewmats[camera_index:camera_index + 1]
                cam_per_view.ks = ks[camera_index:camera_index + 1]

                image, depth, alpha, info = self._rasterization(
                    gs_per_view, cam_per_view, config, RastCtx()
                )
                rendered.append(image)
                depths_per_view.append(depth)
                alphas_per_view.append(alpha)
                infos.append(info)
            meta = infos[0].copy()
            for key in ("means2d", "radii", "proj_filter"):
                meta[key] = torch.cat([info[key] for info in infos], dim=1)
            meta["gaussian_ids"] = None
            meta["n_cameras"] = C
            return (torch.cat(rendered, dim=0), torch.cat(depths_per_view, dim=0),
                torch.cat(alphas_per_view, dim=0), meta)

        camtoworlds = torch.inverse(viewmats)
        if sh_degree is None:
            # Post-activation colors: [3, N] or [C, 3, N].
            colors = colors.unsqueeze(0) if colors.dim() == 2 else colors
        else:
            shs = colors.unsqueeze(0) if colors.dim() == 3 else colors
            rays_d = sh_preprocess(camtoworlds, means)   # [B, 3, N] unit dirs
            k = (sh_degree + 1) ** 2
            colors = spherical_harmonics(
                sh_degree,
                rays_d.reshape(B, 3, num_gs),
                shs[0, :k, :, :].reshape(B, k, 3, num_gs),
            )
            colors = (colors + 0.5).clip(min=0.0)

        rast_ctx = self.rast_ctx if rast_ctx is None else rast_ctx
        rast_ctx.prealloc_if_need(B, C, num_gs, means.device)
        vi = torch.ones(B, C, num_gs, dtype=torch.float32, device=means.device)
        sigma = torch.tensor([1.0], dtype=torch.float32, device=means.device)
        means2d, depths, conics, opacities, radius, covars2d, colors, cnt, proj_filter, vi_culling = projection_three_dims_gaussian_fused(
            means.reshape(B, 3, num_gs),
            colors,
            None,
            quats.reshape(B, 4, num_gs),
            scales.reshape(B, 3, num_gs),
            opacities.reshape(B, num_gs),
            viewmats.reshape(B, C, 4, 4).contiguous(),
            ks.reshape(B, C, 3, 3),
            vi, sigma,
            rast_ctx,
            width,
            height
        )

        # Baseline (migration source) fed inverse depth (1/z) to the render stage,
        # while this projection op already returns camera-space z. Restore the
        # baseline rendered-depth metric by inverting here. The tile sort key
        # below keeps the direct depth z so the ascending gaussian_sort still
        # yields the same nearest-first compositing order. Clamp guards culled
        # (depth==0) entries against inf; they are never rendered.
        inv_depths = 1.0 / depths.clamp(min=1e-6)

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
            max_tile_gauss = torch.amax(tile_sums).item()
            lb_sched_tensor = get_render_schedule_cpp(tile_sums_cpu, vector_num).npu()

            sorted_gs_ids = gaussian_sort(
                lb_sched_tensor,
                tile_sums_3d.to(torch.int32),
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
                cf_gs, pix_coords = self.get_render_input(gs_tuple, inv_depths, _cam_view)
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
                    bg_color,
                )
                cf_render_colors = self.tile2image(
                    cf_render_colors.permute(1, 2, 0), height, width
                )
                # simt backend returns depth as [T, P]
                if cf_render_depths.dim() == 2:
                    cf_render_depths = cf_render_depths.unsqueeze(-1)
                cf_render_depths = self.tile2image(
                    cf_render_depths.permute(1, 2, 0), height, width
                )

                render_colors.append(cf_render_colors.permute(1, 2, 0))
                render_depths.append(cf_render_depths.permute(1, 2, 0))
        render_colors = torch.stack(render_colors)
        render_depths = torch.stack(render_depths)

        gaussian_ids = None
        if C == 1 and rast_ctx.features.use_arch35:
            compact_count = int(cnt[0, 0].item())
            gaussian_ids = rast_ctx.filter_outputs["dense_indices"][0, 0, :compact_count].long()

        meta = {
            "gaussian_ids": gaussian_ids,
            "means2d": means2d,
            "radii": radius,
            "width": width,
            "height": height,
            "n_cameras": C,
            "proj_filter": proj_filter,
            "n_gaussians": num_gs,
        }
        return render_colors, render_depths, None, meta
