# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/rendering.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import math

from typing import Dict, Optional, Tuple
from typing_extensions import Literal
import torch
from torch import Tensor

import acl
from rasterization.utils import validate_inputs
from rasterization.config import Config
from rasterization.host_preprocess import get_cov2d_scales
from meta_gauss_render import (
    spherical_harmonics,
    projection_three_dims_gaussian_fused,
    cube_based_splatting,
    get_batch_render_schedule_cpp
)


class Rasterizer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tile_size = 16
        self.pix_coord = None
        self.padded_width = None
        self.padded_height = None
        self.tile_grid = None
        self.tilenum_height = None
        self.tilenum_width = None
        self.means2d = None
        self.depths = None
        self.conics = None
        self.compensations = None
        self.det = None
        self.radius = None
        self.covars2d = None
        self.combined_culling = None
        self.proj_filter = None
        self.cnt_float = None
        self.dense_indices = None
        self.proj_filter_packed = None
        self.v_pw = None
        self.v_quats = None
        self.v_scales = None
        self.v_colors = None
        self.v_opacities = None
        self.dL_dvi = None
        self.dL_dsigma = None
        self.v_spherical_colors = None
        self.lb_sched = None
        self.lb_sched_bwd = None
        self.last_N = 0
        self.vi_sh_output = None
        self.opacity_one = None
        self.means_culling = None
        self.colors_culling = None
        self.quats_culling = None
        self.scales_culling = None
        self.means2d_culling = None
        self.depths_culling = None
        self.radius_culling = None
        self.covars2d_culling = None
        self.conics_culling = None
        self.opacities_culling = None
        self.vi_culling = None
        self.compensations_culling = None
        self.filter_float_out = None
        self.cnt_float_out = None
        self.dense_indices_out = None
        self.proj_filter_packed_out = None
        self.cbs_gather_result = None
        self.cbs_tiles_per_gauss = None
        self.cbs_gather_valid_gauss = None
        self.cbs_tiles_per_gauss_cumsum = None
        self.cbs_tile_gs_num = None
        self.cbs_tile_gs_start_pos = None
        self.cbs_sort_offset = None
        self.cbs_d_gs = None

    def tile2image(self, rendered_image, B, C, height, width, channel_dim=3):

        rendered_image = rendered_image.reshape(B, C, math.ceil(self.padded_height/self.tile_size),
                                                math.ceil(self.padded_width/self.tile_size), -1, self.tile_size, self.tile_size)
        rendered_image = rendered_image.permute(0,1,2,5,3,6,4)
        rendered_image = rendered_image.reshape(B, C, math.ceil(self.padded_height/self.tile_size)*self.tile_size,
                                                math.ceil(self.padded_width/self.tile_size)*self.tile_size, -1)
        return rendered_image[:, :, :height,:width, :]

    def ascend_rasterize_splats(
        self,
        cam: Tuple,
        size: Tuple,
        tile_size: int,
        active_sh_degree: int,
        splats: dict,
    ) -> Tuple[Tensor, Tensor, Dict]:
        width, height = size
        if self.tile_grid is None:
            self.tile_size = tile_size
            self.tilenum_height = math.ceil(height / tile_size)
            self.tilenum_width = math.ceil(width / tile_size)
            self.padded_height = self.tilenum_height * tile_size
            self.padded_width = self.tilenum_width * tile_size
            device = splats["means"].device
            self.tile_grid = torch.stack(torch.meshgrid(torch.arange(0, self.padded_height, tile_size), \
                torch.arange(0, self.padded_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).to(device)
            self.pix_coord = torch.stack(torch.meshgrid(torch.arange(self.padded_width),
                torch.arange(self.padded_height), indexing='xy'), dim=-1).to(device)

        means = splats["means"]  # [N, 3]
        quats = splats["quats"]  # [N, 4]
        scales = torch.exp(splats["scales"])  # [N, 3]


        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

        camtoworlds, Ks, render_mode = cam
        viewmats = torch.inverse(camtoworlds)

        N = means.shape[0]

        C = viewmats.shape[0]
        B = 1

        if self.proj_filter is None or self.last_N != N:
            self.proj_filter = torch.empty((B, C, N), device=means.device, dtype=torch.float32)

        if self.cnt_float is None or self.last_N != N:
            self.cnt_float = torch.empty((B, C), device=means.device, dtype=torch.float32)

        if self.dense_indices is None or self.last_N != N:
            self.dense_indices = torch.empty((B, C, N), device=means.device, dtype=torch.int32)

        if self.means2d is None or self.last_N != N:
            self.means2d = torch.empty((B, C, 2, N), device=means.device, dtype=torch.float32)

        if self.depths is None or self.last_N != N:
            self.depths = torch.empty((B, C, 1, N), device=means.device, dtype=torch.float32)

        if self.conics is None or self.last_N != N:
            self.conics = torch.empty((B, C, 3, N), device=means.device, dtype=torch.float32)

        if self.compensations is None or self.last_N != N:
            self.compensations  = torch.empty((B, C, 1, N), device=means.device, dtype=torch.float32)

        if self.det is None or self.last_N != N:
            self.det  = torch.empty((B, C, 1, N), device=means.device, dtype=torch.float32)

        if self.radius is None or self.last_N != N:
            self.radius = torch.empty((B, C, 2, N), device=means.device, dtype=torch.float32)

        if self.covars2d is None or self.last_N != N:
            self.covars2d  = torch.empty((B, C, 3, N), device=means.device, dtype=torch.float32)

        if self.proj_filter_packed is None or self.last_N != N:
            proj_filter_packed_dim = (N + 7) // 8
            self.proj_filter_packed = torch.empty((B, C, proj_filter_packed_dim), device=means.device, dtype=torch.uint8)

        if self.v_pw is None or self.last_N != N:
            self.v_pw = torch.zeros((B, 3, N), device=means.device, dtype=torch.float32)

        if self.v_quats is None or self.last_N != N:
            self.v_quats = torch.zeros((B, 4, N), device=means.device, dtype=torch.float32)

        if self.v_scales is None or self.last_N != N:
            self.v_scales = torch.zeros((B, 3, N), device=means.device, dtype=torch.float32)

        # K = colors.shape[1]
        if self.v_colors is None or self.last_N != N:
            self.v_colors = torch.zeros((B, 3, N), device=means.device, dtype=torch.float32)

        if self.v_opacities is None or self.last_N != N:
            self.v_opacities = torch.zeros((B, N), device=means.device, dtype=torch.float32)

        if self.dL_dvi is None or self.last_N != N:
            self.dL_dvi = torch.zeros((B * C, N), device=means.device, dtype=torch.float32)

        if self.dL_dsigma is None or self.last_N != N:
            self.dL_dsigma = torch.zeros((B * C, N), device=means.device, dtype=torch.float32)

        if self.v_spherical_colors is None or self.last_N != N:
            self.v_spherical_colors = torch.zeros((B, 3, N), device=means.device, dtype=torch.float32)
            self.v_spherical_camera = torch.zeros((B, 3, N), device=means.device, dtype=torch.float32)

        if self.vi_sh_output is None or self.last_N != N:
            self.vi_sh_output = torch.ones(B,1,N, dtype=torch.float32, device=means.device)

        block_dim = acl.get_device_capability(0,1)[0]
        if self.lb_sched is None or self.lb_sched_bwd is None:
            tile_gs_num = torch.ones((B, C, self.tile_grid.shape[0]), device=means.device, dtype=torch.int32)
            self.lb_sched = get_batch_render_schedule_cpp(tile_gs_num, block_dim)
            self.lb_sched_bwd = get_batch_render_schedule_cpp(tile_gs_num, block_dim//2)

        proj_filter_packed_dim = (N + 7) // 8

        if self.means_culling is None or self.last_N != N:
            self.means_culling = torch.empty((B, C, 3, N), device=means.device, dtype=torch.float32)
        if self.colors_culling is None or self.last_N != N:
            self.colors_culling = torch.empty((B, C, 7, N), device=means.device, dtype=torch.float32)
        if self.quats_culling is None or self.last_N != N:
            self.quats_culling = torch.empty((B, C, 4, N), device=means.device, dtype=torch.float32)
        if self.scales_culling is None or self.last_N != N:
            self.scales_culling = torch.empty((B, C, 3, N), device=means.device, dtype=torch.float32)
        if self.means2d_culling is None or self.last_N != N:
            self.means2d_culling = torch.empty((B, C, 2, N), device=means.device, dtype=torch.float32)
        if self.depths_culling is None or self.last_N != N:
            self.depths_culling = torch.empty((B, C, N), device=means.device, dtype=torch.float32)
        if self.radius_culling is None or self.last_N != N:
            self.radius_culling = torch.empty((B, C, 2, N), device=means.device, dtype=torch.float32)
        if self.covars2d_culling is None or self.last_N != N:
            self.covars2d_culling = torch.empty((B, C, 3, N), device=means.device, dtype=torch.float32)
        if self.conics_culling is None or self.last_N != N:
            self.conics_culling = torch.empty((B, C, 3, N), device=means.device, dtype=torch.float32)
        if self.opacities_culling is None or self.last_N != N:
            self.opacities_culling = torch.empty((B, C, N), device=means.device, dtype=torch.float32)
        if self.vi_culling is None or self.last_N != N:
            self.vi_culling = torch.empty((B, C, N), device=means.device, dtype=torch.float32)
        if self.compensations_culling is None or self.last_N != N:
            self.compensations_culling = torch.empty((B, C, N), device=means.device, dtype=torch.float32)
        if self.filter_float_out is None or self.last_N != N:
            self.filter_float_out = torch.zeros((B, C, N), device=means.device, dtype=torch.float32)
        if self.cnt_float_out is None or self.last_N != N:
            self.cnt_float_out = torch.zeros((B, C), device=means.device, dtype=torch.float32)
        if self.dense_indices_out is None or self.last_N != N:
            self.dense_indices_out = torch.empty((B, C, N), device=means.device, dtype=torch.int32)
        if self.proj_filter_packed_out is None or self.last_N != N:
            self.proj_filter_packed_out = torch.empty((B, C, proj_filter_packed_dim), device=means.device, dtype=torch.uint8)

        num_tile = self.tile_grid.shape[0]
        if self.cbs_gather_result is None or self.last_N != N:
            self.cbs_gather_result = torch.empty((B, C, 6, N), device=means.device, dtype=torch.float32)
        if self.cbs_tiles_per_gauss is None or self.last_N != N:
            self.cbs_tiles_per_gauss = torch.empty((B, C, N), device=means.device, dtype=torch.float32)
        if self.cbs_gather_valid_gauss is None or self.last_N != N:
            self.cbs_gather_valid_gauss = torch.empty((B, C, block_dim), device=means.device, dtype=torch.int32)
        if self.cbs_tiles_per_gauss_cumsum is None or self.last_N != N:
            self.cbs_tiles_per_gauss_cumsum = torch.empty((B, C, N), device=means.device, dtype=torch.int32)
        if self.cbs_tile_gs_num is None or self.cbs_tile_gs_num.shape != (B, C, num_tile):
            self.cbs_tile_gs_num = torch.empty((B, C, num_tile), device=means.device, dtype=torch.int32)
        if self.cbs_tile_gs_start_pos is None or self.cbs_tile_gs_start_pos.shape != (B, C, num_tile):
            self.cbs_tile_gs_start_pos = torch.empty((B, C, num_tile), device=means.device, dtype=torch.int32)
        if self.cbs_sort_offset is None:
            self.cbs_sort_offset = torch.empty((B, C), device=means.device, dtype=torch.int32)

        # image_ids = kwargs.pop("image_ids", None)
        # CBS参数条件获取
        lc_wsr_enable = self.cfg.lc_wsr_enable
        learnable_background = self.cfg.learnable_background
        if_depth_correct = self.cfg.if_depth_correct
        use_weight_loss = self.cfg.use_weight_loss

        sigma = splats.get("sigma", torch.tensor([1.0], dtype=torch.float32, device=quats.device))
        weight_background = splats.get("weight_background", None)
        # CBS: 视角相关SH（Vi）参数
        if lc_wsr_enable and "features_do" in splats and "features_ro" in splats:
            sh_opacity = torch.cat((splats["features_do"], splats["features_ro"]), dim=1)
        else:
            sh_opacity = None

        if lc_wsr_enable is False:
            opacities = torch.sigmoid(splats["opacities"])  # [N,]
        else:
            if self.opacity_one is None or self.last_N != N:
                self.opacity_one = torch.ones(N, dtype=torch.float32, device=means.device, requires_grad=False)
                opacities = self.opacity_one
            else:
                opacities = self.opacity_one

        if self.last_N != N:
            self.last_N = N

        rasterize_mode = "classic"

        render_colors, render_depth, info = self._ascend_rasterization(
            gs=(means, quats, scales, opacities, colors),
            camera=(viewmats, Ks, self.cfg.camera_model, render_mode),
            size=size,
            sh_degree=active_sh_degree,
            tile_size=tile_size,
            camtoworlds=camtoworlds,
            sigma = sigma,
            weight_background=weight_background,
            camera_sh=sh_opacity,
            rasterize_mode=rasterize_mode,
            lc_wsr_enable=lc_wsr_enable,
            learnable_background=learnable_background,
            if_depth_correct=if_depth_correct,
            use_weight_loss=use_weight_loss,
        )
        return render_colors, render_depth, info


    def save_image(self, tensor, path):
        from torchvision import transforms
        """
        将GPU上的形状为 [3, H, W] 且值在 0~1 之间的图像张量保存为 PNG 文件。

        参数:
            tensor (torch.Tensor): 形状为 [3, H, W] 的张量，值在 0~1 之间。
            path (str): 保存图像的文件路径。
        """
        if len(tensor.shape) == 2:
            h, w = tensor.shape
            tensor = tensor.reshape(1,h,w).repeat(3,1,1)

        # 将张量缩放到 0~255 范围
        tensor = (tensor * 255).byte()
        # 确保张量在 CPU 上
        #tensor = tensor.cpu()

        if tensor.shape[0] == 4:
            # 使用Image.fromarray并指定模式为RGBA
            image = transforms.ToPILImage(mode="RGBA")(tensor)
        else:
            # 将张量转换为 PIL 图像
            image = transforms.ToPILImage()(tensor)

        # 保存图像
        if str(path)[-4:] == ".jpg":
            image.save(path, 'JPEG', quality=95)
        else:
            image.save(path)


    def _ascend_rasterization(
        self,
        gs,
        camera,
        size: Tuple,
        sh_degree: Optional[int] = None,
        tile_size: int = 16,
        camtoworlds: Optional[Tensor] = None,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
        lc_wsr_enable: bool = False,
        learnable_background: bool = False,
        if_depth_correct: bool = False,
        use_weight_loss: bool = False,
        sigma=None,
        weight_background=None,
        camera_sh=None,
        compute_scales_2d=False
    ) -> Tuple[Tensor, Tensor, Dict]:
        means, quats, scales, opacities, colors = gs
        viewmats, Ks, camera_model, render_mode = camera

        N = means.shape[0]
        C = viewmats.shape[0]
        B = 1
        validate_inputs(gs, camera, sh_degree, N, C)
        width, height = size

        # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        if camtoworlds is None:
            camtoworlds = torch.inverse(viewmats) # [C, 4, 4]
        if colors.dim() == 3:
            # Turn [N, K, 3] into [C, N, K, 3]
            shs = colors.expand(C, -1, -1, -1)
        else:
            # colors is already [C, N, K, 3]
            shs = colors

        # build colors
        rays_o = camtoworlds[0, :3, 3]
        rays_d = means - rays_o
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_d = rays_d.permute(1, 0).unsqueeze(0).contiguous()  # [N, 3] -> [3, N] -> [1, 3, N]
        k = (sh_degree + 1) ** 2
        sh_coeffs = shs[0, :, :k, :].permute(1, 2, 0).unsqueeze(0).contiguous()  # [N, k, 3] -> [k, 3, N] -> [1, k, 3, N]
        colors = spherical_harmonics(sh_degree, rays_d, sh_coeffs,
            self.v_spherical_colors)
        colors = (colors + 0.5).clip(min=0.0)

        # CBS: compute vi after projection, multiply onto opacity

        if lc_wsr_enable and camera_sh is not None and sigma is not None:
            k = (sh_degree+1)**2
            camera_sh_output = spherical_harmonics(sh_degree, rays_d, camera_sh[:,:k,:].permute(1, 2, 0).unsqueeze(0).contiguous(), self.v_spherical_camera)
            self.vi_sh_output = camera_sh_output[:, :1, :].contiguous()

        # ascend gauss projection
        # print(f"************ projection_three_dims_gaussian_fused ********************")
        means2d, depths, conics, opacities, radius, covars2d, colors, cnt, proj_filter = projection_three_dims_gaussian_fused(
            means.reshape(B,N,3),   # [N,3] --> [B,N,3]
            colors,
            None,
            quats.reshape(B,N,4),   # [N,4] --> [B,N,4]
            scales.reshape(B,N,3),  # [N,3] --> [B,N,3]
            opacities.reshape(B,N),
            viewmats.reshape(B,C,4,4).contiguous(),
            Ks.reshape(B,C,3,3),
            self.vi_sh_output, # vi
            sigma, # sigma
            width,
            height,
            0.3,
            0.01, # 0.2
            1e10, #
            False,
            if_depth_correct, # if_depth_correct
            "pinhole",
            self.means2d,
            self.depths,
            self.conics,
            self.compensations,
            self.det,
            self.radius,
            self.covars2d,
            self.proj_filter,
            self.cnt_float,
            self.dense_indices,
            self.proj_filter_packed,
            self.v_pw,
            self.v_quats,
            self.v_scales,
            self.v_colors,
            self.v_opacities,
            self.dL_dvi,
            self.dL_dsigma,
            self.means_culling,
            self.colors_culling,
            self.quats_culling,
            self.scales_culling,
            self.means2d_culling,
            self.depths_culling,
            self.radius_culling,
            self.covars2d_culling,
            self.conics_culling,
            self.opacities_culling,
            self.vi_culling,
            self.compensations_culling,
            self.filter_float_out,
            self.cnt_float_out,
            self.dense_indices_out,
            self.proj_filter_packed_out
        )
        depths = 1 / depths
        render_channel_num = colors.shape[2] + 1
        if self.cbs_d_gs is None or self.cbs_d_gs.shape != (B, C, 6 + render_channel_num, N):
            self.cbs_d_gs = torch.empty((B, C, 6 + render_channel_num, N), device=means.device, dtype=torch.float32)
        # print(f"************ cube_based_splatting ********************")
        # breakpoint()
        render_result, weight = cube_based_splatting(means2d, opacities[None,:], conics, covars2d,
                                        cnt[None,:], self.tile_grid.float(), colors, depths,
                                        width, height, tile_size, use_weight_loss, self.lb_sched, self.lb_sched_bwd, False, True, True,
                                        self.cbs_gather_result,
                                        self.cbs_tiles_per_gauss,
                                        self.cbs_gather_valid_gauss,
                                        self.cbs_tiles_per_gauss_cumsum,
                                        self.cbs_tile_gs_num,
                                        self.cbs_tile_gs_start_pos,
                                        self.cbs_sort_offset,
                                        self.cbs_d_gs)
        trans_render_result = self.tile2image(render_result, B, C, height, width)[0] # b, c, imageh, imagew, 4
        trans_weight = self.tile2image(weight, B, C, height, width)[0] # b, c, imageh, imagew, 4
        render_colors = trans_render_result[..., :3]
        render_depths = trans_render_result[..., 3:4]

        if compute_scales_2d:
            scales_2d = None
            # get_cov2d_scales(
            #     means3D, scales, rotations, viewpoint_camera.FoVx, viewpoint_camera.FoVy, viewpoint_camera.world_view_transform, viewpoint_camera.image_width, viewpoint_camera.image_height)
        else:
            scales_2d = None
        # torch.set_printoptions(threshold=float('inf'))
        camera_ids = None
        gaussian_ids = self.dense_indices_out[0, 0, :cnt]
        meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "means2d": means2d,
        "radii": radius,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "tile_width": self.padded_width // tile_size,
        "tile_height": self.padded_height // tile_size,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        # raw channel-first [B,C,2,N]; runner.py adapts before strategy call
        "proj_filter": self.proj_filter_packed_out,
        "n_gaussians": means2d.shape[-1],
        "weight": trans_weight,
        "scales_2d": scales_2d
        }
        return render_colors, render_depths, meta
