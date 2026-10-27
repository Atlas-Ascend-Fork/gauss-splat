# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
from collections import namedtuple

import acl
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from gauss_splat import calc_render, get_render_schedule_cpp
from gauss_splat import gsplat_c

torch.npu.set_device('npu:0')
torch.manual_seed(1234)

ExecResults = namedtuple('ExecResults', ['render_colors', 'render_depths'])
Inputs = namedtuple('Inputs', ['means2d', 'colors', 'opacities', 'conics_0', 'conics_1', 'conics_2', 'sorted_gs_ids', \
                               'tile_offsets', 'pix_coords', 'depths', 'padded_width', 'padded_height'])


def inverse_cov2d_v2(cov2_00, cov2_01, cov2_11, scale=1):
    det = cov2_00 * cov2_11 - cov2_01 * cov2_01
    inv_x_0 = cov2_11 / det * scale
    inv_x_1 = -cov2_01 / det * scale
    inv_x_2 = cov2_00 / det * scale
    return inv_x_0, inv_x_1, inv_x_2


@torch.no_grad()
def sort_gs(all_in_mask, depths):
    num_tile = all_in_mask.shape[1]
    # tile offset
    tile_offset = torch.sum(all_in_mask, dim=0).cumsum(dim=0)
    
    sorted_gs_ids = torch.zeros(tile_offset[-1], dtype=torch.int32, device=all_in_mask.device)
    for tile_id in range(num_tile):
        prev_offset = tile_offset[tile_id - 1] if tile_id > 0 else 0
        tile_in_mask = all_in_mask[:, tile_id]
        tile_depths = depths[tile_in_mask]
        tile_gs_ids = tile_in_mask.nonzero()[:, 0]
        _, local_sort_index = torch.sort(tile_depths, stable=True)
        sorted_gs_ids[prev_offset:tile_offset[tile_id]] = tile_gs_ids[local_sort_index]
    
    return sorted_gs_ids, tile_offset


@torch.no_grad()
def get_radius_v2(cov2_00, cov2_01, cov2_11):
    det = cov2_00 * cov2_11 - cov2_01 * cov2_01
    mid = 0.5 * (cov2_00 + cov2_11)
    lambda1 = mid + torch.sqrt((mid**2 - det).clip(min=0.1))
    return 3.0 * torch.sqrt(lambda1).ceil()


@torch.no_grad()
def get_rect_v2(means_x, means_y, radii, width, height):
    rect_min_0 = torch.clamp(means_x - radii, 0, width - 1.0)
    rect_min_1 = torch.clamp(means_y - radii, 0, height - 1.0)
    rect_max_0 = torch.clamp(means_x + radii, 0, width - 1.0)
    rect_max_1 = torch.clamp(means_y + radii, 0, height - 1.0)
    return rect_min_0, rect_min_1, rect_max_0, rect_max_1


def build_tile_gs_mask(means_2d, radii, img_size, tile_grid, tile_size):
    means_x, means_y = means_2d[:, 0], means_2d[:, 1]
    image_width, image_height = img_size
    num_gs = means_x.shape[0]
    num_tile = tile_grid.shape[0]
    rmin_w, rmin_h, rmax_w, rmax_h = get_rect_v2(means_x, means_y, radii, image_width, image_height)

    w_right_bound = torch.clamp(rmax_w[:, None].expand(num_gs, num_tile),
                                max=(tile_grid[None, :, 1] + tile_size).expand(num_gs, num_tile))
    w_left_bound = torch.clamp(rmin_w[:, None].expand(num_gs, num_tile),
                                min=(tile_grid[None, :, 1]).expand(num_gs, num_tile))
    h_upper_bound = torch.clamp(rmax_h[:, None].expand(num_gs, num_tile),
                                max=(tile_grid[None, :, 0] + tile_size).expand(num_gs, num_tile))
    h_lower_bound = torch.clamp(rmin_h[:, None].expand(num_gs, num_tile),
                                min=(tile_grid[None, :, 0]).expand(num_gs, num_tile))
    all_in_mask = (w_right_bound > w_left_bound) & (h_upper_bound > h_lower_bound)

    return all_in_mask


def compute_gauss_weight(dxy, local_conic_0, local_conic_1, local_conic_2):
    dx_x, dx_y, dx_2_x, dx_2_y = dxy
    gauss_weight = torch.exp(
        -(0.5 * (dx_2_x * local_conic_0.unsqueeze(dim=1)
        + dx_2_y * local_conic_2.unsqueeze(dim=1))
        + dx_x * dx_y * local_conic_1.unsqueeze(dim=1))
    )
    return gauss_weight


def render_sort_kv(img_size, gs_attr, tile_offsets, tile_size):
    image_width, image_height = img_size
    means_x, means_y, conics_0, conics_1, conics_2, color_r, color_g, color_b, opacity, depths = gs_attr
    render_color = []
    render_depth = []
    tile_grid = torch.stack(torch.meshgrid(torch.arange(0, image_height, tile_size),
                            torch.arange(0, image_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).tolist()
    local_tiles = range(0, len(tile_grid))
    tile_grid_list = tile_grid
    for i in local_tiles:
        prev_offset = tile_offsets[i - 1] if i > 0 else 0
        cur_offset = tile_offsets[i]
        with torch.no_grad():
            h, w = tile_grid_list[i]
            h_max = h + tile_size
            w_max = w + tile_size
            pix_coord = torch.stack(torch.meshgrid(
                torch.arange(image_width), torch.arange(image_height), indexing='xy'), dim=-1).to(means_x.device)
            tile_coord = pix_coord[h:h_max, w:w_max, :].reshape(-1, 2)
        local_means_x = means_x[prev_offset:cur_offset]
        local_means_y = means_y[prev_offset:cur_offset]
        local_conic_0 = conics_0[prev_offset:cur_offset]     
        local_conic_1 = conics_1[prev_offset:cur_offset]     
        local_conic_2 = conics_2[prev_offset:cur_offset]     
        local_opacity = opacity[prev_offset:cur_offset]
        local_color_r = color_r[prev_offset:cur_offset]
        local_color_g = color_g[prev_offset:cur_offset]
        local_color_b = color_b[prev_offset:cur_offset]
        local_depth = depths[prev_offset:cur_offset].reshape(-1, 1)

        dx_x = tile_coord[None, :, 0] - local_means_x[:, None]
        dx_y = tile_coord[None, :, 1] - local_means_y[:, None]
        dx_2_x = torch.square(dx_x)
        dx_2_y = torch.square(dx_y)

        dxy = dx_x, dx_y, dx_2_x, dx_2_y

        gauss_weight = compute_gauss_weight(dxy, local_conic_0, local_conic_1, local_conic_2)
        if getattr(gsplat_c, "sf_render_method", "") == "simt":
            alpha = (gauss_weight * local_opacity).clamp(max=0.999)
            alpha = torch.where(alpha >= 1.0 / 255.0, alpha, 0.0)
            trans_after = torch.exp(torch.cumsum(torch.log1p(-alpha), dim=0))
            trans_before = torch.cat((torch.ones_like(alpha[:1]), trans_after[:-1]), dim=0)
            weights = torch.where(trans_after > 1e-4, trans_before * alpha, 0.0)
            local_color = torch.stack([local_color_r, local_color_g, local_color_b], dim=1)
            render_color.append((weights[..., None] * local_color[:, None, :]).sum(dim=0))
            render_depth.append((weights[..., None] * local_depth[:, None, :]).sum(dim=0))
            continue
        alpha = (gauss_weight[..., None] * local_opacity[..., None]).clip(max=0.999)
        reshaped_alpha = alpha.squeeze(dim=-1).reshape(alpha.shape[0], 16, 64)
        max_values, _ = torch.max(reshaped_alpha, dim=2, keepdim=False)
        keep_mask = (max_values >= 0.01)
        full_mask = keep_mask.unsqueeze(dim=-1).expand_as(reshaped_alpha).float()
        sparsified_alpha = (reshaped_alpha * full_mask).reshape(alpha.shape[0], -1)
        alpha = sparsified_alpha.unsqueeze(dim=-1)

        transparency = torch.exp((torch.log(torch.cat(
                        [torch.ones_like(alpha[:1, :]), 1 - alpha[:-1, :]], dim=0))).cumsum(dim=0))
        mask_trans = (transparency >= 0.01)

        local_color = torch.stack([local_color_r, local_color_g, local_color_b], dim=1)
        if tile_size == 64:
            color = ((transparency * alpha * local_color[:, None, :])).sum(dim=0).reshape(h_max - h, w_max - w, -1)
        else:
            mask_trans_all = torch.all(mask_trans == 0, dim=1)

            ii = 0
            while (ii < mask_trans.shape[0] - 3):
                if mask_trans_all[ii] == 1:
                    mask_trans[ii: ii + 4] = False
                else:
                    mask_trans[ii: ii + 4] = True
                ii = ii + 4
            
            while (ii < mask_trans.shape[0]):
                if mask_trans_all[ii] == 1:
                    mask_trans[ii] = False
                else:
                    mask_trans[ii] = True
                ii += 1

            color = ((transparency * alpha * local_color[:, None, :]) *
                     mask_trans).sum(dim=0).reshape(h_max - h, w_max - w, -1)
        color = torch.nn.functional.pad(color, (0, tile_size - (w_max - w), 0, tile_size - (h_max - h))).reshape(-1, 3)
        if tile_size == 64:
            depth = ((transparency * alpha * local_depth[:, None, :])).sum(dim=0).reshape(h_max - h, w_max - w, -1)
        else:
            depth = ((transparency * alpha * local_depth[:, None, :]) *
                     mask_trans).sum(dim=0).reshape(h_max - h, w_max - w, -1)
        depth = torch.nn.functional.pad(depth,
                                (0, 0, 0, tile_size - (w_max - w), 0, tile_size - (h_max - h))).reshape(-1, 1)
        render_depth.append(depth)
        render_color.append(color)
    render_color = torch.stack(render_color, dim=0).permute(2, 0, 1)
    render_depth = torch.stack(render_depth, dim=0).permute(2, 0, 1)
    if getattr(gsplat_c, "sf_render_method", "") == "simt":
        render_depth = render_depth.squeeze(0)
    return render_color, render_depth


def render_with_gsids(img_size, gs_attr, tile_offsets, sorted_gs_ids, tile_size):

    def clone_attributes(gs_attr, sorted_gs_ids):
        mean_ndc, conics_0, conics_1, conics_2, colors, opacities, depths = gs_attr
        means_x = torch.index_select(mean_ndc[:, 0], 0, sorted_gs_ids)
        means_y = torch.index_select(mean_ndc[:, 1], 0, sorted_gs_ids)
        conics_0 = torch.index_select(conics_0, 0, sorted_gs_ids)
        conics_1 = torch.index_select(conics_1, 0, sorted_gs_ids)
        conics_2 = torch.index_select(conics_2, 0, sorted_gs_ids)
        opacities = torch.index_select(opacities, 0, sorted_gs_ids)
        depths = torch.index_select(depths, 0, sorted_gs_ids)

        color_r = torch.index_select(colors[:, 0], 0, sorted_gs_ids)
        color_g = torch.index_select(colors[:, 1], 0, sorted_gs_ids)
        color_b = torch.index_select(colors[:, 2], 0, sorted_gs_ids)
        render_gs_attr = means_x, means_y, conics_0, conics_1, conics_2, color_r, color_g, color_b, opacities, depths
        return render_gs_attr
    
    render_gs_attr = clone_attributes(
        gs_attr, sorted_gs_ids
    ) 
    render_color, render_depth = render_sort_kv(img_size, render_gs_attr, tile_offsets, tile_size)

    return render_color, render_depth


class TestCalcRenderForwardDoubleClipGsids(TestCase):

    def setUp(self):
        self.test_cases = [
            [102000, 32 * 3, 32 * 4, 32],
        ]
        self.test_results = self.gen_results()

    def gen_inputs(self, shape):
        n_gs, image_width, image_height, tile_size = shape

        padded_width = math.ceil(image_width / tile_size) * tile_size
        padded_height = math.ceil(image_height / tile_size) * tile_size
        tile_grid = torch.stack(torch.meshgrid(torch.arange(0, padded_height, tile_size),
                                            torch.arange(0, padded_width, tile_size),
                                            indexing='ij'), dim=-1).view(-1, 2).npu()
        pix_coord = torch.stack(torch.meshgrid(torch.arange(padded_width),
                                               torch.arange(padded_height), indexing='xy'), dim=-1).npu()
        pix_coords = pix_coord.reshape(padded_height // tile_size, tile_size, padded_width // tile_size, tile_size, 2) \
                                    .permute(0, 2, 1, 3, 4).reshape(padded_height // tile_size *
                                            padded_width // tile_size, tile_size * tile_size, 2) \
                                    .permute(0, 2, 1).to(torch.float32).contiguous()
        
        mean_x = torch.rand((n_gs, 1)).npu() * image_width
        mean_y = torch.rand((n_gs, 1)).npu() * image_height
        means2d = torch.cat([mean_x, mean_y], dim=1)
        colors = torch.rand((n_gs, 3)).npu()
        opacities = torch.rand((n_gs, 1)).npu()
        cov2d = torch.rand((n_gs, 2, 2)).npu()

        cov2d_00, cov2d_01, cov2d_11 = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
        conics_0, conics_1, conics_2 = inverse_cov2d_v2(cov2d_00, cov2d_01, cov2d_11)
        depths = torch.rand((n_gs)).npu() * 20
        radii = get_radius_v2(cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1])
        img_size = (image_width, image_height)
        all_in_mask = build_tile_gs_mask(
            means2d,
            radii, img_size,
            tile_grid=tile_grid, tile_size=tile_size)

        sorted_gs_ids, tile_offsets = sort_gs(all_in_mask.to(torch.bool), depths)
        return Inputs(
                    means2d.clone().requires_grad_(),
                    colors.clone().requires_grad_(),
                    opacities.clone().requires_grad_(),
                    conics_0.clone().requires_grad_(),
                    conics_1.clone().requires_grad_(),
                    conics_2.clone().requires_grad_(),
                    sorted_gs_ids,
                    tile_offsets,
                    pix_coords,
                    depths.clone().requires_grad_(),
                    padded_width,
                    padded_height
                    )

    def gen_results(self):
        test_results = []
        for test_case in self.test_cases:
            tile_size = test_case[-1]
            op_inputs = self.gen_inputs(test_case)
            cpu_results = self.cpu_to_exec(op_inputs, tile_size)
            npu_results = self.npu_to_exec(op_inputs)
            test_results.append((cpu_results, npu_results))
        return test_results
    
    def cpu_to_exec(self, op_inputs, tile_size):
        means2d, colors, opacities, conics_0, conics_1, conics_2, sorted_gs_ids, \
                        tile_offsets, _, depths, padded_width, padded_height = op_inputs
        img_size = (padded_width, padded_height)
        gs_attr = (means2d, conics_0, conics_1, conics_2, colors, opacities, depths)

        render_color_gt, render_depth_gt = render_with_gsids(img_size, gs_attr, tile_offsets, sorted_gs_ids, tile_size)

        return ExecResults(
            render_colors=render_color_gt.detach().float(),
            render_depths=render_depth_gt.detach().float(),
            )
    
    def npu_to_exec(self, op_inputs):
        means2d, colors, opacities, conics_0, conics_1, conics_2, sorted_gs_ids, \
            tile_offsets, pix_coords, depths, _, _ = op_inputs
        nums = torch.cat([tile_offsets[:1], tile_offsets[1:] - tile_offsets[:-1]])
        num_vector_core, ret = acl.get_device_capability(0, 1)
        lb_sched = get_render_schedule_cpp(nums.cpu().to(torch.int64), 
                                           num_vector_core).clone().detach().to(torch.int64).npu()
        cf_means2 = means2d.transpose(0, 1)
        cf_colors3 = colors.transpose(0, 1)
        cf_render_colors, cf_render_depths = calc_render(cf_means2,
                                                        conics_0, conics_1, conics_2,
                                                        opacities.squeeze(dim=-1),
                                                        cf_colors3,
                                                        depths[None, :],
                                                        pix_coords,
                                                        lb_sched,
                                                        sorted_gs_ids
                                                        )
        return ExecResults(
            render_colors=cf_render_colors.detach().float(),
            render_depths=cf_render_depths.detach().float(),
            )
    
    def test_calc_render_forward_double_clip_gsids(self):
        for cpu_results, npu_results in self.test_results:
            self.assertRtolEqual(cpu_results.render_colors.cpu().numpy(), npu_results.render_colors.cpu().numpy())
            self.assertRtolEqual(cpu_results.render_depths.cpu().numpy(), npu_results.render_depths.cpu().numpy())

if __name__ == "__main__":
    run_tests()


