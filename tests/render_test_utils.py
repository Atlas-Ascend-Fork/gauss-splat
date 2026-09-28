# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import inspect
import math
from collections import namedtuple

import torch


def cpu_inputs(inputs):
    values = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().clone().requires_grad_(value.requires_grad)
        values.append(value)
    return type(inputs)(*values)


def make_covariance(n_gs):
    matrix = torch.rand(n_gs, 2, 2)
    return matrix @ matrix.transpose(-1, -2) + 0.3 * torch.eye(2)


def transmittance_mask(transparency):
    mask = torch.ones_like(transparency, dtype=torch.bool)
    # The kernel checks only before complete groups of four, never in the tail.
    for start in range(4, transparency.shape[0] - 3, 4):
        if torch.all(transparency[start] < 0.01):
            mask[start:] = False
            break
    return mask


def _report_mismatch(testcase, context, ref_cpu, out_cpu, error):
    ref64, out64 = ref_cpu.double(), out_cpu.double()
    diff = (out64 - ref64).abs()
    relative = torch.where(ref64 != 0, diff / ref64.abs(),
                           torch.where(diff == 0, 0.0, float('inf')))
    flat_index = int(diff.reshape(-1).argmax())
    index = []
    remainder = flat_index
    for size in reversed(ref_cpu.shape):
        index.append(remainder % size)
        remainder //= size
    index = tuple(reversed(index))
    testcase.fail(
        f"{context}; comparator=assertRtolEqual{inspect.signature(testcase.assertRtolEqual)}; "
        f"max_abs_error={diff.max().item():.9g}, mean_abs_error={diff.mean().item():.9g}, "
        f"max_rel_error={relative.max().item():.9g}, "
        f"differing_elements={int((diff != 0).sum())}/{diff.numel()}; "
        f"worst_abs_index={index}, "
        f"expected={ref64.reshape(-1)[flat_index].item():.9g}, "
        f"actual={out64.reshape(-1)[flat_index].item():.9g}; original_error={error}")


def assert_render_results(testcase, expected, actual, case, fields=None):
    check_fields = fields if fields is not None else expected._fields
    for name in check_fields:
        with testcase.subTest(case=case, output=name, seed=1234):
            ref, out = getattr(expected, name), getattr(actual, name)
            context = (f"{name}: case={case}, seed=1234, "
                       f"expected(shape={tuple(ref.shape)}, dtype={ref.dtype}, device={ref.device}), "
                       f"actual(shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device})")
            testcase.assertEqual(out.shape, ref.shape, context)
            ref_cpu, out_cpu = ref.detach().cpu(), out.detach().cpu()
            finite_ref, finite_out = torch.isfinite(ref_cpu), torch.isfinite(out_cpu)
            testcase.assertTrue(bool(finite_ref.all() and finite_out.all()),
                                f"{context}; nonfinite_expected={int((~finite_ref).sum())}, "
                                f"nonfinite_actual={int((~finite_out).sum())}")
            if ref_cpu.numel() == 0:
                continue
            try:
                # Preserve the installed torch_npu comparator and its default tolerances.
                testcase.assertRtolEqual(ref_cpu.numpy(), out_cpu.numpy())
            except AssertionError as error:
                _report_mismatch(testcase, context, ref_cpu, out_cpu, error)


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


def _tile_pixel_coords(image_width, image_height, tile_origin, tile_size, device):
    h, w = tile_origin
    with torch.no_grad():
        pix_coord = torch.stack(torch.meshgrid(
            torch.arange(image_width), torch.arange(image_height), indexing='xy'), dim=-1).to(device)
        return pix_coord[h:h + tile_size, w:w + tile_size, :].reshape(-1, 2)


def _sparsify_alpha(alpha, tile_size):
    reshaped_alpha = alpha.squeeze(dim=-1).reshape(alpha.shape[0], tile_size * tile_size // 64, 64)
    max_values, _ = torch.max(reshaped_alpha, dim=2, keepdim=False)
    keep_mask = (max_values >= 0.01)
    full_mask = keep_mask.unsqueeze(dim=-1).expand_as(reshaped_alpha).float()
    sparsified_alpha = (reshaped_alpha * full_mask).reshape(alpha.shape[0], -1)
    return sparsified_alpha.unsqueeze(dim=-1)


def _composite_tile(weight, local_color, local_depth, tile_box, tile_size):
    h, h_max, w, w_max = tile_box
    color = (weight * local_color[:, None, :]).sum(dim=0).reshape(h_max - h, w_max - w, -1)
    color = torch.nn.functional.pad(color, (0, tile_size - (w_max - w), 0, tile_size - (h_max - h))).reshape(-1, 3)
    depth = (weight * local_depth[:, None, :]).sum(dim=0).reshape(h_max - h, w_max - w, -1)
    depth = torch.nn.functional.pad(depth,
                            (0, 0, 0, tile_size - (w_max - w), 0, tile_size - (h_max - h))).reshape(-1, 1)
    return color, depth


def render_sort_kv(img_size, gs_attr, tile_offsets, tile_size):
    if tile_size != 32:
        raise ValueError('The rendering test reference currently supports tile_size=32 only')
    image_width, image_height = img_size
    device = gs_attr[0].device
    render_color = []
    render_depth = []
    tile_grid = torch.stack(torch.meshgrid(torch.arange(0, image_height, tile_size),
                            torch.arange(0, image_width, tile_size), indexing='ij'), dim=-1).view(-1, 2).tolist()
    for i, (h, w) in enumerate(tile_grid):
        prev_offset = tile_offsets[i - 1] if i > 0 else 0
        cur_offset = tile_offsets[i]
        h_max, w_max = h + tile_size, w + tile_size
        tile_box = (h, h_max, w, w_max)
        tile_coord = _tile_pixel_coords(image_width, image_height, (h, w), tile_size, device)
        local_attrs = tuple(attr[prev_offset:cur_offset] for attr in gs_attr)

        if cur_offset == prev_offset:
            # Keep empty outputs connected to every differentiable input.
            zero = sum(attr[prev_offset:cur_offset].sum() for attr in gs_attr)
            render_color.append(zero.expand(tile_size * tile_size, 3))
            render_depth.append(zero.expand(tile_size * tile_size, 1))
            continue

        (local_means_x, local_means_y, local_conic_0, local_conic_1, local_conic_2,
         local_color_r, local_color_g, local_color_b, local_opacity, local_depth) = local_attrs
        local_depth = local_depth.reshape(-1, 1)

        dx_x = tile_coord[None, :, 0] - local_means_x[:, None]
        dx_y = tile_coord[None, :, 1] - local_means_y[:, None]
        dx_2_x = torch.square(dx_x)
        dx_2_y = torch.square(dx_y)

        gauss_weight = compute_gauss_weight((dx_x, dx_y, dx_2_x, dx_2_y),
                                            local_conic_0, local_conic_1, local_conic_2)
        alpha = (gauss_weight[..., None] * local_opacity[..., None]).clip(max=0.999)
        alpha = _sparsify_alpha(alpha, tile_size)

        transparency = torch.exp((torch.log(torch.cat(
                        [torch.ones_like(alpha[:1, :]), 1 - alpha[:-1, :]], dim=0))).cumsum(dim=0))
        mask_trans = transmittance_mask(transparency)

        local_color = torch.stack([local_color_r, local_color_g, local_color_b], dim=1)
        weight = transparency * alpha * mask_trans
        color, depth = _composite_tile(weight, local_color, local_depth, tile_box, tile_size)
        render_depth.append(depth)
        render_color.append(color)
    render_color = torch.stack(render_color, dim=0).permute(2, 0, 1)
    render_depth = torch.stack(render_depth, dim=0).permute(2, 0, 1)
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


RenderInputs = namedtuple('RenderInputs', ['means2d', 'colors', 'opacities', 'conics_0', 'conics_1', 'conics_2',
                                           'sorted_gs_ids', 'tile_offsets', 'pix_coords', 'depths', 'padded_width',
                                           'padded_height'])


def gen_render_inputs(shape):
    torch.manual_seed(1234)
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
    cov2d = make_covariance(n_gs).npu()

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
    return RenderInputs(
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
        padded_height)
