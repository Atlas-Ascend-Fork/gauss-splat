/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CSRC_FUNCTIONS_H_
#define CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> calc_render_fwd_double_clip_gsids(const at::Tensor &gs,
    const at::Tensor &tileCoords, const at::Tensor &offsets, const at::Tensor &gsIds);

at::Tensor calc_render_bwd_var_clip_gsids(
    const at::Tensor &vColor, const at::Tensor &vDepth, const at::Tensor &lastCumsum, const at::Tensor &error,
    const at::Tensor &gs, const at::Tensor &tileCoords,
    const at::Tensor &offsets, const at::Tensor &gsIds, const at::Tensor &gsClipIndex, const at::Tensor &alphaClipIndex);

at::Tensor gaussian_sort(const at::Tensor& lb_sched, const at::Tensor& gaussian_cnt, const at::Tensor& depths,
                         const at::Tensor& gs_ids, const at::Tensor& sorted_offset, int32_t max_tile_gauss);

at::Tensor get_render_schedule(const at::Tensor& input, int num_bins);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_three_dims_gaussian_forward(
    at::Tensor& means, at::Tensor& covars, at::Tensor& opacities, at::Tensor& viewmats, at::Tensor& ks,
    int32_t width, int32_t height, double eps, bool calc_compensations, std::string camera_model);

at::Tensor quat_scales_to_covars(at::Tensor& quat, at::Tensor& scales);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fully_fused_projection_bwd(const at::Tensor &means,
    const at::Tensor &quats, const at::Tensor &scales, const at::Tensor &conics, const at::Tensor &viewmats,
    const at::Tensor &Ks, const at::Tensor &v_means2d, const at::Tensor &v_depths, const at::Tensor &v_conics,
    const at::Tensor &v_colors_culling, const at::Tensor &v_opacities_culling, const at::Tensor &filter,
    const c10::optional<at::Tensor> &compensations, int width, int height);

at::Tensor spherical_harmonics_forward(at::Tensor& dirs, at::Tensor& coeffs, int32_t degrees_to_use);

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(at::Tensor &dirs, at::Tensor &coeffs,
                                                           at::Tensor &v_colors, int degree);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> flash_gaussian_build_mask(
    at::Tensor& means2d, at::Tensor& opacity, at::Tensor& conics, at::Tensor& covars2d, at::Tensor& depths,
    at::Tensor& cnt, at::Tensor& tile_grid, double image_width, double image_height, int32_t tile_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
gaussian_filter(at::Tensor &means, at::Tensor &colors, at::Tensor &det, at::Tensor &opacities,
                at::Tensor &means2d, at::Tensor &depths, at::Tensor &radius, at::Tensor &conics,
                at::Tensor &covars2d, const c10::optional<at::Tensor> &compensations,
                int width, int height, double near_plane, double far_plane);
#endif // CSRC_FUNCTIONS_H_
