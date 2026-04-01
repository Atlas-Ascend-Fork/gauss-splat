// Copyright (c) 2024, Huawei Technologies.All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef CSRC_FUNCTIONS_H_
#define CSRC_FUNCTIONS_H_

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> calc_render_fwd_double_clip_gsids(const at::Tensor &gs,
    const at::Tensor &tileCoords, const at::Tensor &offsets, const at::Tensor &gsIds);

at::Tensor calc_render_bwd_var_clip_gsids(
    const at::Tensor &vColor, const at::Tensor &vDepth, const at::Tensor &lastCumsum, const at::Tensor &error,
    const at::Tensor &gs, const at::Tensor &tileCoords,
    const at::Tensor &offsets, const at::Tensor &gsIds, const at::Tensor &gsClipIndex, const at::Tensor &alphaClipIndex);

std::tuple<at::Tensor, at::Tensor> gaussian_sort(const at::Tensor &all_in_mask, const at::Tensor &depths);

at::Tensor get_render_schedule(const at::Tensor &nums_tensor, int num_bins);

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

at::Tensor flash_gaussian_build_mask(at::Tensor& means2d, at::Tensor& opacity, at::Tensor& conics,
                                             at::Tensor& covars2d, at::Tensor& cnt, at::Tensor& tile_grid,
                                             double image_width, double image_height, int32_t tile_size);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor>
gaussian_filter(at::Tensor &means, at::Tensor &colors, at::Tensor &det, at::Tensor &opacities,
                at::Tensor &means2d, at::Tensor &depths, at::Tensor &radius, at::Tensor &conics,
                at::Tensor &covars2d, const c10::optional<at::Tensor> &compensations,
                int width, int height, double near_plane, double far_plane);
#endif // CSRC_FUNCTIONS_H_
