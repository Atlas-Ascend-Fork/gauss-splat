/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <torch/extension.h>

#include <mutex>
#include <string>

#include "functions.h"

std::string g_opApiSoPath;
std::once_flag init_flag; // Flag for one-time initialization

void init_op_api_so_path(const std::string& path)
{
    std::call_once(init_flag, [&]() { g_opApiSoPath = path; });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_init_op_api_so_path", &init_op_api_so_path);

    m.def("calc_render_fwd_double_clip_gsids", &calc_render_fwd_double_clip_gsids);

    m.def("calc_render_bwd_var_clip_gsids", &calc_render_bwd_var_clip_gsids);

    m.def("gaussian_sort", &gaussian_sort);

    m.def("get_render_schedule", &get_render_schedule);

    m.def("projection_three_dims_gaussian_forward", &projection_three_dims_gaussian_forward);

    m.def("quat_scales_to_covars", &quat_scales_to_covars);

    m.def("fully_fused_projection_bwd", &fully_fused_projection_bwd);

    m.def("spherical_harmonics_forward", &spherical_harmonics_forward);

    m.def("spherical_harmonics_bwd", &spherical_harmonics_bwd);

    m.def("flash_gaussian_build_mask", &flash_gaussian_build_mask);

    m.def("gaussian_filter", &gaussian_filter);
}
