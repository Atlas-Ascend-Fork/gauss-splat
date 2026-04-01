// Copyright (c) 2024-2025 Huawei Technologies Co., Ltd
// All rights reserved.
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
