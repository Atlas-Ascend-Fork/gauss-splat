/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

namespace {
static const int64_t DIRS_DIM = 3;
static const int64_t COEFFS_DIM = 4;
static const int64_t VCOLORS_DIM = 3;
static const int64_t DIRS_ELEMENT = 3;
static const int64_t DIRS_IDX = 2;
static const int64_t COEFFS_IDX = 3;
static const int64_t VCOLORS_IDX = 1;
static const int64_t COEFFS_ELEMENT = 3;
static const int64_t VCOLORS_ELEMENT = 3;
static const int64_t B_IDX = 0;
static const int64_t N_IDX = 1;
static const int64_t K_IDX = 2;
static const int64_t DIM0_FORCHANGE = 0;
static const int64_t DIM1_FORCHANGE = 1;
static const int64_t DIM2_FORCHANGE = 2;
static const int64_t DIM3_FORCHANGE = 3;

} // namespace
std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(at::Tensor &dirs, at::Tensor &coeffs, at::Tensor &v_colors,
                                                           int degree)
{
    TORCH_CHECK(dirs.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(coeffs.device() == dirs.device(), "Inconsistent device.");
    TORCH_CHECK(v_colors.device() == dirs.device(), "Inconsistent device.");

    TORCH_CHECK(dirs.dim() == DIRS_DIM, "dirs's dim should be 3.");
    TORCH_CHECK(coeffs.dim() == COEFFS_DIM, "coeffs's dim should be 4.");
    TORCH_CHECK(v_colors.dim() == VCOLORS_DIM, "v_colors's dim should be 3.");

    int64_t batchSize = coeffs.sizes()[B_IDX];
    int64_t gaussianNum = coeffs.sizes()[N_IDX];
    int64_t coeffsNum = coeffs.sizes()[K_IDX];

    TORCH_CHECK(coeffsNum == (degree + 1) * (degree + 1), "invalid degree.");
    TORCH_CHECK(dirs.sizes()[DIRS_IDX] == DIRS_ELEMENT, "dirs's lastdim should be 3.");
    TORCH_CHECK(coeffs.sizes()[COEFFS_IDX] == COEFFS_ELEMENT, "coeffs's lastdim should be 3.");
    TORCH_CHECK(v_colors.sizes()[VCOLORS_IDX] == VCOLORS_ELEMENT, "v_colors's last second dim should be 3.");

    dirs = dirs.permute({DIM0_FORCHANGE, DIM2_FORCHANGE, DIM1_FORCHANGE}).contiguous();
    coeffs = coeffs.permute({DIM0_FORCHANGE, DIM2_FORCHANGE, DIM3_FORCHANGE, DIM1_FORCHANGE}).contiguous();

    at::Tensor v_dirs = at::zeros({batchSize, DIRS_DIM, gaussianNum}, dirs.options().dtype(at::kFloat)).contiguous();
    at::Tensor v_coeffs = at::zeros({batchSize, coeffsNum, COEFFS_ELEMENT, gaussianNum},
                                    dirs.options().dtype(at::kFloat)).contiguous();

    EXEC_NPU_CMD(aclnnSphericalHarmonicsBwd, dirs, coeffs, v_colors, degree, v_dirs, v_coeffs);

    v_dirs = v_dirs.permute({DIM0_FORCHANGE, DIM2_FORCHANGE, DIM1_FORCHANGE}).contiguous();
    v_coeffs = v_coeffs.permute({DIM0_FORCHANGE, DIM3_FORCHANGE, DIM1_FORCHANGE, DIM2_FORCHANGE}).contiguous();
    return std::tie(v_dirs, v_coeffs);
}