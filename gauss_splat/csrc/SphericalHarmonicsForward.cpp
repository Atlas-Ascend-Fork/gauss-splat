/**
 * Adapted from
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) 2019, Facebook CORPORATION.
 
 * All rights reserved.
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include "functions.h"
#include "OpApiCommon.h"

namespace {
constexpr uint32_t SH_TAIL_DIM = 3;
}

using namespace NPU_NAME_SPACE;
using namespace std;

at::Tensor spherical_harmonics_forward(at::Tensor& dirs, at::Tensor& coeffs, int32_t degrees_to_use)
{
    TORCH_CHECK(torch_npu::utils::is_npu(dirs), "dirs must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(coeffs), "coeffs must be NPU tensor");
    TORCH_CHECK(dirs.scalar_type() == at::kFloat,
        "dirs: float32 tensor expected but got a tensor with dtype: ", dirs.scalar_type());
    TORCH_CHECK(coeffs.scalar_type() == at::kFloat,
        "coeffs: float32 tensor expected but got a tensor with dtype: ", coeffs.scalar_type());

    uint32_t coeffsDim = coeffs.dim();
    uint32_t coeffsNum = coeffs.sizes()[coeffsDim - 2];
    if ((degrees_to_use + 1) * (degrees_to_use + 1) > coeffsNum) {
        AT_ERROR("Coeffs doesn't provide enough spherical harmonics bases functions to compute spherical harmonics.");
    }

    auto batchSize = dirs.sizes()[0];
    auto gaussianNum = dirs.sizes()[1];

    uint32_t dirsNumel = dirs.numel();
    uint32_t coeffsNumel = coeffs.numel();
    dirsNumel = static_cast<uint32_t>(dirsNumel / SH_TAIL_DIM);
    coeffsNumel = static_cast<uint32_t>(coeffsNumel / SH_TAIL_DIM / coeffsNum);
    dirs = dirs.reshape({dirsNumel, SH_TAIL_DIM}).contiguous().permute({1, 0}).contiguous();
    coeffs = coeffs.reshape({coeffsNumel, coeffsNum, SH_TAIL_DIM}).contiguous().permute({1, 2, 0}).contiguous();

    TORCH_CHECK(dirs.sizes()[0] == SH_TAIL_DIM,
        "The last dimension of Dirs Tensor Must be 3.");
    TORCH_CHECK(coeffs.sizes()[1] == SH_TAIL_DIM,
        "The last dimension of Coeffs Tensor Must be 3.");

    at::Tensor output = at::zeros(dirs.sizes(), dirs.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnSphericalHarmonicsForward, dirs, coeffs, degrees_to_use, output);

    output = output.reshape({SH_TAIL_DIM, batchSize, gaussianNum}).contiguous().permute({1, 0, 2}).contiguous();
    return output;
}