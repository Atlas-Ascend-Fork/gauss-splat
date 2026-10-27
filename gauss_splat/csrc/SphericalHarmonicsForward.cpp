/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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

    // Keep the batch dim so tensor shapes match the arch35 host tiling
    // contract ([B,3,N] / [B,K,C,N]) and the kernel GM addressing:
    // dirs [B,N,3] -> [B,3,N], coeffs [B,N,K,3] -> [B,K,3,N].
    dirs = dirs.permute({0, 2, 1}).contiguous();
    coeffs = coeffs.permute({0, 2, 3, 1}).contiguous();

    TORCH_CHECK(dirs.sizes()[1] == SH_TAIL_DIM,
        "The second dimension of Dirs Tensor Must be 3.");
    TORCH_CHECK(coeffs.sizes()[2] == SH_TAIL_DIM,
        "The third dimension of Coeffs Tensor Must be 3.");

    at::Tensor output = at::zeros({batchSize, SH_TAIL_DIM, gaussianNum},
                                  dirs.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnSphericalHarmonicsForward, dirs, coeffs, degrees_to_use, output);

    return output;
}