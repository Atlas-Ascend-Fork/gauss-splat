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

using namespace NPU_NAME_SPACE;
using namespace std;

namespace {
static const int64_t COLORS_DIM = 3;
static const int64_t BATCH_INDEX = 0;
static const int64_t CHANNEL_INDEX = 1;
static const int64_t GAUSS_INDEX = 2;
}

std::tuple<at::Tensor, at::Tensor> color_exposure_bwd(
    at::Tensor& colors, at::Tensor& vColorsIn, at::Tensor& exposure,
    const c10::optional<at::Tensor> &index, const c10::optional<at::Tensor>& weight,
    int32_t clamp, int32_t calcExpGrad)
{
    TORCH_CHECK(torch_npu::utils::is_npu(exposure), "exposure must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(colors), "colors must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(vColorsIn), "vColorsIn must be NPU tensor");
    TORCH_CHECK(colors.dim() == COLORS_DIM, "colors's dim should be 3.");
    TORCH_CHECK(vColorsIn.dim() == COLORS_DIM, "vColorsIn's dim should be 3.");

    auto batchSize = vColorsIn.sizes()[BATCH_INDEX];
    auto channelNum = vColorsIn.sizes()[CHANNEL_INDEX];
    auto gaussNum = vColorsIn.sizes()[GAUSS_INDEX];

    at::Tensor vColorsOut = at::zeros({batchSize, channelNum, gaussNum}, vColorsIn.options().dtype(at::kFloat));
    at::Tensor vExposure;
    if (calcExpGrad) {
        vExposure = at::zeros_like(exposure);
    } else {
        vExposure = at::empty({0}, colors.options().dtype(at::kFloat));
    }

    EXEC_NPU_CMD(aclnnColorExposureBwd, colors, vColorsIn, exposure,
        index, weight, clamp, calcExpGrad, vColorsOut, vExposure);

    return std::tie(vColorsOut, vExposure);
}