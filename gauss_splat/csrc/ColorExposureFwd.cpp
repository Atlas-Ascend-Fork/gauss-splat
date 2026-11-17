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
static const int64_t MEANS2D_DIM = 3;
static const int64_t INDEX_CHANNEL_NUM = 3;
static const int64_t WEIGHT_CHANNEL_NUM = 3;
static const int64_t EXPOSURE_NUM = 12;
static const int64_t BATCH_INDEX = 0;
static const int64_t CHANNEL_INDEX = 1;
static const int64_t GAUSS_INDEX = 2;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> color_exposure_fwd(
    at::Tensor& colors, at::Tensor& means2d, at::Tensor& exposure, 
    int32_t width, int32_t height, int32_t clamp)
{
    TORCH_CHECK(torch_npu::utils::is_npu(exposure), "exposure must be NPU tensor");
    TORCH_CHECK(colors.dim() == COLORS_DIM, "colors's dim should be 3.");
    TORCH_CHECK(means2d.dim() == MEANS2D_DIM, "means2d's dim should be 3.");

    auto batchSize = colors.sizes()[BATCH_INDEX];
    auto channelNum = colors.sizes()[CHANNEL_INDEX];
    auto gaussNum = colors.sizes()[GAUSS_INDEX];
    bool useGrid = exposure.numel() > EXPOSURE_NUM;

    at::Tensor output = at::zeros({batchSize, channelNum, gaussNum}, colors.options().dtype(at::kFloat));
    at::Tensor index, weight;
    if (useGrid) {
        // index/weight need output for backward if use grid.
        index = at::zeros({batchSize, INDEX_CHANNEL_NUM, gaussNum}, colors.options().dtype(at::kInt));
        weight = at::zeros({batchSize, WEIGHT_CHANNEL_NUM, gaussNum}, colors.options().dtype(at::kFloat));
    } else {
        index = at::empty({0}, colors.options().dtype(at::kInt));
        weight = at::empty({0}, colors.options().dtype(at::kFloat));
    }

    EXEC_NPU_CMD(aclnnColorExposureFwd, colors, means2d, exposure,
        width, height, clamp, output, index, weight);

    return std::tie(output, index, weight);
}