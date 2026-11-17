/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_ADAM_MULTI_LR_TILING_H
#define FUSED_ADAM_MULTI_LR_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedAdamMultiLrTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tensorNum);
    TILING_DATA_FIELD_DEF(uint64_t, ubTotalSize);
    TILING_DATA_FIELD_DEF(float, beta1);
    TILING_DATA_FIELD_DEF(float, beta2);
    TILING_DATA_FIELD_DEF(float, eps);
    TILING_DATA_FIELD_DEF(float, weightDecay);
    TILING_DATA_FIELD_DEF(float, oneMinusBeta1);
    TILING_DATA_FIELD_DEF(float, oneMinusBeta2);
    TILING_DATA_FIELD_DEF(float, invBiasCorrection2);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, paramNums);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, totalTaskNums);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, tailNums);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, taskNumPerScores);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, taskNumPerLcores);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, numScores);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, numLcores);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 64, blockDims);
    TILING_DATA_FIELD_DEF_ARR(int32_t, 64, taskNumPerLoops);
    TILING_DATA_FIELD_DEF_ARR(float, 64, stepSizes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedAdamMultiLr, FusedAdamMultiLrTilingData)
}

#endif