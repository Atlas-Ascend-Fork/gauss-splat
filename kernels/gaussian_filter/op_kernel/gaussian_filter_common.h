/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_FILTER_COMMON_H
#define GAUSSIAN_FILTER_COMMON_H

#include "kernel_operator.h"

namespace GaussianFilterNs {
using namespace AscendC;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;
constexpr int64_t INT64_ONE_BLOCK_NUM = 4;
constexpr int64_t FLOAT_SIZE = 4;
constexpr int64_t INT16_SIZE = 2;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t MASK_BATCH_SIZE = 256;
constexpr int64_t MAX_CORE_NUM = 48;

constexpr int64_t MEANS_DIM = 3;
constexpr int64_t COLORS_DIM = 3;
constexpr int64_t DET_DIM = 1;
constexpr int64_t OPACITIES_DIM = 1;
constexpr int64_t COMPENSATIONS_DIM = 1;
constexpr int64_t MEANS2D_DIM = 2;
constexpr int64_t DEPTHS_DIM = 1;
constexpr int64_t RADIUS_DIM = 2;
constexpr int64_t CONICS_DIM = 3;
constexpr int64_t COVARS2D_DIM = 3;
constexpr int64_t MEANS_CULLING_DIM = 3;
constexpr int64_t COLORS_CULLING_DIM = 3;
constexpr int64_t MEANS2D_CULLING_DIM = 2;
constexpr int64_t DEPTHS_CULLING_DIM = 1;
constexpr int64_t RADIUS_CULLING_DIM = 2;
constexpr int64_t CONICS_CULLING_DIM = 3;
constexpr int64_t COVARS2D_CULLING_DIM = 3;
constexpr int64_t OPACITIES_CULLING_DIM = 1;
constexpr int64_t FILTER_DIM = 1;
constexpr int64_t CNT_DIM = 0;
constexpr int64_t QUE_LEN = 4;
constexpr int64_t CALBUF_LEN1 = 3;
constexpr int64_t CALBUF_LEN2 = 2;
constexpr int64_t CNTPERLOOP = 2;
constexpr int64_t ALIGN_8 = 8;
constexpr int64_t RADIUS_OFFSET = 2;
constexpr int64_t DPETHS_OFFSET = 2;
constexpr int64_t MEANS2D_OFFSET = 2;
constexpr int64_t MEANSCULLING_OFFSET = 2;
constexpr int64_t PROCESS_DPETHS_OFFSET = 3;
constexpr int64_t COVARS2D_OFFSET = 3;
constexpr int64_t COVARS2DCULLING_OFFSET = 2;
constexpr int64_t CONICSCULLING_OFFSET = 2;
constexpr int64_t CONICSCULLINGGM_OFFSET = 3;

constexpr int64_t COLORCULLING_OFFSET = 2;
constexpr int64_t COLORCULLINGGM_OFFSET = 3;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

template <typename T> __aicore__ inline T Min(T a, T b) { return a > b ? b : a; }

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

__aicore__ inline int64_t Align256(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + MASK_BATCH_SIZE - 1) / MASK_BATCH_SIZE * MASK_BATCH_SIZE / bytes;
}

template <HardEvent event> __aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace GaussianFilterNs
#endif // GAUSSIAN_FILTER_COMMON_H