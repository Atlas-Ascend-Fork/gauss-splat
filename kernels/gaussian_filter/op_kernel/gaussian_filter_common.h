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