/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file gaussian_sort_common.h
 * \brief gaussian sort op kernel common
 */

#ifndef GAUSSIAN_SORT_COMMON_H
#define GAUSSIAN_SORT_COMMON_H

#include "kernel_operator.h"

namespace GaussianSortCommon {
using namespace AscendC;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t ONE_REPEAT_SORT_NUM = 32;
constexpr uint32_t ONE_REPEAT_CONCAT_NUM = 16;
constexpr uint32_t QUEUE_DEPTHS_NUM = 1;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t KVFACTOR = 2;
constexpr uint32_t MRGSORT_WS_TENSOR_NUM = 2;
constexpr uint32_t UINT8_BIT_NUM = 8;
constexpr uint32_t MRGSORT_OUT_MULT_NUM = 2;
constexpr uint32_t REPEAT_STRIDE_SIZE = 8;
constexpr float MAX_FP32 = 3.4e38;
// 向上取整
template <typename T>
__aicore__ inline T Ceil(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}
// 32B对齐
template <typename T>
__aicore__ inline T Align(T elementNum, T bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

// 不同流水事件接口
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}
}  // namespace GaussianSort

#endif