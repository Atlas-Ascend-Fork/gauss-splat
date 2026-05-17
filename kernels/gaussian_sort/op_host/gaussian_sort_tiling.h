/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file gaussian_sort_tiling.h
 * \brief gaussian sort op host tiling
 */

#ifndef GAUSSIAN_SORT_TILING_H
#define GAUSSIAN_SORT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GaussianSortTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, cameraNum);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
TILING_DATA_FIELD_DEF(uint32_t, gaussNum);
TILING_DATA_FIELD_DEF(uint32_t, scheduleNum);
TILING_DATA_FIELD_DEF(uint32_t, maxSortNum);      // UB单次最大支持排序高斯球数
TILING_DATA_FIELD_DEF(uint32_t, maxMaskNum);      // 最大高斯球相交数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GaussianSort, GaussianSortTilingData)
}  // namespace optiling

#endif