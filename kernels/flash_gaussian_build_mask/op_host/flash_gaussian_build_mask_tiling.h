/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_GAUSSIAN_BUILD_MASK_TILING_H
#define FLASH_GAUSSIAN_BUILD_MASK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FlashGaussianBuildMaskTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, tileNumPerScore);
    TILING_DATA_FIELD_DEF(uint32_t, tileNumPerLcore);
    TILING_DATA_FIELD_DEF(uint32_t, numScore);
    TILING_DATA_FIELD_DEF(uint32_t, numLcore);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumPerLoop);
    TILING_DATA_FIELD_DEF(uint32_t, numTile);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, cameraNum);
    TILING_DATA_FIELD_DEF(uint32_t, gaussNum);
    TILING_DATA_FIELD_DEF(float, tileSize);
    TILING_DATA_FIELD_DEF(float, imageWidth);
    TILING_DATA_FIELD_DEF(float, imageHeight);
    TILING_DATA_FIELD_DEF(uint64_t, ubTotalSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FlashGaussianBuildMask, FlashGaussianBuildMaskTilingData)
}

#endif