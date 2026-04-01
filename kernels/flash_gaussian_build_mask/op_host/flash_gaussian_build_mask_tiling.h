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