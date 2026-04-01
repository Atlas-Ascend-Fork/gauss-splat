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

#ifndef QUAT_SCALES_TO_COVARS_TILING_H
#define QUAT_SCALES_TO_COVARS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(QuatScalesToCovarsTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSizeNum);
    TILING_DATA_FIELD_DEF(uint32_t, gaussianNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumPerScore);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumPerLcore);
    TILING_DATA_FIELD_DEF(uint32_t, numScore);
    TILING_DATA_FIELD_DEF(uint32_t, numLcore);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(int32_t, taskNumPerLoop);
    TILING_DATA_FIELD_DEF(uint64_t, ubTotalSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuatScalesToCovars, QuatScalesToCovarsTilingData)
}

#endif