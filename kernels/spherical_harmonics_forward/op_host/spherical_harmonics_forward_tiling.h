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

#ifndef SPHERICAL_HARMONICS_FORWARD_TILING_H
#define SPHERICAL_HARMONICS_FORWARD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SphericalHarmonicsForwardTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, taskNum);
    TILING_DATA_FIELD_DEF(uint32_t, coeffNum);
    TILING_DATA_FIELD_DEF(uint32_t, degreeUsed);
    TILING_DATA_FIELD_DEF(uint32_t, degreeNum);
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

REGISTER_TILING_DATA_CLASS(SphericalHarmonicsForward, SphericalHarmonicsForwardTilingData)
}

#endif