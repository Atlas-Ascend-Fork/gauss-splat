/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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