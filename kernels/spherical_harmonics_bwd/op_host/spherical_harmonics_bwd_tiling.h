/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPHERICAL_HARMONICS_BWD_TILING_H
#define SPHERICAL_HARMONICS_BWD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SphericalHarmonicsBwdTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, batchNum);
TILING_DATA_FIELD_DEF(int64_t, gaussNum);
TILING_DATA_FIELD_DEF(int64_t, degree);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, bufferLen);
TILING_DATA_FIELD_DEF(int64_t, blockLength);
TILING_DATA_FIELD_DEF(int64_t, lastcoreNum);
TILING_DATA_FIELD_DEF(int64_t, perloopNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SphericalHarmonicsBwd, SphericalHarmonicsBwdTilingData)
} // namespace optiling
#endif // SPHERICAL_HARMONICS_BWD_TILING_H