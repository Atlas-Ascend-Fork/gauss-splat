/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_SSIM_FORWARD_TILING_H
#define FUSED_SSIM_FORWARD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedSsimForwardTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, bcNum);          // B * C (number of independent single-channel images)
TILING_DATA_FIELD_DEF(int64_t, height);
TILING_DATA_FIELD_DEF(int64_t, width);
TILING_DATA_FIELD_DEF(int64_t, windowSize);     // WS, odd
TILING_DATA_FIELD_DEF(int64_t, radius);         // WS / 2
TILING_DATA_FIELD_DEF(int64_t, coresPerImage);  // how many cores split one image's rows
TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);    // row-band height handled by one core
TILING_DATA_FIELD_DEF(int64_t, sysWsBytes);     // bytes reserved at the start of GM workspace by the runtime
TILING_DATA_FIELD_DEF(int64_t, perCoreWsFloats);// user GM scratch per core (in float units)
TILING_DATA_FIELD_DEF(int64_t, chunkW);
TILING_DATA_FIELD_DEF(int64_t, numWChunks);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedSsimForward, FusedSsimForwardTilingData)
} // namespace optiling
#endif // FUSED_SSIM_FORWARD_TILING_H
