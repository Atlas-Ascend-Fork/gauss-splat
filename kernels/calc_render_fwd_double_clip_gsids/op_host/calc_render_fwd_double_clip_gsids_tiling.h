/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_RENDER_FWD_DOUBLE_CLIP_GSIDS_TILING_H
#define CALC_RENDER_FWD_DOUBLE_CLIP_GSIDS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CalcRenderFwdDoubleClipGsidsTilingData)
    TILING_DATA_FIELD_DEF(int64_t, nPixel);
    TILING_DATA_FIELD_DEF(int64_t, tileNum);
    TILING_DATA_FIELD_DEF(int64_t, nGauss);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CalcRenderFwdDoubleClipGsids, CalcRenderFwdDoubleClipGsidsTilingData)
}

#endif // CALC_RENDER_FWD_DOUBLE_CLIP_GSIDS_TILING_H