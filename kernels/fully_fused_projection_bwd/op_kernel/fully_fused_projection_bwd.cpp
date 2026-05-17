/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fully_fused_projection_bwd.h"
using namespace AscendC;
using namespace FullyFusedProjectionBwdNs;

extern "C" __global__ __aicore__ void
fully_fused_projection_bwd(GM_ADDR means, GM_ADDR quats, GM_ADDR scales, GM_ADDR conics, GM_ADDR viewmats, GM_ADDR Ks,
                           GM_ADDR vMeans2d, GM_ADDR vDepths, GM_ADDR vConics, GM_ADDR vColorsCulling,
                           GM_ADDR vOpacitiesCulling, GM_ADDR filter, GM_ADDR compensations, GM_ADDR vPW,
                           GM_ADDR vQuats, GM_ADDR vScales, GM_ADDR vR, GM_ADDR vColors, GM_ADDR vOpacities,
                           GM_ADDR workspace, GM_ADDR tiling)

{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    FullyFusedProjectionBwd op;
    if (TILING_KEY_IS(1)) {
        op.Init(means, quats, scales, conics, viewmats, Ks, vMeans2d, vDepths, vConics, vColorsCulling,
                vOpacitiesCulling, filter, compensations, vPW, vQuats, vScales, vR, vColors, vOpacities, workspace,
                &pipe, &tilingData);
        op.Process();
    }
}