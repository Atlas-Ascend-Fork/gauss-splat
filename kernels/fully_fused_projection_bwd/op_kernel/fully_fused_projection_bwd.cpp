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