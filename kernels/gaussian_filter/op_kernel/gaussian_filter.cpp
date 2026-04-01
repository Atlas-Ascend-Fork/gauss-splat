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

#include "gaussian_filter.h"
using namespace AscendC;
using namespace GaussianFilterNs;

extern "C" __global__ __aicore__ void
gaussian_filter(GM_ADDR means, GM_ADDR colors, GM_ADDR det, GM_ADDR opacities, GM_ADDR means2d, GM_ADDR depths,
                GM_ADDR radius, GM_ADDR conics, GM_ADDR covars2d, GM_ADDR compensations, GM_ADDR meansCulling,
                GM_ADDR colorsCulling, GM_ADDR means2dCulling, GM_ADDR depthsCulling, GM_ADDR radiusCulling,
                GM_ADDR covars2dCulling, GM_ADDR conicsCulling, GM_ADDR opacitiesCulling, GM_ADDR filter, GM_ADDR cnt,
                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        int64_t hasCompensations = tilingData.hasCompensations;
        if (hasCompensations) {
            GaussianFilter<true> op;
            op.Init(means, colors, det, opacities, means2d, depths, radius, conics, covars2d, compensations,
                    meansCulling, colorsCulling, means2dCulling, depthsCulling, radiusCulling, covars2dCulling,
                    conicsCulling, opacitiesCulling, filter, cnt, workspace, &pipe, &tilingData);
            op.Process();
        } else {
            GaussianFilter<false> op;
            op.Init(means, colors, det, opacities, means2d, depths, radius, conics, covars2d, compensations,
                    meansCulling, colorsCulling, means2dCulling, depthsCulling, radiusCulling, covars2dCulling,
                    conicsCulling, opacitiesCulling, filter, cnt, workspace, &pipe, &tilingData);
            op.Process();
        }
    }
}