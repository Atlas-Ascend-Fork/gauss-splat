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

#include "spherical_harmonics_bwd.h"
using namespace AscendC;
using namespace SphericalHarmonicsBwdNs;

extern "C" __global__ __aicore__ void spherical_harmonics_bwd(GM_ADDR dirs, GM_ADDR coeffs, GM_ADDR vColors,
                                                              GM_ADDR vDirs, GM_ADDR vCoeffs, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    SphericalHarmonicsBwd op;
    if (TILING_KEY_IS(1)) {
        op.Init(dirs, coeffs, vColors, vDirs, vCoeffs, &pipe, &tilingData);
        op.Process();
    }
}