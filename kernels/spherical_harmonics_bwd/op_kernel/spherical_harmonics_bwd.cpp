/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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