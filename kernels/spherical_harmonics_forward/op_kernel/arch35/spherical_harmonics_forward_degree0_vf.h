/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPHERICAL_HARMONICS_FORWARD_DEGREE0_VF_H
#define SPHERICAL_HARMONICS_FORWARD_DEGREE0_VF_H

#include "kernel_operator.h"

// Degree-0 elementwise specialization as a register-resident VF body.
//   dst[i] = scale * src[i]   for i in [0, count)
// src/dst are __ubuf__ float pointers. On Ascend950 (DAV_3510) the fp32 vector
// length is 64 (256-byte vector register), so the loop runs ceil(count/64)
// iterations and the trailing partial vector is masked.
__aicore__ inline void SphericalHarmonicsDegree0ScaleVf(__ubuf__ float *dst, __ubuf__ float *src,
                                                        uint32_t count, float scale)
{
    constexpr uint16_t VL = 64;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> vreg;
        AscendC::MicroAPI::MaskReg mask;
        uint32_t remain = count;
        uint16_t loops = static_cast<uint16_t>((count + static_cast<uint32_t>(VL) - 1U) / static_cast<uint32_t>(VL));
        for (uint16_t i = 0; i < loops; ++i) {
            uint32_t base = static_cast<uint32_t>(i) * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(remain);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg, src + base);
            AscendC::MicroAPI::Muls(vreg, vreg, scale, mask);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dst + base, vreg, mask);
        }
    }
}

#endif // SPHERICAL_HARMONICS_FORWARD_DEGREE0_VF_H
