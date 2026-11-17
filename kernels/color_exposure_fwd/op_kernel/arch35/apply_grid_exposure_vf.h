/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_GRID_EXPOSURE_VF_H_H
#define APPLY_GRID_EXPOSURE_VF_H_H
#include "kernel_operator.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_atomic_functions.h"

template<int Threads>
__simt_vf__ __aicore__ __launch_bounds__(Threads)
    inline void ApplyGridExposureVF(
        __ubuf__ float *wxy, __ubuf__ float *wz, __ubuf__ int32_t *index,
        __ubuf__ float *gridExp, __ubuf__ float *colors, __ubuf__ float *output,
        bool clamp, uint32_t validTaskNum, uint32_t taskNum, int32_t gh, int32_t gd
    )
{
    const int tid = static_cast<int>(AscendC::Simt::GetThreadIdx());
    const int stride = static_cast<int>(AscendC::Simt::GetThreadNum());
    for (int i = tid; i < validTaskNum; i+=stride) {
        // calculate weight
        float wx_r = wxy[i];
        float wx_l = 1.0f - wx_r;
        float wy_r = wxy[i + taskNum];
        float wy_l = 1.0f - wy_r;
        float wz_r = wz[i];
        float wz_l = 1.0f - wz_r;
        float w[] = {
            wx_l * wy_l * wz_l,
            wx_l * wy_l * wz_r,
            wx_l * wy_r * wz_l,
            wx_l * wy_r * wz_r,
            wx_r * wy_l * wz_l,
            wx_r * wy_l * wz_r,
            wx_r * wy_r * wz_l,
            wx_r * wy_r * wz_r,
        };

        // calculate id
        int idx = index[i];
        int idy = index[i + taskNum];
        int idz = index[i + 2 * taskNum];
        int id[] = {
            (idx * gh + idy) * gd + idz,
            (idx * gh + idy) * gd + idz + 1,
            (idx * gh + idy + 1) * gd + idz,
            (idx * gh + idy + 1) * gd + idz + 1,
            ((idx + 1) * gh + idy) * gd + idz,
            ((idx + 1) * gh + idy) * gd + idz + 1,
            ((idx + 1) * gh + idy + 1) * gd + idz,
            ((idx + 1) * gh + idy + 1) * gd + idz + 1,
        };

        // interpolate
        const int numExp = 12;
        const int numIndex = 8;
        float exposure[numExp] = { 0.0f };
        for (int j = 0; j < numExp; j++) {
            for (int k = 0; k < numIndex; k++) {
                exposure[j] += w[k] * gridExp[id[k] * numExp + j];
            }
        }

        // apply grid exposure
        float cr = colors[i];
        float cg = colors[i + taskNum];
        float cb = colors[i + 2 * taskNum];
        float cr0 = exposure[0] * cr + exposure[1] * cg + exposure[2] * cb + exposure[3];
        float cg0 = exposure[4] * cr + exposure[5] * cg + exposure[6] * cb + exposure[7];
        float cb0 = exposure[8] * cr + exposure[9] * cg + exposure[10] * cb + exposure[11];
        if (clamp) {
            cr0 = max(cr0, 0.0f);
            cg0 = max(cg0, 0.0f);
            cb0 = max(cb0, 0.0f);
        }
        output[i] = cr0;
        output[i + taskNum] = cg0;
        output[i + 2 * taskNum] = cb0;
    }
}
#endif // APPLY_GRID_EXPOSURE_VF_H_H