/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_COLOR_EXP_GRAD_VF_H_H
#define CALC_COLOR_EXP_GRAD_VF_H_H
#include "kernel_operator.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_atomic_functions.h"

template<int Threads>
__simt_vf__ __aicore__ __launch_bounds__(Threads)
    inline void CalcGradUseGrid(
        __ubuf__ float *colors, __ubuf__ float *vColorsIn,
        __ubuf__ float *vColorsOut, __ubuf__ float *vExposure,
        __ubuf__ float *gridExposure, __ubuf__ float *weight,
        __ubuf__ int32_t *index, uint32_t validTaskNum, uint32_t taskNum,
        int32_t gh, int32_t gd, bool clamp, bool calcExpGrad
    )
{
    const int tid = static_cast<int>(AscendC::Simt::GetThreadIdx());
    const int stride = static_cast<int>(AscendC::Simt::GetThreadNum());
    for (int i = tid; i < validTaskNum; i+=stride) {
        // calculate weight
        float wx_r = weight[i];
        float wx_l = 1.0f - wx_r;
        float wy_r = weight[i + taskNum];
        float wy_l = 1.0f - wy_r;
        float wz_r = weight[i + 2 * taskNum];
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
                exposure[j] += w[k] * gridExposure[id[k] * numExp + j];
            }
        }

        float cr = colors[i];
        float cg = colors[i + taskNum];
        float cb = colors[i + 2 * taskNum];
        float vr = vColorsIn[i];
        float vg = vColorsIn[i + taskNum];
        float vb = vColorsIn[i + 2 * taskNum];

        if (clamp) {
            // apply exposure
            float cr0 = exposure[0] * cr + exposure[1] * cg + exposure[2] * cb + exposure[3];
            float cg0 = exposure[4] * cr + exposure[5] * cg + exposure[6] * cb + exposure[7];
            float cb0 = exposure[8] * cr + exposure[9] * cg + exposure[10] * cb + exposure[11];
            if (cr0 < 0.0f) vr = 0.0f;
            if (cg0 < 0.0f) vg = 0.0f;
            if (cb0 < 0.0f) vb = 0.0f;
        }

        // v_colors_out = exposure.a^T * v_colors_in
        vColorsOut[i] = exposure[0] * vr + exposure[4] * vg + exposure[8] * vb;
        vColorsOut[i + taskNum] = exposure[1] * vr + exposure[5] * vg + exposure[9] * vb;
        vColorsOut[i + 2 * taskNum] = exposure[2] * vr + exposure[6] * vg + exposure[10] * vb;

        if (calcExpGrad) {
            // v_exposure = {out_dot(colors, v_colors_in), v_colors_in}
            exposure[0] = cr * vr;
            exposure[1] = cg * vr;
            exposure[2] = cb * vr;
            exposure[3] = vr;
            exposure[4] = cr * vg;
            exposure[5] = cg * vg;
            exposure[6] = cb * vg;
            exposure[7] = vg;
            exposure[8] = cr * vb;
            exposure[9] = cg * vb;
            exposure[10] = cb * vb;
            exposure[11] = vb;

            // calculate grid exposure grad
            for (int j = 0; j < numIndex; j++) {
                for (int k = 0; k < numExp; k++) {
                    asc_atomic_add(vExposure + id[j] * numExp + k, w[j] * exposure[k]);
                }
            }
        }
    }
}

template<int Threads>
__simt_vf__ __aicore__ __launch_bounds__(Threads)
    inline void CalcGradGlobal(
        __ubuf__ float *colors, __ubuf__ float *vColorsIn,
        __ubuf__ float *vColorsOut, __ubuf__ float *vExposure,
        __ubuf__ float *exposure, uint32_t validTaskNum,
        uint32_t taskNum, bool clamp, bool calcExpGrad
    )
{
    const int tid = static_cast<int>(AscendC::Simt::GetThreadIdx());
    const int stride = static_cast<int>(AscendC::Simt::GetThreadNum());
    for (int i = tid; i < validTaskNum; i+=stride) {
        float cr = colors[i];
        float cg = colors[i + taskNum];
        float cb = colors[i + 2 * taskNum];
        float vr = vColorsIn[i];
        float vg = vColorsIn[i + taskNum];
        float vb = vColorsIn[i + 2 * taskNum];

        if (clamp) {
            // apply exposure
            float cr0 = exposure[0] * cr + exposure[1] * cg + exposure[2] * cb + exposure[3];
            float cg0 = exposure[4] * cr + exposure[5] * cg + exposure[6] * cb + exposure[7];
            float cb0 = exposure[8] * cr + exposure[9] * cg + exposure[10] * cb + exposure[11];
            if (cr0 < 0.0f) vr = 0.0f;
            if (cg0 < 0.0f) vg = 0.0f;
            if (cb0 < 0.0f) vb = 0.0f;
        }

        // v_colors_out = exposure.a^T * v_colors_in
        vColorsOut[i] = exposure[0] * vr + exposure[4] * vg + exposure[8] * vb;
        vColorsOut[i + taskNum] = exposure[1] * vr + exposure[5] * vg + exposure[9] * vb;
        vColorsOut[i + 2 * taskNum] = exposure[2] * vr + exposure[6] * vg + exposure[10] * vb;

        if (calcExpGrad) {
            // v_exposure = {out_dot(colors, v_colors_in), v_colors_in}
            asc_atomic_add(vExposure, cr * vr);
            asc_atomic_add(vExposure + 1, cg * vr);
            asc_atomic_add(vExposure + 2, cb * vr);
            asc_atomic_add(vExposure + 3, vr);
            asc_atomic_add(vExposure + 4, cr * vg);
            asc_atomic_add(vExposure + 5, cg * vg);
            asc_atomic_add(vExposure + 6, cb * vg);
            asc_atomic_add(vExposure + 7, vg);
            asc_atomic_add(vExposure + 8, cr * vb);
            asc_atomic_add(vExposure + 9, cg * vb);
            asc_atomic_add(vExposure + 10, cb * vb);
            asc_atomic_add(vExposure + 11, vb);
        }
    }
}

#endif // CALC_COLOR_EXP_GRAD_VF_H_H