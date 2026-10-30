/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Reverse rendering with the reference per-step Kahan state transition.
// One thread owns one pixel; all warp reductions are reached uniformly.
#pragma once
#include "kernel_operator.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_atomic_functions.h"
#include "bwd_constants.h"
namespace NsCalcRenderBwdClipLastids {
__simt_callee__ __aicore__ inline void ScatterGradient(
    __gm__ float* vGsGm, int flushGid, int laneId,
    float accMeanX, float accMeanY, float accConic0, float accConic1,
    float accConic2, float accOpac, float accColR, float accColG,
    float accColB, float accColD)
{
    float rMeanX  = asc_reduce_add(accMeanX);
    float rMeanY  = asc_reduce_add(accMeanY);
    float rConic0 = asc_reduce_add(accConic0);
    float rConic1 = asc_reduce_add(accConic1);
    float rConic2 = asc_reduce_add(accConic2);
    float rOpac   = asc_reduce_add(accOpac);
    float rColR   = asc_reduce_add(accColR);
    float rColG   = asc_reduce_add(accColG);
    float rColB   = asc_reduce_add(accColB);
    float rColD   = asc_reduce_add(accColD);
    if (laneId == 0) {
        uint64_t rowAddr = static_cast<uint64_t>(flushGid) * NUM_GS_ATTRIBUTES;   // GM address in uint64
        asc_atomic_add(vGsGm + rowAddr + ATTR_MEAN_X,  rMeanX);
        asc_atomic_add(vGsGm + rowAddr + ATTR_MEAN_Y,  rMeanY);
        asc_atomic_add(vGsGm + rowAddr + ATTR_CONIC_0, rConic0);
        asc_atomic_add(vGsGm + rowAddr + ATTR_CONIC_1, rConic1);
        asc_atomic_add(vGsGm + rowAddr + ATTR_CONIC_2, rConic2);
        asc_atomic_add(vGsGm + rowAddr + ATTR_OPACITY, rOpac);
        asc_atomic_add(vGsGm + rowAddr + ATTR_COLOR_R, rColR);
        asc_atomic_add(vGsGm + rowAddr + ATTR_COLOR_G, rColG);
        asc_atomic_add(vGsGm + rowAddr + ATTR_COLOR_B, rColB);
        asc_atomic_add(vGsGm + rowAddr + ATTR_DEPTH,   rColD);
    }
}


// Specialize the launch bound as well as the launch dimension. UB channel
// strides remain THREAD_NUM in both variants; they are not blockDim strides.
template<int Threads>
__simt_vf__ __aicore__ __launch_bounds__(Threads) inline void BwdSimtCompute(
    __gm__ float* gsGm,
    __ubuf__ float* tileCoordXUb, __ubuf__ float* tileCoordYUb,
    __ubuf__ float* gsAttrBatchUb, __ubuf__ float* vColorDepthUb,
    __ubuf__ float* lastCumsumUb, __ubuf__ float* errorUb,
    __ubuf__ float* cumsumBufUb, __ubuf__ int32_t* lastIdsUb,
    __ubuf__ int64_t* scalarParamsUb, __ubuf__ int64_t* gsIdsBatchUb,
    __gm__ float* vGsGm)
{
    (void)gsGm;
    const int tid = static_cast<int>(AscendC::Simt::GetThreadIdx());
    const int lane = tid % WARP_SIZE;
    const int last = lastIdsUb[tid];
    const int64_t batchStart = scalarParamsUb[SCALAR_BATCH_START];
    const int batchSize = static_cast<int>(scalarParamsUb[SCALAR_BATCH_SIZE]);
    float sum = lastCumsumUb[tid], error = errorUb[tid];
    float cumR = cumsumBufUb[CUMSUM_R_OFFSET + tid];
    float cumG = cumsumBufUb[CUMSUM_G_OFFSET + tid];
    float cumB = cumsumBufUb[CUMSUM_B_OFFSET + tid];
    float cumD = cumsumBufUb[CUMSUM_D_OFFSET + tid];
    const float vr = vColorDepthUb[VCOLOR_R_OFFSET + tid];
    const float vg = vColorDepthUb[VCOLOR_G_OFFSET + tid];
    const float vb = vColorDepthUb[VCOLOR_B_OFFSET + tid];
    const float vd = vColorDepthUb[VCOLOR_VDEPTH_OFFSET + tid];
    const float px = tileCoordXUb[tid], py = tileCoordYUb[tid];
    for (int t = batchSize - 1; t >= 0; --t) {
        bool valid = batchStart + t <= last;
        if (!asc_any(static_cast<int>(valid))) { continue; }
        float vx=0, vy=0, vc0=0, vc1=0, vc2=0, vo=0;
        float r=0, g=0, b=0, d=0;
        if (valid) {
            const int row=t*GS_ATTR_STRIDE;
            const float dx=gsAttrBatchUb[row+ATTR_MEAN_X]-px;
            const float dy=gsAttrBatchUb[row+ATTR_MEAN_Y]-py;
            const float c0=gsAttrBatchUb[row+ATTR_CONIC_0];
            const float c1=gsAttrBatchUb[row+ATTR_CONIC_1];
            const float c2=gsAttrBatchUb[row+ATTR_CONIC_2];
            const float opacity=gsAttrBatchUb[row+ATTR_OPACITY];
            float weight=expf(-0.5f*(c0*dx*dx+2.0f*c1*dx*dy+c2*dy*dy));
            const float raw=opacity*weight;
            const float alpha=fminf(ALPHA_UPPER_BOUND,raw);
            valid=alpha>=ALPHA_THRESHOLD;
            if (valid) {
                const float oneMinus=1.0f-alpha;
                const float minus=logf(oneMinus)+error;
                const float next=sum-minus;
                error=(next-sum)+minus;
                sum=next;
                const float trans=expf(sum);
                const float at=alpha*trans;
                const float cr=gsAttrBatchUb[row+ATTR_COLOR_R];
                const float cg=gsAttrBatchUb[row+ATTR_COLOR_G];
                const float cb=gsAttrBatchUb[row+ATTR_COLOR_B];
                const float cd=gsAttrBatchUb[row+ATTR_DEPTH];
                r=at*vr; g=at*vg; b=at*vb; d=at*vd;
                float va=(trans*cr-cumR/oneMinus)*vr
                        +(trans*cg-cumG/oneMinus)*vg
                        +(trans*cb-cumB/oneMinus)*vb
                        +(trans*cd-cumD/oneMinus)*vd;
                if (raw>ALPHA_UPPER_BOUND) { va=0.0f; weight=0.0f; }
                vo=va*weight;
                const float vw=-opacity*vo;
                vx=(c0*dx+c1*dy)*vw;
                vy=(c1*dx+c2*dy)*vw;
                vc0=0.5f*vw*dx*dx; vc1=vw*dx*dy; vc2=0.5f*vw*dy*dy;
                cumR+=at*cr; cumG+=at*cg; cumB+=at*cb; cumD+=at*cd;
            }
        }
        if (!asc_any(static_cast<int>(valid))) { continue; }
        ScatterGradient(vGsGm,static_cast<int>(gsIdsBatchUb[t]),lane,
                        vx,vy,vc0,vc1,vc2,vo,r,g,b,d);
    }
    lastCumsumUb[tid]=sum; errorUb[tid]=error;
    cumsumBufUb[CUMSUM_R_OFFSET+tid]=cumR;
    cumsumBufUb[CUMSUM_G_OFFSET+tid]=cumG;
    cumsumBufUb[CUMSUM_B_OFFSET+tid]=cumB;
    cumsumBufUb[CUMSUM_D_OFFSET+tid]=cumD;
    asc_threadfence_block();
    asc_syncthreads();
}
} // namespace NsCalcRenderBwdClipLastids
