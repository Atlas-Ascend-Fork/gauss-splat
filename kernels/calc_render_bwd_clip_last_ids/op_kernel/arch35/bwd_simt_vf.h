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

// Macro to call BwdSimtPixel with gsAttr row fields.
#define BWD_PIX_CALL(S, PX, PY, VR, VG, VB, VD, ROW, BUF) \
    BwdSimtPixel<HasBg>(BUF, ROW, PX, PY, \
        BUF[ROW+ATTR_CONIC_0], BUF[ROW+ATTR_CONIC_1], \
        BUF[ROW+ATTR_CONIC_2], BUF[ROW+ATTR_OPACITY], \
        BUF[ROW+ATTR_COLOR_R], BUF[ROW+ATTR_COLOR_G], \
        BUF[ROW+ATTR_COLOR_B], BUF[ROW+ATTR_DEPTH], \
        VR, VG, VB, VD, S)

// Per-pixel mutable state extracted from BwdSimtCompute to keep the function
// under the 50-line codecheck limit. All gradient accumulators and
// transmittance state for one pixel live here.
struct PixelState {
    float cumR, cumG, cumB, cumD;     // running cumsum of (alpha*trans*color)
    float vx, vy, vc0, vc1, vc2, vo;  // per-gaussian gradient outputs
    float r, g, b, d;                 // per-gaussian color/depth contribution
    bool valid;                       // pixel still active for this gaussian
    float vTbg, tFinal;               // bg_color path seeds (HasBg only)
#if GS_BWD_TRANS_MUL
    float trans;                      // multiplicative reverse transmittance
#else
    float sum, err;                   // log-domain Kahan sum + compensation
#endif
};

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
        uint64_t rowAddr = static_cast<uint64_t>(flushGid) * NUM_GS_ATTRIBUTES;
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

// Read per-pixel UB state into PixelState.
template<bool HasBg>
__simt_callee__ __aicore__ inline void InitPixelState(
    PixelState& s, int pix,
    __ubuf__ float* cumsumBufUb, __ubuf__ float* lastCumsumUb,
    __ubuf__ float* errorUb, __ubuf__ float* tFinalUb)
{
    s = {};
    s.cumR = cumsumBufUb[CUMSUM_R_OFFSET+pix];
    s.cumG = cumsumBufUb[CUMSUM_G_OFFSET+pix];
    s.cumB = cumsumBufUb[CUMSUM_B_OFFSET+pix];
    s.cumD = cumsumBufUb[CUMSUM_D_OFFSET+pix];
#if GS_BWD_TRANS_MUL
    s.trans = lastCumsumUb[pix];
#else
    s.sum = lastCumsumUb[pix]; s.err = errorUb[pix];
#endif
    if constexpr (HasBg) { s.tFinal = tFinalUb[pix]; }
}

// Write per-pixel state back to UB.
__simt_callee__ __aicore__ inline void WriteBackPixel(
    const PixelState& s, int pix,
    __ubuf__ float* lastCumsumUb, __ubuf__ float* errorUb,
    __ubuf__ float* cumsumBufUb)
{
#if GS_BWD_TRANS_MUL
    lastCumsumUb[pix] = s.trans;
#else
    lastCumsumUb[pix] = s.sum; errorUb[pix] = s.err;
#endif
    cumsumBufUb[CUMSUM_R_OFFSET+pix] = s.cumR;
    cumsumBufUb[CUMSUM_G_OFFSET+pix] = s.cumG;
    cumsumBufUb[CUMSUM_B_OFFSET+pix] = s.cumB;
    cumsumBufUb[CUMSUM_D_OFFSET+pix] = s.cumD;
}

// Per-pixel gradient computation for one gaussian row. Extracted to keep
// BwdSimtCompute under the 50-line codecheck limit.
template<bool HasBg>
__simt_callee__ __aicore__ inline void BwdSimtPixel(
    __ubuf__ float* gsAttrBatchUb, int row, float px, float py,
    float c0, float c1, float c2, float opacity,
    float cr, float cg, float cb, float cd,
    float vr, float vg, float vb, float vd, PixelState& s)
{
    const float dx = gsAttrBatchUb[row+ATTR_MEAN_X] - px;
    const float dy = gsAttrBatchUb[row+ATTR_MEAN_Y] - py;
    float weight = expf(-0.5f*(c0*dx*dx + 2.0f*c1*dx*dy + c2*dy*dy));
    const float raw = opacity * weight;
    const float alpha = fminf(ALPHA_UPPER_BOUND, raw);
    s.valid = alpha >= ALPHA_THRESHOLD;
    s.vx=0; s.vy=0; s.vc0=0; s.vc1=0; s.vc2=0; s.vo=0; s.r=0; s.g=0; s.b=0; s.d=0;
    if (s.valid) {
        const float oneMinus = 1.0f - alpha;
#if (GS_BWD_TRANS_MUL || GS_BWD_VA_FACTOR)
        const float inv = 1.0f / oneMinus;
#endif
#if GS_BWD_TRANS_MUL
        s.trans = s.trans * inv;
        const float transm = s.trans;
#else
        const float minus = logf(oneMinus) + s.err;
        const float next = s.sum - minus;
        s.err = (next - s.sum) + minus;
        s.sum = next;
        const float transm = expf(s.sum);
#endif
        const float at = alpha * transm;
        s.r=at*vr; s.g=at*vg; s.b=at*vb; s.d=at*vd;
#if GS_BWD_VA_FACTOR
        const float dotp  = cr*vr + cg*vg + cb*vb + cd*vd;
        const float cdotp = s.cumR*vr + s.cumG*vg + s.cumB*vb + s.cumD*vd;
        float va = transm*dotp - cdotp*inv;
#else
        float va = (transm*cr-s.cumR/oneMinus)*vr
                 + (transm*cg-s.cumG/oneMinus)*vg
                 + (transm*cb-s.cumB/oneMinus)*vb
                 + (transm*cd-s.cumD/oneMinus)*vd;
#endif
        if constexpr (HasBg) { va += -s.vTbg * s.tFinal / oneMinus; }
        if (raw > ALPHA_UPPER_BOUND) { va=0.0f; weight=0.0f; }
        s.vo = va*weight;
        const float vw = -opacity*s.vo;
        s.vx = (c0*dx+c1*dy)*vw;
        s.vy = (c1*dx+c2*dy)*vw;
        s.vc0 = 0.5f*vw*dx*dx; s.vc1 = vw*dx*dy; s.vc2 = 0.5f*vw*dy*dy;
        s.cumR += at*cr; s.cumG += at*cg; s.cumB += at*cb; s.cumD += at*cd;
    }
}


// Specialize the launch bound as well as the launch dimension. UB channel
// strides remain THREAD_NUM in both variants; they are not blockDim strides.
// bg_color 特性：bgColorUb 为前向叠加过的同一份背景色 [3]，
// HasBg=false 时整条背景色梯度路径不执行（输出与改动前逐位一致）。
template<int Threads, bool HasBg>
__simt_vf__ __aicore__ __launch_bounds__(Threads) inline void BwdSimtCompute(
    __gm__ float* gsGm,
    __ubuf__ float* tileCoordXUb, __ubuf__ float* tileCoordYUb,
    __ubuf__ float* gsAttrBatchUb, __ubuf__ float* vColorDepthUb,
    __ubuf__ float* lastCumsumUb, __ubuf__ float* errorUb,
    __ubuf__ float* cumsumBufUb, __ubuf__ int32_t* lastIdsUb,
    __ubuf__ int64_t* scalarParamsUb, __ubuf__ int64_t* gsIdsBatchUb,
    __gm__ float* vGsGm,
    __ubuf__ float* bgColorUb, __ubuf__ float* tFinalUb)
{
    (void)gsGm; const int tid = static_cast<int>(AscendC::Simt::GetThreadIdx());
    const int lane = tid % WARP_SIZE, last = lastIdsUb[tid];
    const int64_t batchStart = scalarParamsUb[SCALAR_BATCH_START]; const int batchSize = static_cast<int>(scalarParamsUb[SCALAR_BATCH_SIZE]);
    PixelState s;
    InitPixelState<HasBg>(s, tid, cumsumBufUb, lastCumsumUb, errorUb, tFinalUb);
    const float vr=vColorDepthUb[VCOLOR_R_OFFSET+tid], vg=vColorDepthUb[VCOLOR_G_OFFSET+tid];
    const float vb=vColorDepthUb[VCOLOR_B_OFFSET+tid], vd=vColorDepthUb[VCOLOR_VDEPTH_OFFSET+tid];
    const float px=tileCoordXUb[tid], py=tileCoordYUb[tid];
    if constexpr (HasBg) { s.vTbg = bgColorUb[0]*vr + bgColorUb[1]*vg + bgColorUb[2]*vb; }
    for (int t = batchSize - 1; t >= 0; --t) {
        s.valid = batchStart + t <= last;
        if (!asc_any(static_cast<int>(s.valid))) { continue; }
        const int row = t * GS_ATTR_STRIDE;
        s.vx=0;s.vy=0;s.vc0=0;s.vc1=0;s.vc2=0;s.vo=0;s.r=0;s.g=0;s.b=0;s.d=0;
        if (s.valid) { BWD_PIX_CALL(s, px, py, vr, vg, vb, vd, row, gsAttrBatchUb); }
        if (!asc_any(static_cast<int>(s.valid))) { continue; }
        ScatterGradient(vGsGm, static_cast<int>(gsIdsBatchUb[t]), lane,
                        s.vx, s.vy, s.vc0, s.vc1, s.vc2, s.vo,
                        s.r, s.g, s.b, s.d);
    }
    WriteBackPixel(s, tid, lastCumsumUb, errorUb, cumsumBufUb);
    asc_threadfence_block();
    asc_syncthreads();
}
} // namespace NsCalcRenderBwdClipLastids
