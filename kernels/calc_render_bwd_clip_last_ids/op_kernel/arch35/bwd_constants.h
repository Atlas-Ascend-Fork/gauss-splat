/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// kernels/calc_render_bwd_clip_last_ids/op_kernel/arch35/bwd_constants.h
// =============================================================================
//
// ROLE: Compile-time constants for CalcRenderBwdClipLastids (SIMT, arch35 / Ascend950).
//
//   Single source of truth: docs/calc_render_bwd_clip_last_ids/REQUIREMENTS.md §5.6.3
//   (客户实现要求文档 §5.6 常数契约) — shared by the SIMD host orchestrator
//   (calc_render_bwd_clip_last_ids_apt.cpp) and the SIMT VF compute body
//   (bwd_simt_vf.h). Numerical thresholds are compile-time constants by design
//   (Kernel.md §9.2: 禁止写入 TilingData / 以 tilingKey 切换).
//
//   Layout constants follow REQUIREMENTS §5.6.4 UB partition table (12
//   partitions, single buffer ~74KB <= 184KB); scalarParams is 5 x int64
//   (8B aligned); gsAttr rows are padded to a fixed 16-float (64B) stride.
// =============================================================================

#pragma once

#include <cstdint>

namespace NsCalcRenderBwdClipLastids {

// ---------------------------------------------------------------------------
// Thread / batch layout (REQUIREMENTS §5.6.3)
// ---------------------------------------------------------------------------
constexpr int THREAD_NUM = 1024;           // SIMT threads per VF call (1 thread = 1 pixel)
constexpr int PIXEL_PER_THREAD = 1;
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = THREAD_NUM / WARP_SIZE;
constexpr int KERNEL_GS_PER_BATCH = 256;   // gaussians per reverse batch (gsAttrBatch/gsIdsBatch capacity)
constexpr int MAX_TILE_SIZE = 32;          // calPixel = min(nPixel, MAX_TILE_SIZE^2)
constexpr int MIN_DATACOPY_LEN = 16;       // gsAttr row copy length in floats (64B aligned)

// gs packed attribute column order: [N, 10]
constexpr int NUM_GS_ATTRIBUTES = 10;
constexpr int GS_ATTR_STRIDE = 16;         // 10 valid cols + 6 padding floats (64B aligned copy)
constexpr int ATTR_MEAN_X = 0;
constexpr int ATTR_MEAN_Y = 1;
constexpr int ATTR_CONIC_0 = 2;
constexpr int ATTR_CONIC_1 = 3;
constexpr int ATTR_CONIC_2 = 4;
constexpr int ATTR_OPACITY = 5;
constexpr int ATTR_COLOR_R = 6;
constexpr int ATTR_COLOR_G = 7;
constexpr int ATTR_COLOR_B = 8;
constexpr int ATTR_DEPTH = 9;

// vColorDepthUb / cumsumBufUb RGBD contiguous channel offsets (floats)
constexpr int VCOLOR_R_OFFSET = 0;
constexpr int VCOLOR_G_OFFSET = 1024;
constexpr int VCOLOR_B_OFFSET = 2048;
constexpr int VCOLOR_VDEPTH_OFFSET = 3072;
constexpr int CUMSUM_R_OFFSET = 0;
constexpr int CUMSUM_G_OFFSET = 1024;
constexpr int CUMSUM_B_OFFSET = 2048;
constexpr int CUMSUM_D_OFFSET = 3072;

// scalarParamsUb 5 x int64 slots (REQUIREMENTS §5.6.2: prevOffset/currOffset/
// batchSize/batchStart/nGauss; prevOffset/currOffset/nGauss defensive, VJP unused)
constexpr int SCALAR_PREV_OFFSET = 0;
constexpr int SCALAR_CURR_OFFSET = 1;
constexpr int SCALAR_BATCH_SIZE = 2;
constexpr int SCALAR_BATCH_START = 3;
constexpr int SCALAR_NGAUSS = 4;
constexpr int SCALAR_FIRST_BATCH = 5;   // BWD trans-mul: 1 = first batch of this tile (init trans from expf(logT))
constexpr int SCALAR_PARAMS_COUNT = 6;

// ---------------------------------------------------------------------------
// Numerical thresholds (Kernel.md §9.2 — compile-time only, never in TilingData)
// ---------------------------------------------------------------------------
constexpr float ALPHA_THRESHOLD = 1.0f / 255.0f;      // per-pixel valid lower bound
constexpr float ALPHA_UPPER_BOUND = 0.999f;           // alpha clip upper bound (1-alpha >= 1e-3)
constexpr float PER_PIXEL_DONE_THRESHOLD = 1e-4f;     // reserved (FWD done semantics; BWD unused)

constexpr int LAST_IDS_UB_FLOATS = 1024;              // lastIdsUb capacity (int32 x 1024)

// ---------------------------------------------------------------------------
// GS_BWD_VA_FACTOR switch (compile-time; default ON, revertible/comparable).
// When 1: factor the per-(pixel,gaussian) vGs 'va' so the four per-channel
// divisions cumX/oneMinus become one reciprocal inv=1/oneMinus plus multiplies:
//   va = trans*(cr*vr+cg*vg+cb*vb+cd*vd) - inv*(cumR*vr+cumG*vg+cumB*vb+cumD*vd)
// Mathematically identical (tiny fp reassociation). Flip to 0 and rebuild to
// compare/revert; 0 restores the original exact division form.
// ---------------------------------------------------------------------------
#ifndef GS_BWD_VA_FACTOR
#define GS_BWD_VA_FACTOR 1
#endif

// ---------------------------------------------------------------------------
// GS_BWD_TRANS_MUL switch (compile-time; default OFF).
// The reverse loop currently reconstructs the log-transmittance
//   sum -= log(1 - alpha); trans = expf(sum)
// per (pixel,gaussian) pair (two transcendentals). Because the forward step is
// the multiplicative tPrev*(1-alpha), the reverse step is exactly
//   trans /= (1-alpha) = trans * (1/(1-alpha))
// so we can carry `trans` multiplicatively and drop both logf and expf(sum)
// per pair (keeping only expf(quad) for the weight). One expf per thread per
// tile initializes trans = expf(lastCumsum). fp reassociation only; the Kahan
// log-domain error term is not used in this mode. Flip to 1 and rebuild
// to enable the multiplicative form.
// ---------------------------------------------------------------------------
#ifndef GS_BWD_TRANS_MUL
#define GS_BWD_TRANS_MUL 0
#endif

} // namespace NsCalcRenderBwdClipLastids
