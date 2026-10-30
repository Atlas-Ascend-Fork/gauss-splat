/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CalcRenderBwdClipLastids: guarded input copies, load-balanced schedule dispatch,
// cooperative output clearing, and per-step reverse Kahan SIMT computation.
// offsets layout: [VN cumulative bin ends | T scheduled tiles | T tile ends].
// VN is derived from offsets length, not from the device core count.
// Output is cleared within this operator; callers need not pre-zero vGs.

#include "kernel_operator.h"                           // Ascend C kernel framework (AscendC::)
#include "simt_api/common_functions.h"                 // asc_vf_call (C-style SIMT VF launch, Task 36 F3)
#include "bwd_constants.h"                             // single-source layout / threshold constants
#include "bwd_simt_vf.h"                               // SIMT VF compute body (Kernel.md §9.5)
// NOTE: no tiling struct header here — the entry raw-parses the first 3 int64
// tiling fields by offset (TilingData.md §1, baseline-consistent); including
// the tiling struct header would re-classify the kernel as "user-defined
// tiling struct" for the CANN OOM wrapper (see the entry comment).

using namespace AscendC;

namespace NsCalcRenderBwdClipLastids {

// ---------------------------------------------------------------------------
// Small int64 min/max helpers (device-side, no libm dependency)
// ---------------------------------------------------------------------------
__aicore__ inline int64_t MinI64(int64_t a, int64_t b) { return (a <= b) ? a : b; }
__aicore__ inline int64_t MaxI64(int64_t a, int64_t b) { return (a >= b) ? a : b; }

// ===========================================================================
// class CalcRenderBwdClipLastidsKernel — SIMD host orchestrator (single template,
// tilingKey=0). Per DESIGN-BRANCH-0.md §5.1/§5.2/§5.3.
// ===========================================================================
class CalcRenderBwdClipLastidsKernel {
public:
    __aicore__ inline CalcRenderBwdClipLastidsKernel() {}
    __aicore__ inline ~CalcRenderBwdClipLastidsKernel()
    {
        // Event release paired with the Init allocations (Kernel.md §4 contract ①)
        auto* tp = GetTPipePtr();
        if (tp != nullptr) {
            tp->ReleaseEventID<HardEvent::MTE2_V>(static_cast<TEventID>(evtMte2V_));
            tp->ReleaseEventID<HardEvent::MTE2_S>(static_cast<TEventID>(evtMte2S_));
            tp->ReleaseEventID<HardEvent::V_S>(static_cast<TEventID>(evtVS_));
            tp->ReleaseEventID<HardEvent::S_V>(static_cast<TEventID>(evtSV_));
        }
    }

    // -----------------------------------------------------------------------
    // Init — GM binding + UbBuffInit (184KB single buffer, 12 partitions)
    // + typed event allocation (DESIGN-BRANCH-0.md §5.1)
    // -----------------------------------------------------------------------
    __aicore__ inline void Init(
        GM_ADDR vColor, GM_ADDR vDepth, GM_ADDR lastCumsum, GM_ADDR error, GM_ADDR gs,
        GM_ADDR tileCoords, GM_ADDR offsets, GM_ADDR gsIds, GM_ADDR lastIds,
        int64_t nPixel, int64_t tileNum, int64_t nGauss,
        int64_t offsetsLen, int64_t gsIdsLen, GM_ADDR vGs)
    {
        nPixel_ = nPixel;
        tileNum_ = tileNum;
        nGauss_ = nGauss;
        offsetsLen_ = offsetsLen;                       // S (spec formula SN)
        gsIdsLen_ = gsIdsLen;                           // G (spec formula gsIds.shape[0])
        calPixel_ = MinI64(nPixel_, static_cast<int64_t>(THREAD_NUM));  // = min(nPixel, 1024)
        copyPixel_ = (calPixel_ + 7) & ~static_cast<int64_t>(7);        // 32B-aligned MTE2 element count
        // 32B-aligned element bounds (TTK OOM 插桩判界口径 align32(nbytes)/sizeof(T)):
        // guarded copies never read past these (pixel planes + gsAttr rows).
        boundTpElems_  = (tileNum_ * nPixel_ + 7) & ~static_cast<int64_t>(7);              // [T,P] tensors
        bound2TpElems_ = (2 * tileNum_ * nPixel_ + 7) & ~static_cast<int64_t>(7);         // tileCoords [T,2,P]
        bound3TpElems_ = (3 * tileNum_ * nPixel_ + 7) & ~static_cast<int64_t>(7);         // vColor [3,T,P]
        boundGsElems_  = (nGauss_ * NUM_GS_ATTRIBUTES + 7) & ~static_cast<int64_t>(7);    // gs [N,10]

        vecIdx_ = static_cast<int64_t>(GetBlockIdx()) * static_cast<int64_t>(GetSubBlockNum())
                  + static_cast<int64_t>(GetSubBlockIdx());
        vecNum_ = static_cast<int64_t>(GetBlockNum()) * static_cast<int64_t>(GetSubBlockNum());
        if (vecNum_ <= 0) { vecNum_ = 1; }

        // —— GM binding (fp32 path reuses the "first SetGlobalBuffer + offset-derived" pattern) ——
        vColorRGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vColor));  // R channel first
        vColorGGm_ = vColorRGm_[tileNum_ * nPixel_];                          // G/B derived by element offset
        vColorBGm_ = vColorGGm_[tileNum_ * nPixel_];
        vDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vDepth));   // [1,T,P]
        lastCumsumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lastCumsum));
        errorGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(error));
        gsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gs));           // [N,10]
        tileCoordsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tileCoords));
        coreOffsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(offsets));  // 3-segment schedule
        gsIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(gsIds));
        lastIdsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(lastIds));
        vGsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vGs));        // output (cleared by ZeroVGs)

        // —— UB layout: 184KB single buffer (= 248KB - 64KB DCache, HostTiling.md §4) ——
        UbBuffInit();

        // —— Events: one per HardEvent type, allocated via the entry's TPipe (GetTPipePtr) ——
        auto* tp = GetTPipePtr();
        evtMte2V_ = static_cast<event_t>(tp->AllocEventID<HardEvent::MTE2_V>());  // pixel-plane MTE2 -> V/VF
        evtMte2S_ = static_cast<event_t>(tp->AllocEventID<HardEvent::MTE2_S>());  // batch MTE2 -> Scalar/VF
        evtVS_    = static_cast<event_t>(tp->AllocEventID<HardEvent::V_S>());     // ReduceMax V -> Scalar
        evtSV_    = static_cast<event_t>(tp->AllocEventID<HardEvent::S_V>());     // scalarParams S -> VF
    }

    // -----------------------------------------------------------------------
    // LoopProcess — consume a well-formed schedule permutation.
    // VN = S - 2T locates the schedule independently of physical core count.
    // -----------------------------------------------------------------------
    __aicore__ inline void LoopProcess()
    {
        if (tileNum_ <= 0 || nGauss_ <= 0) { return; }   // empty tensor: single-core spin-out (host BlockDim=1)

        // Geometric three-segment location (spec.yaml math_semantics.formula:
        // VN = max(SN - 2*TN, 0); tile-offsets segment = offsets[VN+T .. VN+2T],
        // zero-filled past S — see Process for the guarded reads).
        const int64_t vGeom = MaxI64(offsetsLen_ - 2 * tileNum_, 0);
        const int64_t tileBase = vGeom + tileNum_;

        if (vGeom == vecNum_) {
            // The producer balanced these bins for this physical core count.
            const int64_t begin = vecIdx_ ? coreOffsetsGm_.GetValue(vecIdx_ - 1) : 0;
            const int64_t end = coreOffsetsGm_.GetValue(vecIdx_);
            if (begin < 0 || end < begin || end > tileNum_) { return; }
            for (int64_t j = begin; j < end; ++j) {
                const int64_t tile = coreOffsetsGm_.GetValue(vGeom + j);
                if (tile >= 0 && tile < tileNum_) { Process(tile, tileBase); }
            }
        } else {
            // Captures may contain 64 bins on a 56-core device. Assigning
            // whole bins modulo 56 doubles work on eight cores. Redistribute
            // the schedule permutation evenly instead; each tile still runs
            // once. For best balancing the producer should use vecNum bins.
            for (int64_t j = vecIdx_; j < tileNum_; j += vecNum_) {
                const int64_t tile = coreOffsetsGm_.GetValue(vGeom + j);
                if (tile >= 0 && tile < tileNum_) { Process(tile, tileBase); }
            }
        }
    }

public:
    // -----------------------------------------------------------------------
    // ZeroVGs — kernel-side cooperative zero of vGs [N,10] (host-pre-zero
    // contract made self-contained; see the file-header VGS ZERO INIT note).
    // Each core zeroes its contiguous slice in 8-float (32B) blocks from a
    // zeroed UB tile (gsAttrBatchUb_ partition reused — it is re-populated by
    // the first CopyInBatch after the post-zero cross-core SyncAll, so the
    // reuse cannot alias live data). The exact-length DMA tail (total % 8 floats) is
    // written by the first core. Ordering: Duplicate(V) -> PipeBarrier ->
    // DataCopy(MTE3) -> PipeBarrier -> caller's SyncAll guarantees all zero
    // stores are globally visible before any asc_atomic_add.
    // -----------------------------------------------------------------------
    __aicore__ inline void ZeroVGs()
    {
        const int64_t total = nGauss_ * NUM_GS_ATTRIBUTES;
        if (total <= 0) { return; }                      // N == 0: nothing to zero
        const int64_t nblk = total / 8;            // 8-float (32B) blocks
        const int64_t b0 = vecIdx_ * nblk / vecNum_;
        const int64_t b1 = (vecIdx_ + 1) * nblk / vecNum_;
        if (b1 > b0) {
            constexpr int64_t zeroTileFloats = KERNEL_GS_PER_BATCH * GS_ATTR_STRIDE;  // 4096 floats = 16KB
            Duplicate(gsAttrBatchUb_, (float)0.0f, zeroTileFloats);
            PipeBarrier<PIPE_ALL>();                     // V write visible to MTE3
            const int64_t firstElem = b0 * 8;
            for (int64_t b = b0; b < b1;) {
                int64_t elems = MinI64((b1 - b) * 8, zeroTileFloats) & ~static_cast<int64_t>(7);
                DataCopy(vGsGm_[firstElem + (b - b0) * 8], gsAttrBatchUb_, elems);
                b += elems / 8;
            }
            PipeBarrier<PIPE_ALL>();                     // MTE3 zero stores complete pre-SyncAll
        }
        if (vecIdx_ == 0 && total % 8 != 0) {
            // A scalar GM store may retain a dirty cache line overlapping the
            // last Gaussian and later overwrite another core's atomic sums.
            // Use an exact-length MTE3 store, not SetValue or rounded-up DMA.
            Duplicate(gsAttrBatchUb_, 0.0f, 8);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams tail{1, static_cast<uint32_t>((total % 8) * sizeof(float)), 0, 0, 0};
            DataCopyPad(vGsGm_[nblk * 8], gsAttrBatchUb_, tail);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    // -----------------------------------------------------------------------
    // UbBuffInit — 184KB single buffer, 12 partitions per DESIGN-BRANCH-0 §4.
    // Partition starts 32B-aligned (float offsets (off+7)&~7); scalarParams
    // 8B-aligned ((off+1)&~1). Bound via GetTPipePtr() to the entry's TPipe
    // instance (hard constraint ①, anti UB bus error 264).
    // -----------------------------------------------------------------------
    __aicore__ inline void UbBuffInit()
    {
        auto* tp = GetTPipePtr();
        tp->InitBuffer(ubBuf_, 184 * 1024);             // fixed 184KB (= 248KB - 64KB DCache)
        LocalTensor<float> ub = ubBuf_.Get<float>();

        int64_t off = 0;
        tileCoordXUb_ = ub[off];   off += THREAD_NUM;   // 4KB: 1024 x fp32
        tileCoordYUb_ = ub[off];   off += THREAD_NUM;   // 4KB
        vColorDepthUb_ = ub[off]; off += 4 * THREAD_NUM; // 16KB: RGBD contiguous
        lastCumsumUb_ = ub[off];  off += THREAD_NUM;    // 4KB
        errorUb_ = ub[off];       off += THREAD_NUM;    // 4KB
        cumsumBufUb_ = ub[off];   off += 4 * THREAD_NUM; // 16KB: RGBD contiguous
        lastIdsUb_ = ub[off].ReinterpretCast<int32_t>(); off += LAST_IDS_UB_FLOATS;  // 4KB: int32 x 1024
        off = (off + 1) & ~static_cast<int64_t>(1);     // 8B alignment for int64 slot array
        scalarParamsUb_ = ub[off].ReinterpretCast<int64_t>(); off += 2 * SCALAR_PARAMS_COUNT;  // 40B: 5 x int64
        off = (off + 7) & ~static_cast<int64_t>(7);     // 32B alignment
        gsAttrBatchUb_ = ub[off]; off += KERNEL_GS_PER_BATCH * GS_ATTR_STRIDE;  // 16KB: 256 x 16 fp32
        gsIdsBatchUb_ = ub[off].ReinterpretCast<int64_t>(); off += KERNEL_GS_PER_BATCH * 2;   // 2KB: 256 x int64
        tmpReduceUb_ = ub[off].ReinterpretCast<int32_t>(); off += LAST_IDS_UB_FLOATS;         // 4KB ReduceMax tmp
        off = (off + 7) & ~static_cast<int64_t>(7);
        maxLastIdsUb_ = ub[off].ReinterpretCast<int32_t>(); off += 8;          // 32B: 8 x int32 result
        // total = off floats (75,872B = 18,968 floats) <= 184KB budget (DESIGN-BRANCH-0 §4)
        // [Task 36 F1/F2] gsIdsBatch partition was KERNEL_GS_PER_BATCH/4 floats (256B/32
        // slots); the full-batch DataCopy/SetValue (<= 256 int64 = 2KB, DESIGN-BRANCH-0 §4
        // table "gsIdsBatchUb | 2KB | 256 x int64") overflowed it by 1,792B into tmpReduce.
        // Fixed to KERNEL_GS_PER_BATCH*2 floats (512 floats = 2KB = 256 int64 slots).
    }

    // -----------------------------------------------------------------------
    // CopyInPixelPlane — 7-way pixel-plane MTE2 copy-in for one tile
    // (DESIGN-BRANCH-0.md §5.2). In-bounds contract: REQUIREMENTS §5.9 sanctions
    // alignment overreads ONLY for gsAttr/gsIds batch loads — pixel-plane copies
    // must stay inside the tensor's 32B-aligned element bound (TTK OOM 插桩
    // dyn 套件按 align32(nbytes) 判界，越界读触发 aivec err286 trap，P<8 非对齐
    // 末 tile 曾全军覆没）。Fast path: aligned DataCopy when the whole aligned
    // window fits; otherwise aligned-main DataCopy + scalar tail (tail reads
    // exactly [mainLen, calPixel) — always in bounds since calPixel <= bound -
    // plane). The stale UB tail beyond calPixel is quarantined by the lastIds
    // -1 invalidation (pixels >= calPixel contribute exact 0).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInPixelPlane(int64_t tileIdx)
    {
        const int64_t plane = tileIdx * nPixel_;                          // pixel-plane base offset ([T,P] isomorphic)
        CopyPlaneGuarded(tileCoordXUb_, tileCoordsGm_, tileIdx * 2 * nPixel_, bound2TpElems_);
        CopyPlaneGuarded(tileCoordYUb_, tileCoordsGm_, tileIdx * 2 * nPixel_ + nPixel_, bound2TpElems_);
        CopyPlaneGuarded(vColorDepthUb_[VCOLOR_R_OFFSET], vColorRGm_, plane, bound3TpElems_);
        CopyPlaneGuarded(vColorDepthUb_[VCOLOR_G_OFFSET], vColorGGm_, plane, bound3TpElems_);
        CopyPlaneGuarded(vColorDepthUb_[VCOLOR_B_OFFSET], vColorBGm_, plane, bound3TpElems_);
        CopyPlaneGuarded(vColorDepthUb_[VCOLOR_VDEPTH_OFFSET], vDepthGm_, plane, boundTpElems_);
        CopyPlaneGuarded(lastCumsumUb_, lastCumsumGm_, plane, boundTpElems_);
        CopyPlaneGuarded(errorUb_, errorGm_, plane, boundTpElems_);
        CopyPlaneGuarded(lastIdsUb_, lastIdsGm_, plane, boundTpElems_);   // int32
        SetFlag<HardEvent::MTE2_V>(evtMte2V_);                          // pixel-plane MTE2 write -> V/VF read
    }

    // Aligned-bound guarded plane copy (template on dtype; see CopyInPixelPlane
    // note). Fast 32B-aligned MTE2 path only for fully-aligned grids
    // (calPixel == copyPixel, i.e. P a multiple of 8 / >= 1024 — the
    // perf-critical real-workload shapes; the whole aligned window then always
    // fits the tensor's 32B-aligned bound). Unaligned grids (P < 8 etc.) use an
    // exact in-bounds scalar copy for EVERY tile: mixing per-tile MTE2 and
    // scalar plane fills on unaligned grids trips the TTK OOM 插桩 per-copy
    // check with intermittent aivec err286 traps (empirically bisected), while
    // the pure-scalar grid path and the pure-fast aligned path are both proven
    // clean under the same instrumentation. Scalar writes are fenced to the
    // V/VF consumer by the S_V sync in Process (lastIds tail invalidation).
    template <typename T>
    __aicore__ inline void CopyPlaneGuarded(LocalTensor<T> ub, GlobalTensor<T> gm,
                                            int64_t plane, int64_t boundElems)
    {
        if (calPixel_ == copyPixel_ && plane + copyPixel_ <= boundElems) {
            DataCopy(ub, gm[plane], copyPixel_);                         // aligned-grid fast path
            return;
        }
        for (int64_t i = 0; i < calPixel_; ++i) {                        // in-bounds scalar copy (exact pixels)
            ub.SetValue(i, gm.GetValue(plane + i));
        }
    }

    // -----------------------------------------------------------------------
    // CopyInBatch — per-batch copy-in: gsIds batch + gsAttr per-row fixed
    // 16-float rows (DESIGN-BRANCH-0.md §5.2, two MTE2_S segments).
    //
    // gsIds consumption follows the spec formula defensive guards exactly:
    //   posC = clip(pos, 0, G-1)          (positions past G clamp to the last entry)
    //   gsrc = gsIds[posC] (G == 0 → 0)   (empty gsIds reads as all-zero ids)
    //   gidx = clip(gsrc, 0, N-1)         (id values clamp into [0, N))
    // The raw position (batchStart + t) stays the walk-test index (gi <=
    // binFinal), matching golden's unclipped-pos walk semantics. Fast path:
    // when the whole aligned batch window lies inside gsIds ([0, G)), the
    // batch is loaded with one DataCopy (real-workload shape; the <=7-slot
    // aligned overread past batchSize is sanctioned by REQUIREMENTS §5.9 and
    // never consumed). Garbage-offsets shapes (batch window reaching past G,
    // or G == 0) take per-element clamped scalar reads — no OOB GM access.
    //
    // gsAttr row load keeps the sanctioned 16-float (64B) fast path while the
    // whole window stays inside gs's 32B-aligned element bound; the boundary
    // rows fall back to aligned-main DataCopy + scalar tail over the 10 valid
    // columns (gidx is value-clamped to [0, N-1] before the row address is
    // formed, so every row source is in bounds).
    // -----------------------------------------------------------------------
    __aicore__ inline void CopyInBatch(int64_t batchStart, int64_t batchSize)
    {
        const int64_t copyLen = MinI64((batchSize + 7) & ~static_cast<int64_t>(7),
                                       static_cast<int64_t>(KERNEL_GS_PER_BATCH));  // 8-int64 aligned tail batch
        // (1) gsIds batch → gsIdsBatchUb (spec guards; see the comment above)
        if (gsIdsLen_ > 0 && batchStart >= 0 && batchStart + copyLen <= gsIdsLen_) {
            // fast path: whole aligned batch window inside gsIds (MTE2 write -> Scalar read)
            DataCopy(gsIdsBatchUb_, gsIdsGm_[batchStart], copyLen);
            SetFlag<HardEvent::MTE2_S>(evtMte2S_);
            WaitFlag<HardEvent::MTE2_S>(evtMte2S_);
        } else {
            // clamped per-element path: posC = clip(pos, 0, G-1); G == 0 → id 0
            for (int64_t t = 0; t < batchSize; ++t) {
                const int64_t pos = batchStart + t;
                const int64_t posC = (gsIdsLen_ > 0)
                    ? MinI64(MaxI64(pos, 0), gsIdsLen_ - 1) : 0;
                const int64_t gsrc = (gsIdsLen_ > 0) ? gsIdsGm_.GetValue(posC) : 0;
                gsIdsBatchUb_.SetValue(t, gsrc);
            }
        }
        // (2) gsAttr rows + gidx value clamp: row t source = gs[clip(gsIdsBatch[t], 0, N-1) * 10],
        //     fixed 16-float (64B) load; trailing Set is consumed by the caller's WaitFlag
        //     before the VF call. The clamped gidx is written back so the VF body's
        //     scatter target (gsIdsBatchUb[t]) is always inside [0, N-1].
        for (int64_t t = 0; t < batchSize; ++t) {
            const int64_t gsrc = gsIdsBatchUb_.GetValue(t);
            const int64_t gaussId = MinI64(MaxI64(gsrc, 0), nGauss_ - 1);
            if (gaussId != gsrc) {
                gsIdsBatchUb_.SetValue(t, gaussId);      // value clamp visible to the VF body
            }
            const int64_t rowBase = gaussId * NUM_GS_ATTRIBUTES;
            if (rowBase + MIN_DATACOPY_LEN <= boundGsElems_) {
                DataCopy(gsAttrBatchUb_[t * GS_ATTR_STRIDE], gsGm_[rowBase], MIN_DATACOPY_LEN);
            } else {
                const int64_t mainLen = (boundGsElems_ - rowBase) & ~static_cast<int64_t>(7); // >= 8
                if (mainLen > 0) {
                    DataCopy(gsAttrBatchUb_[t * GS_ATTR_STRIDE], gsGm_[rowBase], mainLen);
                }
                for (int64_t i = mainLen; i < NUM_GS_ATTRIBUTES; ++i) { // valid cols [mainLen, 10)
                    gsAttrBatchUb_.SetValue(t * GS_ATTR_STRIDE + i, gsGm_.GetValue(rowBase + i));
                }
            }
        }
        SetFlag<HardEvent::MTE2_S>(evtMte2S_);                          // batch data (gsAttr) MTE2 write -> VF read
    }

    // -----------------------------------------------------------------------
    // Process(tileIdx) — per-tile orchestration (DESIGN-BRANCH-0.md §5.3)
    // -----------------------------------------------------------------------
    __aicore__ inline void Process(int64_t tileIdx, int64_t tileBase)
    {
        // —— 1. pixel-plane 7-way MTE2 load ——
        CopyInPixelPlane(tileIdx);

        // —— 2. suffix accumulation buffer zero (Vector Duplicate, 4 channels at once) ——
        Duplicate(cumsumBufUb_, (float)0.0f, 4 * THREAD_NUM);

        // —— 3. tile gaussian range from the tile-offset segment
        //      (spec formula: currT[t] = offsets[VN+T+t] zero-filled past S;
        //       prevT[0] = 0, prevT[t] = currT[t-1]; REQUIREMENTS §5.3.3.
        //       Reads are guarded against the offsets tensor length S so a
        //       short segment (S < VN + 2T) reads as 0, matching golden.) ——
        const int64_t currOffset = (tileBase + tileIdx < offsetsLen_)
            ? coreOffsetsGm_.GetValue(tileBase + tileIdx) : 0;
        const int64_t prevOffset = (tileIdx > 0)
            ? ((tileBase + tileIdx - 1 < offsetsLen_)
                ? coreOffsetsGm_.GetValue(tileBase + tileIdx - 1) : 0)
            : 0;

        // —— 4. empty-tile guard: skip the gaussian loop entirely ——
        if (currOffset <= prevOffset) {
            // Pair the MTE2_V flag set by CopyInPixelPlane before skipping: an
            // un-waited set_flag leaves the MTE2->V event pipeline imbalanced,
            // and empty-tile-heavy schedules (many consecutive guard skips) can
            // backpressure/stall the event delivery (observed as intermittent
            // VECTOR_CORE_TIMEOUT hangs under TTK back-to-back launches; CANN
            // guards this exact Set/Wait count difference for super-kernel
            // chaining). Every SetFlag<MTE2_V> is now paired with a WaitFlag.
            WaitFlag<HardEvent::MTE2_V>(evtMte2V_);
            PipeBarrier<PIPE_ALL>();
            return;
        }

        // —— 5. triple-skip (a): ReduceMax over lastIds to shrink the reverse start ——
        WaitFlag<HardEvent::MTE2_V>(evtMte2V_);          // pixel plane arrived
        // Forward sum/error remain the per-pixel Kahan state. Do not
        // replace the recurrence with a reassociated suffix sum.
        const int64_t launchThreads = THREAD_NUM;
        if (calPixel_ < launchThreads) {
            for (int64_t p = calPixel_; p < launchThreads; ++p) {
                lastIdsUb_.SetValue(p, -1);
            }
            SetFlag<HardEvent::S_V>(evtSV_);
            WaitFlag<HardEvent::S_V>(evtSV_);
        }
        ReduceMax(maxLastIdsUb_, lastIdsUb_, tmpReduceUb_, launchThreads, false);
        SetFlag<HardEvent::V_S>(evtVS_);                 // V write -> Scalar read
        WaitFlag<HardEvent::V_S>(evtVS_);
        const int64_t maxLastIds = static_cast<int64_t>(maxLastIdsUb_.GetValue(0));
        const int64_t loopUpperBound = MinI64(currOffset - 1, maxLastIds);  // drop i > maxLastIds no-contribution gaussians
        const int64_t loopLowerBound = prevOffset;
        if (loopUpperBound < loopLowerBound) {
            PipeBarrier<PIPE_ALL>();
            return;
        }

        // —— 6. reverse batch loop (<= 256 gaussians per batch; R2 axis: batching only
        //        changes the copy granularity, never the in-thread order) ——
        int64_t i = loopUpperBound;
        while (i >= loopLowerBound) {
            const int64_t batchEnd = i;
            const int64_t batchStart = MaxI64(loopLowerBound, i - (KERNEL_GS_PER_BATCH - 1));
            const int64_t batchSize = batchEnd - batchStart + 1;

            CopyInBatch(batchStart, batchSize);
            WaitFlag<HardEvent::MTE2_S>(evtMte2S_);      // batch data ready (gsAttr tail Set consumed here)

            // scalarParamsUb 5 x int64 (Scalar write -> VF read)
            scalarParamsUb_.SetValue(SCALAR_PREV_OFFSET, prevOffset);
            scalarParamsUb_.SetValue(SCALAR_CURR_OFFSET, currOffset);
            scalarParamsUb_.SetValue(SCALAR_BATCH_SIZE, batchSize);
            scalarParamsUb_.SetValue(SCALAR_BATCH_START, batchStart);
            scalarParamsUb_.SetValue(SCALAR_NGAUSS, nGauss_);          // defensive (VJP unused)
            SetFlag<HardEvent::S_V>(evtSV_);
            WaitFlag<HardEvent::S_V>(evtSV_);

            asc_vf_call<BwdSimtCompute<THREAD_NUM>>(dim3{THREAD_NUM, 1, 1},
            (__gm__ float*)gsGm_.GetPhyAddr(),
            (__ubuf__ float*)tileCoordXUb_.GetPhyAddr(),
            (__ubuf__ float*)tileCoordYUb_.GetPhyAddr(),
            (__ubuf__ float*)gsAttrBatchUb_.GetPhyAddr(),
            (__ubuf__ float*)vColorDepthUb_.GetPhyAddr(),
            (__ubuf__ float*)lastCumsumUb_.GetPhyAddr(),
            (__ubuf__ float*)errorUb_.GetPhyAddr(),
            (__ubuf__ float*)cumsumBufUb_.GetPhyAddr(),
            (__ubuf__ int32_t*)lastIdsUb_.GetPhyAddr(),
            (__ubuf__ int64_t*)scalarParamsUb_.GetPhyAddr(),
            (__ubuf__ int64_t*)gsIdsBatchUb_.GetPhyAddr(),
            (__gm__ float*)vGsGm_.GetPhyAddr());
            PipeBarrier<PIPE_ALL>();                     // batch isolation: VF writes -> next batch MTE2 reads
            i = batchStart - 1;
        }
        PipeBarrier<PIPE_ALL>();                         // tile isolation
    }

private:
    // tiling-derived sizes
    int64_t nPixel_ = 0;
    int64_t tileNum_ = 0;
    int64_t nGauss_ = 0;
    int64_t offsetsLen_ = 0;   // S = offsets.shape[0] (spec formula SN; geometric VN source)
    int64_t gsIdsLen_ = 0;     // G = gsIds.shape[0] (spec formula gsIds.shape[0]; posC clamp)
    int64_t calPixel_ = 0;     // min(nPixel, 1024)
    int64_t copyPixel_ = 0;    // (calPixel+7)&~7 — 32B-aligned MTE2 count
    int64_t boundTpElems_ = 0;   // align8(T*P)      — [T,P] tensor aligned bounds
    int64_t bound2TpElems_ = 0;  // align8(2*T*P)    — tileCoords
    int64_t bound3TpElems_ = 0;  // align8(3*T*P)    — vColor
    int64_t boundGsElems_ = 0;   // align8(N*10)     — gs
    int64_t vecIdx_ = 0;
    int64_t vecNum_ = 1;

    // events (typed allocation, Kernel.md §4 contract ①)
    event_t evtMte2V_;
    event_t evtMte2S_;
    event_t evtVS_;
    event_t evtSV_;

    // GM tensors
    GlobalTensor<float> vColorRGm_;
    GlobalTensor<float> vColorGGm_;
    GlobalTensor<float> vColorBGm_;
    GlobalTensor<float> vDepthGm_;
    GlobalTensor<float> lastCumsumGm_;
    GlobalTensor<float> errorGm_;
    GlobalTensor<float> gsGm_;
    GlobalTensor<float> tileCoordsGm_;
    GlobalTensor<int64_t> coreOffsetsGm_;
    GlobalTensor<int64_t> gsIdsGm_;
    GlobalTensor<int32_t> lastIdsGm_;
    GlobalTensor<float> vGsGm_;

    // UB single buffer + partitions (DESIGN-BRANCH-0 §4 table)
    TBuf<TPosition::VECCALC> ubBuf_;
    LocalTensor<float> tileCoordXUb_;
    LocalTensor<float> tileCoordYUb_;
    LocalTensor<float> vColorDepthUb_;
    LocalTensor<float> lastCumsumUb_;
    LocalTensor<float> errorUb_;
    LocalTensor<float> cumsumBufUb_;
    LocalTensor<int32_t> lastIdsUb_;
    LocalTensor<int64_t> scalarParamsUb_;
    LocalTensor<float> gsAttrBatchUb_;
    LocalTensor<int64_t> gsIdsBatchUb_;
    LocalTensor<int32_t> tmpReduceUb_;
    LocalTensor<int32_t> maxLastIdsUb_;
};

} // namespace NsCalcRenderBwdClipLastids

// ===========================================================================
// extern "C" __global__ __aicore__ void calc_render_bwd_clip_last_ids(...)
//
// Kernel entry. Signature per proto.md §3 / Interface.md (9 inputs + 1 output
// + workspace + tiling). Three hard constraints (Kernel.md §1.2):
//   (1) TPipe declared inside the entry body (anti UB bus error 264; UbBuffInit
//       binds to it via GetTPipePtr());
//   (2) SetAtomicNone at entry and exit (writeback atomicity is controlled
//       explicitly by asc_atomic_add inside the VF body);
//   (3) PipeBarrier<PIPE_ALL> before exit (all atomic writebacks complete).
// Empty tensors (T=0 / N=0) return early after the cooperative vGs zero +
// SyncAll prologue — the output is exactly zero (host short-circuits BlockDim=1,
// HostTiling.md §5; the single block's SyncAll is a trivial one-block barrier).
// ===========================================================================
extern "C" __global__ __aicore__ void calc_render_bwd_clip_lastids(
    GM_ADDR vColor, GM_ADDR vDepth, GM_ADDR lastCumsum, GM_ADDR error, GM_ADDR gs,
    GM_ADDR tileCoords, GM_ADDR offsets, GM_ADDR gsIds, GM_ADDR lastIds,
    GM_ADDR vGs,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // AIV (AI Vector) only task type (baseline-consistent entry; no
    // REGISTER_NONE_TILING / GET_TILING_DATA_WITH_STRUCT — see below).
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // Tiling raw offset parsing per TilingData.md §1 / REQUIREMENTS §5.6.1
    // ("kernel 入口仅按偏移 0/1/2 × int64 裸解析前 3 字段，与基线入口解析方式一致"):
    // nPixel/tileNum/nGauss at int64 offsets 0/1/2; offsetsLen/gsIdsLen at
    // int64 offsets 3/4 (byte 24/32 — deliberately INSIDE the first 40 bytes,
    // the baseline tiling layout's proven-reliable region: the aclnn runtime's
    // tiling-buffer delivery was observed to leave bytes past the legacy
    // 40-byte layout intermittently uninitialized/stale, and a garbage
    // offsetsLen/gsIdsLen defeats the gsIds fast-path guard -> intermittent
    // aivec err95 MTE invalid-GM-address faults, Task 30); ubSize/blockDim/
    // useTwoPass are serialized but not consumed.
    // GET_TILING_DATA_WITH_STRUCT is deliberately NOT used: it classifies the
    // kernel as "user-defined tiling struct" for the CANN dyn-compile OOM
    // wrapper, which then reads the tensors_size table at align8(sizeof(host
    // macro tiling class)) instead of the serialized tiling size —
    // mismatching TTK's table offset (align8(serialized size)) and making the
    // instrumented kernel judge in-bounds accesses with garbage sizes
    // (intermittent aivec err286 traps under the TTK OOM 插桩 dyn suite).
    // The baseline-style raw parse keeps both sides on the serialized-size layout.
    auto* tilingPtr = reinterpret_cast<__gm__ uint8_t*>(tiling);
    const int64_t nPixelTiling =
        *(reinterpret_cast<__gm__ int64_t*>(tilingPtr) + 0);
    const int64_t tileNumTiling =
        *(reinterpret_cast<__gm__ int64_t*>(tilingPtr) + 1);
    const int64_t nGaussTiling =
        *(reinterpret_cast<__gm__ int64_t*>(tilingPtr) + 2);
    const int64_t offsetsLenTiling =
        *(reinterpret_cast<__gm__ int64_t*>(tilingPtr) + 3);   // byte offset 24
    const int64_t gsIdsLenTiling =
        *(reinterpret_cast<__gm__ int64_t*>(tilingPtr) + 4);   // byte offset 32

    SetAtomicNone();
    // ⚠️ TPipe must be declared in the entry body (anti UB bus error 264);
    // UbBuffInit binds to this instance via GetTPipePtr().
    TPipe pipe;

    NsCalcRenderBwdClipLastids::CalcRenderBwdClipLastidsKernel op;
    op.Init(vColor, vDepth, lastCumsum, error, gs,
            tileCoords, offsets, gsIds, lastIds,
            nPixelTiling, tileNumTiling, nGaussTiling,
            offsetsLenTiling, gsIdsLenTiling, vGs);

    // —— vGs kernel-side zero init (file-header VGS ZERO INIT note): the spec
    //    output semantics are vGs = 0 + Σ contributions (golden np.zeros +
    //    np.add.at), independent of the caller's buffer initial content. All
    //    cores zero their slice, then the hardware cross-core SyncAll orders
    //    the zero stores before any core's asc_atomic_add. Executed
    //    unconditionally (also for the T=0 / N=0 empty-tensor early return
    //    below) so the output is exactly zero when there is nothing to
    //    accumulate. All launched blocks reach this SyncAll uniformly (no
    //    asymmetric early exit before it — the design's SyncAll prohibition
    //    (Kernel.md §9.1 T2) targets per-tile-loop sync points, not this
    //    symmetric prologue).
    op.ZeroVGs();
    AscendC::SyncAll<true>();

    if (tileNumTiling == 0 || nGaussTiling == 0) {
        // Empty tensor (T=0 / N=0): nothing to accumulate — vGs keeps the
        // zeroed state (spec invariant uncovered_gauss_rows_zero).
        PipeBarrier<PIPE_ALL>();
        return;
    }

    op.LoopProcess();

    (void)workspace;                                    // workspace is always 0 and not consumed (DESIGN-BRANCH-0 §8)

    PipeBarrier<PIPE_ALL>();                            // exit sync: all atomic writebacks complete
    SetAtomicNone();                                    // exit guard
}
