/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gaussian_sort.cpp
 * \brief gaussian sort op kernel — 4-way streaming tree-mergesort
 */

#include "gaussian_sort_common.h"
#include "kernel_operator.h"
using namespace GaussianSortCommon;
using namespace AscendC;

class GaussianSort {
public:
    __aicore__ inline GaussianSort()
    {
    }

    __aicore__ inline void GetTilingData(GaussianSortTilingData tiling_data)
    {
        batchSize_ = tiling_data.batchSize;
        cameraNum_ = tiling_data.cameraNum;
        tileNum_ = tiling_data.tileNum;
        gaussNum_ = tiling_data.gaussNum;
        scheduleNum_ = tiling_data.scheduleNum;
        maxSortNum_ = tiling_data.maxSortNum;
        maxMaskNum_ = tiling_data.maxMaskNum;
    }

    __aicore__ inline void Init(GM_ADDR lb_sched, GM_ADDR gaussian_cnt, GM_ADDR depths, GM_ADDR gs_ids,
                                GM_ADDR sorted_offset, GM_ADDR sorted_gs_ids, GM_ADDR userWorkspace,
                                GaussianSortTilingData tiling_data)
    {
        blockNum_ = GetBlockNum();
        ASSERT(blockNum_ != 0 && "Block Dim can not be Zero!");
        GetTilingData(tiling_data);
        blockIndex_ = GetBlockIdx();

        coreOffsetsGM_.SetGlobalBuffer((__gm__ int64_t *)lb_sched);
        scheduleGM_ = coreOffsetsGM_[blockNum_];
        tileOffsetsGM_ = scheduleGM_[tileNum_];

        tileCntGaussGM_.SetGlobalBuffer((__gm__ int32_t *)gaussian_cnt);
        depthsGM_.SetGlobalBuffer((__gm__ float *)depths);
        gsIdsGM_.SetGlobalBuffer((__gm__ float *)gs_ids);
        sortedOffsetGM_.SetGlobalBuffer((__gm__ int64_t *)sorted_offset);
        sortedGsIdsGM_.SetGlobalBuffer((__gm__ int32_t *)sorted_gs_ids);

        // Ping-pong WS layout: slot 0 = initial sub-lists, slot 1 = merge destination
        int64_t vectorWSOffset = static_cast<int64_t>(maxMaskNum_) * blockIndex_ * MRGSORT_WS_TENSOR_NUM;
        GM_ADDR wsBase = userWorkspace + vectorWSOffset * sizeof(float);
        sortedTmpInWS_.SetGlobalBuffer((__gm__ float *)wsBase); // legacy alias = wsBuf_[0]
        wsBuf_[0].SetGlobalBuffer((__gm__ float *)wsBase);
        GM_ADDR ws1Addr = wsBase + static_cast<int64_t>(maxMaskNum_) * KVFACTOR * sizeof(float);
        wsBuf_[1].SetGlobalBuffer((__gm__ float *)ws1Addr);
    }

    __aicore__ inline void InitSortUB()
    {
        pipe_.Reset();
        pipe_.InitBuffer(inQueueGsIds_, BUFFER_NUM, sortProcessNum_ * sizeof(float));
        pipe_.InitBuffer(inQueueDepths_, BUFFER_NUM, sortProcessNum_ * sizeof(float));
        pipe_.InitBuffer(outQueueSortedGsIds_, BUFFER_NUM, sortProcessNum_ * sizeof(int32_t));
        uint32_t buffSize = sortProcessNum_ * sizeof(float);
        pipe_.InitBuffer(sortGsIdsTmpBuf_, buffSize);
        pipe_.InitBuffer(sortedTmpBuf_, buffSize * KVFACTOR);
        pipe_.InitBuffer(sortTmpBuf_, buffSize * KVFACTOR);
        sortedInLocal_ = sortedTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
    }

    __aicore__ inline void DataCopyIn(uint32_t loopId)
    {
        uint64_t offset = sortOffset_ + loopId * sortNumPerLoop_;
        LocalTensor<float> depthsLocal = inQueueDepths_.AllocTensor<float>();
        LocalTensor<float> gsIdsLocal = inQueueGsIds_.AllocTensor<float>();

        if (loopId == sortLoopNum_ - 1) {
            Duplicate(depthsLocal, MAX_FP32, sortProcessNum_);
        }

        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        if (sortAlignedNum_ > 0 && loopId == sortLoopNum_ - 1) {
            uint32_t blockLen = sortMoveNum_ * sizeof(float);
            DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
            DataCopyPadExtParams<float> depthsPadParams{true, 0, (uint8_t)sortAlignedNum_, MAX_FP32};
            DataCopyPad(depthsLocal, depthsGM_[offset], copyParams, depthsPadParams);
            DataCopyPadExtParams<float> idsPadParams{true, 0, (uint8_t)sortAlignedNum_, 0};
            DataCopyPad(gsIdsLocal, gsIdsGM_[offset], copyParams, idsPadParams);
        } else {
            DataCopy(depthsLocal, depthsGM_[offset], sortMoveNum_);
            DataCopy(gsIdsLocal, gsIdsGM_[offset], sortMoveNum_);
        }
        inQueueDepths_.EnQue<float>(depthsLocal);
        inQueueGsIds_.EnQue<float>(gsIdsLocal);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void CopyOutSortedGsToWS(uint32_t loopId)
    {
        uint32_t sortedOffset = GetSortOffset<float>(loopId * sortNumPerLoop_);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopy(sortedTmpInWS_[sortedOffset], sortedInLocal_, GetSortLen<float>(sortProcessNum_));
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void SortSingle(uint32_t loopId)
    {
        DataCopyIn(loopId);

        LocalTensor<float> depthsLocal = inQueueDepths_.DeQue<float>();
        LocalTensor<float> gsIdsLocal = inQueueGsIds_.DeQue<float>();
        Muls(depthsLocal, depthsLocal, -1.0f, sortProcessNum_);
        LocalTensor<float> depthsLocalTmp = depthsLocal;
        LocalTensor<float> tempTensor = sortTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        Concat(depthsLocalTmp, depthsLocal, tempTensor, sortProcessNum_ / ONE_REPEAT_CONCAT_NUM);

        LocalTensor<int32_t> gsIdsLocalTmp = sortGsIdsTmpBuf_.Get<int32_t>(sortProcessNum_);
        Cast(gsIdsLocalTmp, gsIdsLocal, RoundMode::CAST_TRUNC, sortProcessNum_);
        Sort<float, true>(sortedInLocal_, depthsLocalTmp, gsIdsLocalTmp.ReinterpretCast<uint32_t>(), tempTensor,
                          sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        CopyOutSortedGsToWS(loopId);
    }

    __aicore__ inline void SortTiling()
    {
        uint32_t subSeqSortNumAlign = Align<uint32_t>(maxSortNum_, sizeof(float));
        sortNumPerLoop_ = (subSeqSortNumAlign / ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        sortNumPerLoop_ = sortNumPerLoop_ ? sortNumPerLoop_ : ONE_REPEAT_SORT_NUM;
        sortLoopNum_ = Ceil<uint32_t>(sortTileNum_, sortNumPerLoop_);
        sortTailNum_ = sortTileNum_ % sortNumPerLoop_;
        uint32_t sortTailNumAlign = Align<uint32_t>(sortTailNum_, sizeof(float));
        sortTailSortNum_ = Ceil<uint32_t>(sortTailNumAlign, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        sortAlignedNum_ = sortTailNumAlign - sortTailNum_;
        sortTailSortNum_ = sortTailSortNum_ ? sortTailSortNum_ : sortNumPerLoop_;
        sortTailNum_ = sortTailNum_ ? sortTailNum_ : sortNumPerLoop_;
    }

    __aicore__ inline void SubQueSort()
    {
        sortProcessNum_ = sortNumPerLoop_;
        sortMoveNum_ = sortNumPerLoop_;
        for (uint32_t loopId = 0; loopId < sortLoopNum_; loopId++) {
            if (loopId == sortLoopNum_ - 1) {
                sortProcessNum_ = sortTailSortNum_;
                sortMoveNum_ = sortTailNum_;
            }
            InitSortUB();
            SortSingle(loopId);
        }
    }

    // 4-way streaming tree-mergesort: dispatch on useDB_ for single/double-buffer output
    __aicore__ inline void InitMergeUB()
    {
        pipe_.Reset();
        uint32_t chunkBytes = MERGE_CHUNK * KVFACTOR * sizeof(float);
        for (uint32_t i = 0; i < MERGE_K; i++) {
            pipe_.InitBuffer(mergeInBuf_[i], chunkBytes);
        }
        if (useDB_) {
            pipe_.InitBuffer(outMrgQueue_, 2u, chunkBytes * MERGE_K);
        } else {
            pipe_.InitBuffer(mergeOutBuf_, chunkBytes * MERGE_K);
            mergeOutLocal_ = mergeOutBuf_.Get<float>(GetSortLen<float>(MERGE_CHUNK * MERGE_K));
        }
        pipe_.InitBuffer(extractKvBuf_, chunkBytes);
        pipe_.InitBuffer(extractKeysBuf_, MERGE_CHUNK * sizeof(float));
        pipe_.InitBuffer(outQueueSortedGsIds_, BUFFER_NUM, MERGE_CHUNK * sizeof(int32_t));

        for (uint32_t i = 0; i < MERGE_K; i++) {
            mergeInLocal_[i] = mergeInBuf_[i].Get<float>(GetSortLen<float>(MERGE_CHUNK));
        }
        extractKvLocal_ = extractKvBuf_.Get<float>(GetSortLen<float>(MERGE_CHUNK));
        extractKeysLocal_ = extractKeysBuf_.Get<float>(MERGE_CHUNK);
    }

    // Bulk-copy a single sub-list from wsBuf_[inWS] to wsBuf_[outWS], chunked through UB.
    __aicore__ inline void CopyListWS(uint32_t inWS, uint64_t inOff, uint32_t outWS, uint64_t outOff, uint32_t length)
    {
        uint32_t rem = length;
        uint32_t cur = 0;
        while (rem > 0) {
            uint32_t copyN = (rem > MERGE_CHUNK) ? MERGE_CHUNK : rem;
            uint32_t readN = Ceil<uint32_t>(copyN, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DataCopy(mergeInLocal_[0], wsBuf_[inWS][GetSortOffset<float>(inOff + cur)], GetSortLen<float>(readN));
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams params{1, static_cast<uint32_t>(copyN * KVFACTOR * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsBuf_[outWS][GetSortOffset<float>(outOff + cur)], mergeInLocal_[0], params);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            rem -= copyN;
            cur += copyN;
        }
    }

    __aicore__ inline void RefillIfExhausted(uint32_t inWS, uint64_t baseOff, uint32_t i, uint32_t *inHead,
                                             uint32_t *inAvail, uint32_t *consumed, uint32_t totalLen)
    {
        if (inHead[i] >= inAvail[i] && consumed[i] < totalLen) {
            uint32_t gmRem = totalLen - consumed[i];
            uint32_t loadN = (gmRem > MERGE_CHUNK) ? MERGE_CHUNK : gmRem;
            uint32_t readN = Ceil<uint32_t>(loadN, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DataCopy(mergeInLocal_[i], wsBuf_[inWS][GetSortOffset<float>(baseOff + consumed[i])],
                     GetSortLen<float>(readN));
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            inAvail[i] = loadN;
            inHead[i] = 0;
            consumed[i] += loadN;
        }
    }

    // Merge MERGE_K sorted sub-lists using streaming MrgSort (isExhaustedSuspension=true)
    __aicore__ inline void Merge4Streaming(uint32_t inWS, uint32_t outWS, uint64_t inOffs[MERGE_K],
                                           uint32_t inLens[MERGE_K], uint32_t numLists, uint64_t outOffsetBase)
    {
        if (numLists == 0) {
            return;
        }
        if (numLists == 1) {
            CopyListWS(inWS, inOffs[0], outWS, outOffsetBase, inLens[0]);
            return;
        }

        uint32_t consumed[MERGE_K] = {0, 0, 0, 0};
        uint32_t inHead[MERGE_K] = {0, 0, 0, 0};
        uint32_t inAvail[MERGE_K] = {0, 0, 0, 0};

        // Initial chunk load for each list.
        for (uint32_t i = 0; i < numLists; i++) {
            if (inLens[i] == 0)
                continue;
            uint32_t loadN = (inLens[i] > MERGE_CHUNK) ? MERGE_CHUNK : inLens[i];
            uint32_t readN = Ceil<uint32_t>(loadN, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DataCopy(mergeInLocal_[i], wsBuf_[inWS][GetSortOffset<float>(inOffs[i])], GetSortLen<float>(readN));
            inAvail[i] = loadN;
            consumed[i] = loadN;
            inHead[i] = 0;
        }
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        uint64_t outOff = outOffsetBase;
        bool hasMore = true;

        while (hasMore) {
            // Build validBit and per-slot counts from currently-loaded UB chunks.
            uint16_t validBit = 0;
            uint16_t cntList[MERGE_K] = {0, 0, 0, 0};
            uint32_t aliveCount = 0;
            for (uint32_t i = 0; i < numLists; i++) {
                uint32_t ubAvail = inAvail[i] - inHead[i];
                if (ubAvail > 0) {
                    cntList[i] = static_cast<uint16_t>(ubAvail);
                    validBit |= (1u << i);
                    aliveCount++;
                }
            }
            if (aliveCount == 0) {
                hasMore = false;
                break;
            }

            if (aliveCount == 1) {
                // Bulk-stream remaining data of the single alive list to output
                for (uint32_t i = 0; i < numLists; i++) {
                    uint32_t ubAvail = inAvail[i] - inHead[i];
                    uint32_t totalRem = ubAvail + (inLens[i] - consumed[i]);
                    if (totalRem == 0)
                        continue;
                    uint64_t srcOff = inOffs[i] + (consumed[i] - ubAvail);
                    CopyListWS(inWS, srcOff, outWS, outOff, totalRem);
                    outOff += totalRem;
                    consumed[i] = inLens[i];
                    inHead[i] = inAvail[i];
                    break;
                }
                hasMore = false;
                break;
            }

            // ≥2 alive: streaming MrgSort<true>. Inactive slots point to mergeInLocal_[0]
            // (CannSort convention) with cnt=0 and validBit cleared.
            MrgSortSrcList<float> srcList(
                (validBit & 0x1) ? mergeInLocal_[0][GetSortOffset<float>(inHead[0])] : mergeInLocal_[0],
                (validBit & 0x2) ? mergeInLocal_[1][GetSortOffset<float>(inHead[1])] : mergeInLocal_[0],
                (validBit & 0x4) ? mergeInLocal_[2][GetSortOffset<float>(inHead[2])] : mergeInLocal_[0],
                (validBit & 0x8) ? mergeInLocal_[3][GetSortOffset<float>(inHead[3])] : mergeInLocal_[0]);

            uint32_t sortedNum[MERGE_K] = {0, 0, 0, 0};
            uint32_t produced = 0;
            if (useDB_) {
                // DB path: TQue auto-managed events, 2-slot rotation, V_N+1 || MTE3_N overlap.
                LocalTensor<float> outBuf = outMrgQueue_.AllocTensor<float>();
                MrgSort<float, true>(outBuf, srcList, cntList, sortedNum, validBit, 1);
                for (uint32_t i = 0; i < numLists; i++) {
                    inHead[i] += sortedNum[i];
                    produced += sortedNum[i];
                }
                outMrgQueue_.EnQue<float>(outBuf);
                LocalTensor<float> outDeq = outMrgQueue_.DeQue<float>();
                if (produced > 0) {
                    DataCopyExtParams params{1, static_cast<uint32_t>(produced * KVFACTOR * sizeof(float)), 0, 0, 0};
                    DataCopyPad(wsBuf_[outWS][GetSortOffset<float>(outOff)], outDeq, params);
                    outOff += produced;
                }
                outMrgQueue_.FreeTensor(outDeq);
            } else {
                // Single-buffer path: lower per-call overhead, used for K < THRESHOLD.
                MrgSort<float, true>(mergeOutLocal_, srcList, cntList, sortedNum, validBit, 1);
                for (uint32_t i = 0; i < numLists; i++) {
                    inHead[i] += sortedNum[i];
                    produced += sortedNum[i];
                }
                if (produced > 0) {
                    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                    DataCopyExtParams params{1, static_cast<uint32_t>(produced * KVFACTOR * sizeof(float)), 0, 0, 0};
                    DataCopyPad(wsBuf_[outWS][GetSortOffset<float>(outOff)], mergeOutLocal_, params);
                    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                    outOff += produced;
                }
            }

            // Refill any exhausted UB chunks before next MrgSort call.
            for (uint32_t i = 0; i < numLists; i++) {
                RefillIfExhausted(inWS, inOffs[i], i, inHead, inAvail, consumed, inLens[i]);
            }
        }
    }

    // Read the final globally-sorted kv-pair stream from wsBuf_[wsIdx] and emit IDs to sortedGsIdsGM_.
    __aicore__ inline void ExtractWSToOut(uint32_t wsIdx, uint32_t totalLen)
    {
        uint32_t rem = totalLen;
        uint32_t inOff = 0;
        uint32_t outOff = 0;
        while (rem > 0) {
            uint32_t cnt = (rem > MERGE_CHUNK) ? MERGE_CHUNK : rem;
            uint32_t cntAligned = Ceil<uint32_t>(cnt, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            DataCopy(extractKvLocal_, wsBuf_[wsIdx][GetSortOffset<float>(inOff)], GetSortLen<float>(cntAligned));
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

            LocalTensor<int32_t> idsLocal = outQueueSortedGsIds_.AllocTensor<int32_t>();
            Extract(extractKeysLocal_, idsLocal.ReinterpretCast<uint32_t>(), extractKvLocal_,
                    cntAligned / ONE_REPEAT_SORT_NUM);
            outQueueSortedGsIds_.EnQue<int32_t>(idsLocal);

            LocalTensor<int32_t> outIds = outQueueSortedGsIds_.DeQue<int32_t>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyExtParams outParams{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(sortedGsIdsGM_[sortedOffset_ + outOff], outIds, outParams);
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            outQueueSortedGsIds_.FreeTensor(outIds);

            rem -= cnt;
            inOff += cnt;
            outOff += cnt;
        }
        sortedOffset_ += totalLen;
    }

    // Top-level: iterate merge passes until 1 list remains, then extract.
    __aicore__ inline void TreeMerge()
    {
        // Decide output-staging path based on initial K (sortLoopNum_).
        // K large → DB amortizes TQue overhead and gives V/MTE3 hardware overlap.
        // K small → single-buffer path avoids per-call TQue dispatch overhead.
        useDB_ = (sortLoopNum_ >= MERGE_DB_K_THRESHOLD);
        InitMergeUB();

        uint32_t numLists = sortLoopNum_;
        uint32_t curIn = 0; // SubQueSort wrote into wsBuf_[0]
        uint32_t fullLen = sortNumPerLoop_;
        uint32_t lastLen = (sortLoopNum_ > 0) ? sortTailSortNum_ : 0; // length of last sub-list, padded

        if (numLists == 1) {
            ExtractWSToOut(curIn, sortTileNum_);
            return;
        }

        while (numLists > 1) {
            uint32_t numNew = (numLists + MERGE_K - 1) / MERGE_K;
            uint32_t curOut = 1u - curIn;
            uint32_t newFullLen = fullLen * MERGE_K;
            uint32_t newLastLen = 0;

            for (uint32_t g = 0; g < numNew; g++) {
                uint32_t startList = g * MERGE_K;
                uint32_t cntInGroup = numLists - startList;
                if (cntInGroup > MERGE_K)
                    cntInGroup = MERGE_K;

                uint64_t inOffs[MERGE_K] = {0, 0, 0, 0};
                uint32_t inLens[MERGE_K] = {0, 0, 0, 0};
                uint32_t groupTotal = 0;
                for (uint32_t k = 0; k < cntInGroup; k++) {
                    uint32_t globalIdx = startList + k;
                    inOffs[k] = static_cast<uint64_t>(globalIdx) * fullLen;
                    inLens[k] = (globalIdx == numLists - 1) ? lastLen : fullLen;
                    groupTotal += inLens[k];
                }
                uint64_t outBase = static_cast<uint64_t>(g) * newFullLen;
                if (g == numNew - 1) {
                    newLastLen = groupTotal;
                }
                Merge4Streaming(curIn, curOut, inOffs, inLens, cntInGroup, outBase);
            }

            numLists = numNew;
            fullLen = newFullLen;
            lastLen = newLastLen;
            curIn = curOut;
        }

        // After all passes, wsBuf_[curIn] holds one sorted list of length sum(initial lens) =
        // sortNumPerLoop_ × (sortLoopNum_-1) + sortTailSortNum_. Only the first sortTileNum_
        // entries are real (the rest are MAX_FP32 sentinel padding which sort to the tail).
        ExtractWSToOut(curIn, sortTileNum_);
    }

    __aicore__ inline void DataCopyOut()
    {
        // Legacy path used by SortInUB (single-list-fits-in-UB case).
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        LocalTensor<int32_t> sortedGsIdsLocal = outQueueSortedGsIds_.DeQue<int32_t>();
        uint32_t blockLen = sortMoveNum_ * sizeof(int32_t);
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        DataCopyPad(sortedGsIdsGM_[sortedOffset_], sortedGsIdsLocal, copyParams);
        outQueueSortedGsIds_.FreeTensor(sortedGsIdsLocal);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        sortedOffset_ += sortMoveNum_;
    }

    __aicore__ inline void SortInGM()
    {
        SortTiling();
        SubQueSort();
        TreeMerge();
    }

    __aicore__ inline void SortInUB()
    {
        InitSortUB();
        sortLoopNum_ = 1;
        sortMoveNum_ = sortTileNum_;
        DataCopyIn(0);

        LocalTensor<float> depthsLocal = inQueueDepths_.DeQue<float>();
        LocalTensor<float> gsIdsLocal = inQueueGsIds_.DeQue<float>();
        Muls(depthsLocal, depthsLocal, -1.0f, sortProcessNum_);
        LocalTensor<float> sortedLocal = sortedTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        LocalTensor<float> depthsLocalTmp = depthsLocal;
        LocalTensor<float> tempTensor = sortTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        Concat(depthsLocalTmp, depthsLocal, tempTensor, sortProcessNum_ / ONE_REPEAT_CONCAT_NUM);

        LocalTensor<int32_t> sortedGsIdsLocal = outQueueSortedGsIds_.AllocTensor<int32_t>();
        LocalTensor<int32_t> gsIdsLocalTmp = sortedGsIdsLocal;
        Cast(gsIdsLocalTmp, gsIdsLocal, RoundMode::CAST_TRUNC, sortProcessNum_);
        Sort<float, true>(sortedLocal, depthsLocalTmp, gsIdsLocalTmp.ReinterpretCast<uint32_t>(), tempTensor,
                          sortProcessNum_ / ONE_REPEAT_SORT_NUM);

        Extract(depthsLocal, sortedGsIdsLocal.ReinterpretCast<uint32_t>(), sortedLocal,
                sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        outQueueSortedGsIds_.EnQue<int32_t>(sortedGsIdsLocal);
        inQueueDepths_.FreeTensor(depthsLocal);
        inQueueGsIds_.FreeTensor(gsIdsLocal);
        DataCopyOut();
    }

    __aicore__ inline void TileSort()
    {
        uint32_t tileSortGsNumAlign = Align<uint32_t>(sortTileNum_, sizeof(float));
        sortAlignedNum_ = tileSortGsNumAlign - sortTileNum_;
        sortProcessNum_ = Ceil<uint32_t>(tileSortGsNumAlign, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        if (sortProcessNum_ <= maxSortNum_) {
            SortInUB();
        } else {
            SortInGM();
        }
    }

    __aicore__ inline void LoopProcess()
    {
        for (uint32_t batchIdx = 0; batchIdx < batchSize_; batchIdx++) {
            for (uint32_t cameraIdx = 0; cameraIdx < cameraNum_; cameraIdx++) {
                int64_t baseOffset = batchIdx * cameraNum_ + cameraIdx;
                int64_t baseScheduleOffset = baseOffset * scheduleNum_;
                int64_t startScheduleIdx = 0;
                if (blockIndex_ > 0) {
                    startScheduleIdx = coreOffsetsGM_.GetValue(baseScheduleOffset + blockIndex_ - 1);
                }
                int64_t endScheduleIdx = coreOffsetsGM_.GetValue(baseScheduleOffset + blockIndex_);
                int64_t sortedOffsetIdx = baseOffset;
                int64_t baseSortedOffset = 0;
                if (sortedOffsetIdx > 0) {
                    baseSortedOffset = sortedOffsetGM_.GetValue(sortedOffsetIdx - 1);
                }
                sortedOffset_ = 0;
                sortOffset_ = 0;
                for (int64_t scheduleIdx = startScheduleIdx; scheduleIdx < endScheduleIdx; scheduleIdx++) {
                    int64_t tileIdx = scheduleGM_.GetValue(baseScheduleOffset + scheduleIdx);
                    int64_t tileOffset = baseOffset * tileNum_ + tileIdx;
                    int64_t sortedTileOffset = 0;
                    if (tileIdx > 0) {
                        sortedTileOffset = tileOffsetsGM_.GetValue(baseScheduleOffset + tileIdx - 1);
                    }
                    sortOffset_ = tileOffset * gaussNum_;
                    sortedOffset_ = baseSortedOffset + sortedTileOffset;
                    sortTileNum_ = tileCntGaussGM_.GetValue(tileOffset);
                    if (sortTileNum_ > 0) {
                        TileSort();
                    }
                }
            }
        }
    }

private:
    TPipe pipe_;

    // SubQueSort + SortInUB queues / buffers
    TQue<QuePosition::VECIN, QUEUE_DEPTHS_NUM> inQueueGsIds_, inQueueDepths_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTHS_NUM> outQueueSortedGsIds_;
    TBuf<TPosition::VECCALC> sortTmpBuf_, sortGsIdsTmpBuf_, sortedTmpBuf_;
    LocalTensor<float> sortedInLocal_;

    // Tree-merge UB layout
    TBuf<TPosition::VECCALC> mergeInBuf_[MERGE_K];
    // Two output-staging paths (only one is allocated per tile based on useDB_):
    //   - Single-buffer (small K): TBuf, manual V_MTE3 sync.
    //   - Double-buffer (large K): TQue<VECOUT, 2>, SDK-managed events + 2-slot rotation.
    TBuf<TPosition::VECCALC> mergeOutBuf_;
    TQue<QuePosition::VECOUT, 2u> outMrgQueue_;
    LocalTensor<float> mergeOutLocal_; // valid only when useDB_=false
    bool useDB_ = false;
    // Single-buffered scratch for ExtractWSToOut's kv-pair input (no DB needed there).
    TBuf<TPosition::VECCALC> extractKvBuf_;
    TBuf<TPosition::VECCALC> extractKeysBuf_;
    LocalTensor<float> mergeInLocal_[MERGE_K];
    LocalTensor<float> extractKvLocal_;
    LocalTensor<float> extractKeysLocal_;

    // GM tensors
    GlobalTensor<int64_t> coreOffsetsGM_, scheduleGM_, tileOffsetsGM_, sortedOffsetGM_;
    GlobalTensor<int32_t> tileCntGaussGM_;
    GlobalTensor<float> depthsGM_, gsIdsGM_;
    GlobalTensor<int32_t> sortedGsIdsGM_;
    GlobalTensor<float> sortedTmpInWS_; // alias = wsBuf_[0]
    GlobalTensor<float> wsBuf_[2];      // ping-pong workspaces

    // Tiling state
    uint32_t blockIndex_, blockNum_;
    uint32_t batchSize_, cameraNum_, tileNum_, gaussNum_, scheduleNum_, maxMaskNum_, maxSortNum_;
    uint32_t sortTileNum_, sortAlignedNum_, sortMoveNum_, sortProcessNum_, sortLoopNum_, sortNumPerLoop_,
        sortTailSortNum_, sortTailNum_;
    uint64_t sortOffset_;
    uint64_t sortedOffset_;
};

extern "C" __global__ __aicore__ void gaussian_sort(GM_ADDR lb_sched, GM_ADDR gaussian_cnt, GM_ADDR depths,
                                                    GM_ADDR gs_ids, GM_ADDR sorted_offset, GM_ADDR sorted_gs_ids,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    GaussianSort op;
    op.Init(lb_sched, gaussian_cnt, depths, gs_ids, sorted_offset, sorted_gs_ids, userWorkspace, tiling_data);
    op.LoopProcess();
}
