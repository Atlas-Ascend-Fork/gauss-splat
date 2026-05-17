/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gaussian_sort.cpp
 * \brief gaussian sort op kernel
 */

#include "gaussian_sort_common.h"
#include "kernel_operator.h"
using namespace GaussianSortCommon;
using namespace AscendC;

class GaussianSort {
public:
    __aicore__ inline GaussianSort() {}

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

    // lb_sched [B,C,X]
    // depths, gs_ids [B,C,T,N]
    __aicore__ inline void Init(GM_ADDR lb_sched, GM_ADDR gaussian_cnt, GM_ADDR depths, GM_ADDR gs_ids,
                                GM_ADDR sorted_offset, GM_ADDR sorted_gs_ids, GM_ADDR userWorkspace,
                                GaussianSortTilingData tiling_data)
    {
        blockNum_ = GetBlockNum();
        ASSERT(blockNum_ != 0 && "Block Dim can not be Zero!");
        GetTilingData(tiling_data);
        blockIndex_ = GetBlockIdx();

        coreOffsetsGM_.SetGlobalBuffer((__gm__ int64_t*)lb_sched);
        scheduleGM_ = coreOffsetsGM_[blockNum_];
        tileOffsetsGM_ = scheduleGM_[tileNum_];

        tileCntGaussGM_.SetGlobalBuffer((__gm__ int32_t*)gaussian_cnt);
        depthsGM_.SetGlobalBuffer((__gm__ float*)depths);
        gsIdsGM_.SetGlobalBuffer((__gm__ float*)gs_ids);
        sortedOffsetGM_.SetGlobalBuffer((__gm__ int64_t*)sorted_offset);
        sortedGsIdsGM_.SetGlobalBuffer((__gm__ int32_t*)sorted_gs_ids);

        // workspace
        int64_t vectorWSOffset = maxMaskNum_ * blockIndex_ * MRGSORT_WS_TENSOR_NUM;
        GM_ADDR sortedTmpInWS = userWorkspace + vectorWSOffset * sizeof(float);
        sortedTmpInWS_.SetGlobalBuffer((__gm__ float*)sortedTmpInWS);
    }

    __aicore__ inline void InitSortUB()
    {
        // 8N
        // resetUB
        pipe_.Reset();
        // input
        pipe_.InitBuffer(inQueueGsIds_, BUFFER_NUM, sortProcessNum_ * sizeof(float));   // N
        pipe_.InitBuffer(inQueueDepths_, BUFFER_NUM, sortProcessNum_ * sizeof(float));  // N
        // output
        pipe_.InitBuffer(outQueueSortedGsIds_, BUFFER_NUM, sortProcessNum_ * sizeof(int32_t));  // N
        // tmp
        uint32_t buffSize = sortProcessNum_ * sizeof(float);
        pipe_.InitBuffer(sortGsIdsTmpBuf_, buffSize);          // N
        pipe_.InitBuffer(sortedTmpBuf_, buffSize * KVFACTOR);  // 2N
        pipe_.InitBuffer(sortTmpBuf_, buffSize * KVFACTOR);    // 2N
        // local tensor
        sortedInLocal_ = sortedTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
    }

    __aicore__ inline void InitMrgSortUB()
    {
        // 7N
        // resetUB
        pipe_.Reset();
        uint32_t buffSize = sortProcessNum_ * sizeof(float);
        // out
        pipe_.InitBuffer(outQueueSortedGsIds_, BUFFER_NUM, buffSize);  // 0.5N
        // WS
        pipe_.InitBuffer(wsSortedInBuf_, buffSize * KVFACTOR);                                // N
        pipe_.InitBuffer(wsSortedTargetInBuf_, buffSize * KVFACTOR);                          // N
        pipe_.InitBuffer(wsSortedTargetOutBuf_, buffSize * KVFACTOR * MRGSORT_OUT_MULT_NUM);  // 2N
        pipe_.InitBuffer(wsSortedOutBuf_, buffSize);                                          // 0.5N
        // tmp
        pipe_.InitBuffer(sortTmpBuf_, buffSize * KVFACTOR);  // N
        pipe_.InitBuffer(wsSortedDepthsBuf_, buffSize);      // 0.5N
        // local tensor
        sortedInLocal_ = wsSortedInBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        sortedTargetInLocal_ = wsSortedTargetInBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        sortedTargetOutLocal_ =
            wsSortedTargetOutBuf_.Get<float>(GetSortLen<float>(sortProcessNum_ * MRGSORT_OUT_MULT_NUM));
    }

    __aicore__ inline void CopyInMrgSortGsFromWS(uint32_t targetOffset, uint32_t compareOffset)
    {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopy(sortedTargetInLocal_, sortedTmpInWS_[targetOffset], GetSortLen<float>(sortNumPerLoop_));
        DataCopy(sortedInLocal_, sortedTmpInWS_[compareOffset], GetSortLen<float>(sortProcessNum_));
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
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

    __aicore__ inline void CopyOutMrgSortedGsToWS(uint32_t targetOffset, uint32_t compareOffset)
    {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopy(sortedTmpInWS_[targetOffset], sortedTargetOutLocal_[0], GetSortLen<float>(sortNumPerLoop_));
        uint32_t sortOffset = GetSortOffset<float>(sortNumPerLoop_);
        DataCopy(sortedTmpInWS_[compareOffset], sortedTargetOutLocal_[sortOffset], GetSortLen<float>(sortProcessNum_));
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void DataCopyOut()
    {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        LocalTensor<int32_t> sortedGsIdsLocal = outQueueSortedGsIds_.DeQue<int32_t>();
        uint32_t blockLen = sortMoveNum_ * sizeof(int32_t);
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        DataCopyPad(sortedGsIdsGM_[sortedOffset_], sortedGsIdsLocal, copyParams);
        outQueueSortedGsIds_.FreeTensor(sortedGsIdsLocal);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        // 更新偏移量
        sortedOffset_ += sortMoveNum_;
    }

    __aicore__ inline void SortSingle(uint32_t loopId)
    {
        // copy in
        DataCopyIn(loopId);

        LocalTensor<float> depthsLocal = inQueueDepths_.DeQue<float>();
        LocalTensor<float> gsIdsLocal = inQueueGsIds_.DeQue<float>();
        // sort 仅支持降序，需求为升序
        Muls(depthsLocal, depthsLocal, -1.0f, sortProcessNum_);
        // contact
        LocalTensor<float> depthsLocalTmp = depthsLocal;
        LocalTensor<float> tempTensor = sortTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        Concat(depthsLocalTmp, depthsLocal, tempTensor, sortProcessNum_ / ONE_REPEAT_CONCAT_NUM);

        LocalTensor<int32_t> gsIdsLocalTmp = sortGsIdsTmpBuf_.Get<int32_t>(sortProcessNum_);
        Cast(gsIdsLocalTmp, gsIdsLocal, RoundMode::CAST_TRUNC, sortProcessNum_);
        // sort
        Sort<float, true>(sortedInLocal_, depthsLocalTmp, gsIdsLocalTmp.ReinterpretCast<uint32_t>(), tempTensor,
                          sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        // copy out
        CopyOutSortedGsToWS(loopId);
    }

    __aicore__ inline void MrgSortInUB(uint32_t targetId, uint32_t compareId)
    {
        // copy in
        uint32_t targetOffset = GetSortOffset<float>(targetId * sortNumPerLoop_);
        uint32_t compareOffset = GetSortOffset<float>(compareId * sortNumPerLoop_);
        CopyInMrgSortGsFromWS(targetOffset, compareOffset);
        // MrgSort
        uint16_t validBit = 0b11;
        int32_t repeatTimes = 1;
        const uint16_t elementCountList[4] = {static_cast<uint16_t>(sortNumPerLoop_),
                                              static_cast<uint16_t>(sortProcessNum_), static_cast<uint16_t>(0),
                                              static_cast<uint16_t>(0)};
        uint32_t sortedNum[4];
        MrgSortSrcList sortList =
            MrgSortSrcList(sortedTargetInLocal_, sortedInLocal_, sortedTargetInLocal_, sortedTargetInLocal_);
        MrgSort<float, false>(sortedTargetOutLocal_, sortList, elementCountList, sortedNum, validBit, repeatTimes);
        // copy out
        CopyOutMrgSortedGsToWS(targetOffset, compareOffset);
    }

    __aicore__ inline void MrgSortExtraceInUB(uint32_t offset)
    {
        LocalTensor<float> sortedDepthsLocal = wsSortedDepthsBuf_.Get<float>(sortProcessNum_);
        LocalTensor<float> sortedOutLocal = wsSortedOutBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        LocalTensor<int32_t> sortedGsIdsLocal = outQueueSortedGsIds_.AllocTensor<int32_t>();
        // 前期处理前半部分，最后一次，处理后半部分
        DataCopy(sortedOutLocal, sortedTargetOutLocal_[offset], GetSortLen<float>(sortProcessNum_));
        // extract
        Extract(sortedDepthsLocal, sortedGsIdsLocal.ReinterpretCast<uint32_t>(), sortedOutLocal,
                sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        outQueueSortedGsIds_.EnQue<int32_t>(sortedGsIdsLocal);
    }

    __aicore__ inline void SortTiling()
    {
        uint32_t subSeqSortNum = maxSortNum_ / MRGSORT_OUT_MULT_NUM;
        uint32_t subSeqSortNumAlign = Align<uint32_t>(subSeqSortNum, sizeof(float));
        // 按排序接口补齐，避免UB溢出,采取向下取整
        sortNumPerLoop_ = (subSeqSortNumAlign / ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        // 临界处理
        sortNumPerLoop_ = sortNumPerLoop_ ? sortNumPerLoop_ : ONE_REPEAT_SORT_NUM;
        sortLoopNum_ = Ceil<uint32_t>(sortTileNum_, sortNumPerLoop_);
        // 尾块
        sortTailNum_ = sortTileNum_ % sortNumPerLoop_;
        // 按搬运32B要求补齐
        uint32_t sortTailNumAlign = Align<uint32_t>(sortTailNum_, sizeof(float));
        // 排序接口对齐
        sortTailSortNum_ = Ceil<uint32_t>(sortTailNumAlign, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        sortAlignedNum_ = sortTailNumAlign - sortTailNum_;
        // 尾块处理
        sortTailSortNum_ = sortTailSortNum_ ? sortTailSortNum_ : sortNumPerLoop_;
        sortTailNum_ = sortTailNum_ ? sortTailNum_ : sortNumPerLoop_;
    }

    __aicore__ inline void SubQueSort()
    {
        sortProcessNum_ = sortNumPerLoop_;
        sortMoveNum_ = sortNumPerLoop_;
        for (uint32_t loopId = 0; loopId < sortLoopNum_; loopId++) {
            // 处理尾块
            if (loopId == sortLoopNum_ - 1) {
                sortProcessNum_ = sortTailSortNum_;
                sortMoveNum_ = sortTailNum_;
            }
            InitSortUB();
            SortSingle(loopId);
        }
    }
    __aicore__ inline void BubleMrgSort()
    {
        for (uint32_t i = 0; i < sortLoopNum_; i++) {
            uint32_t moveOutOffset = 0;
            sortProcessNum_ = sortNumPerLoop_;
            for (uint32_t j = i + 1; j < sortLoopNum_; j++) {
                InitMrgSortUB();
                // 处理尾块
                if (j == sortLoopNum_ - 1) {
                    sortProcessNum_ = sortTailSortNum_;
                }
                MrgSortInUB(i, j);
            }
            sortProcessNum_ = sortNumPerLoop_;
            sortMoveNum_ = sortNumPerLoop_;
            // 搬出尾块处理适配
            if (i == sortLoopNum_ - 1) {
                moveOutOffset = GetSortOffset<float>(sortNumPerLoop_);
                sortProcessNum_ = sortTailSortNum_;
                sortMoveNum_ = sortTailNum_;
            }
            // 解析高斯球ID
            MrgSortExtraceInUB(moveOutOffset);
            // 搬出
            DataCopyOut();
        }
    }

    __aicore__ inline void SortInGM()
    {
        // 切分子序列
        SortTiling();
        // 子序列排序
        SubQueSort();
        // 冒泡排序
        BubleMrgSort();
    }

    __aicore__ inline void SortInUB()
    {
        InitSortUB();
        // copy in
        sortLoopNum_ = 1;
        sortMoveNum_ = sortTileNum_;
        DataCopyIn(0);

        LocalTensor<float> depthsLocal = inQueueDepths_.DeQue<float>();
        LocalTensor<float> gsIdsLocal = inQueueGsIds_.DeQue<float>();
        // sort 仅支持降序，需求为升序
        Muls(depthsLocal, depthsLocal, -1.0f, sortProcessNum_);
        LocalTensor<float> sortedLocal = sortedTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        // contact
        LocalTensor<float> depthsLocalTmp = depthsLocal;
        LocalTensor<float> tempTensor = sortTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        Concat(depthsLocalTmp, depthsLocal, tempTensor, sortProcessNum_ / ONE_REPEAT_CONCAT_NUM);

        LocalTensor<int32_t> sortedGsIdsLocal = outQueueSortedGsIds_.AllocTensor<int32_t>();
        LocalTensor<int32_t> gsIdsLocalTmp = sortedGsIdsLocal;
        Cast(gsIdsLocalTmp, gsIdsLocal, RoundMode::CAST_TRUNC, sortProcessNum_);
        // sort
        Sort<float, true>(sortedLocal, depthsLocalTmp, gsIdsLocalTmp.ReinterpretCast<uint32_t>(), tempTensor,
                          sortProcessNum_ / ONE_REPEAT_SORT_NUM);

        LocalTensor<float> sortedDepthsLocal = depthsLocal;
        // extract
        Extract(depthsLocal, sortedGsIdsLocal.ReinterpretCast<uint32_t>(), sortedLocal,
                sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        outQueueSortedGsIds_.EnQue<int32_t>(sortedGsIdsLocal);
        inQueueDepths_.FreeTensor(depthsLocal);
        inQueueGsIds_.FreeTensor(gsIdsLocal);
        // copy out
        DataCopyOut();
    }

    __aicore__ inline void TileSort()
    {
        // 搬运32B对齐
        uint32_t tileSortGsNumAlign = Align<uint32_t>(sortTileNum_, sizeof(float));
        sortAlignedNum_ = tileSortGsNumAlign - sortTileNum_;
        // 排序接口对齐
        sortProcessNum_ = Ceil<uint32_t>(tileSortGsNumAlign, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        if (sortProcessNum_ <= maxSortNum_) {
            SortInUB();
        } else {
            SortInGM();
        }
    }

    __aicore__ inline void LoopProcess()
    {
        // Batch
        for (uint32_t batchIdx = 0; batchIdx < batchSize_; batchIdx++) {
            // Camera
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
                // Tile
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

    // input
    TQue<QuePosition::VECIN, QUEUE_DEPTHS_NUM> inQueueGsIds_, inQueueDepths_;
    // output
    TQue<QuePosition::VECOUT, QUEUE_DEPTHS_NUM> outQueueSortedGsIds_;

    // sort
    TBuf<TPosition::VECCALC> sortTmpBuf_, sortGsIdsTmpBuf_, sortedTmpBuf_;
    // mrgSort
    TBuf<TPosition::VECCALC> wsSortedInBuf_, wsSortedTargetInBuf_, wsSortedTargetOutBuf_, wsSortedOutBuf_,
        wsSortedDepthsBuf_;

    // input
    GlobalTensor<int64_t> coreOffsetsGM_, scheduleGM_, tileOffsetsGM_, sortedOffsetGM_;
    GlobalTensor<int32_t> tileCntGaussGM_;
    GlobalTensor<float> depthsGM_, gsIdsGM_;
    // output
    GlobalTensor<int32_t> sortedGsIdsGM_;
    // workspace
    GlobalTensor<float> sortedTmpInWS_;

    // sort
    LocalTensor<float> sortedInLocal_;
    // mgrSort
    LocalTensor<float> sortedTargetInLocal_, sortedTargetOutLocal_;

    // input
    uint32_t blockIndex_, blockNum_;
    uint32_t batchSize_, cameraNum_, tileNum_, gaussNum_, scheduleNum_, maxMaskNum_, maxSortNum_;
    // sort
    uint32_t sortTileNum_, sortAlignedNum_, sortMoveNum_, sortProcessNum_, sortLoopNum_, sortNumPerLoop_,
        sortTailSortNum_, sortTailNum_;
    uint64_t sortOffset_;
    // output
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
