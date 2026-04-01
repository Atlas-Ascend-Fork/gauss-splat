/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

    __aicore__ inline void Init(GM_ADDR all_in_mask, GM_ADDR tile_sums, GM_ADDR tile_offsets, GM_ADDR depths,
                                GM_ADDR sorted_gs_ids, GM_ADDR userWorkspace, GaussianSortTilingData tiling_data)
    {
        blockIndex_ = GetBlockIdx();
        if (blockIndex_ < tiling_data.formerNum) {
            tileNum_ = tiling_data.formerTileNum;
            tileNumOffset_ = blockIndex_ * tiling_data.formerTileNum;
        } else {
            tileNum_ = tiling_data.tailTileNum;
            tileNumOffset_ = tiling_data.formerNum * (tiling_data.formerTileNum - tiling_data.tailTileNum) +
                             blockIndex_ * tiling_data.tailTileNum;
        }

        maskLoopNum_ = tiling_data.maskLoopNum;
        maskNumPerLoop_ = tiling_data.maskNumPerLoop;
        maskTailNum_ = tiling_data.maskTailNum;
        maskAlignedNum_ = tiling_data.maskAlignedNum;
        maxSortNum_ = tiling_data.maxSortNum;
        nGauss_ = tiling_data.nGauss;

        tileSumsGM_.SetGlobalBuffer((__gm__ int32_t*)tile_sums + tileNumOffset_, tileNum_);
        tileOffsetsGM_.SetGlobalBuffer((__gm__ int32_t*)tile_offsets + tileNumOffset_, tileNum_);

        // vector内部计算动态shape下 多tile场景内存分配场景，取Tile最大值。 待优化--计算前置在Host侧
        uint32_t gsTileMaxNum = 0;
        uint32_t vectorMaskGsOffset = 0;
        if (blockIndex_ > 0) {
            vectorMaskGsOffset = tileOffsetsGM_.GetValue(-1);
        }
        uint32_t vectorMaskNum = 0;
        for (uint32_t tileId = 0; tileId < tileNum_; tileId++) {
            uint32_t gsTileNum = tileSumsGM_.GetValue(tileId);
            if (gsTileNum > gsTileMaxNum) {
                gsTileMaxNum = gsTileNum;
            }
            vectorMaskNum += gsTileNum;
        }
        int64_t allInMaskOffset = tileNumOffset_ * nGauss_;
        allInMaskGM_.SetGlobalBuffer((__gm__ float*)all_in_mask + allInMaskOffset, tileNum_ * nGauss_);
        depthsGM_.SetGlobalBuffer((__gm__ float*)depths, nGauss_);
        sortedGsIdsGM_.SetGlobalBuffer((__gm__ int32_t*)sorted_gs_ids + vectorMaskGsOffset, vectorMaskNum);

        // workspace
        // nGauss_ >> maskNGauss 待进一步优化，降低内存占用
        int64_t vectorWSOffset = Align<int64_t>(nGauss_, sizeof(float)) * blockIndex_ * MRGSORT_WS_TENSOR_NUM;
        GM_ADDR maskParamInWS = userWorkspace + vectorWSOffset * sizeof(float);
        int64_t maskParamNum = Align<int64_t>(gsTileMaxNum, sizeof(float));
        // maskIds+maskDepths+sorted(id,depths) 占用4份空间， 排布待优化，尝试sorted复用mask空间，降低一半显存占用
        maskDepthsWS_.SetGlobalBuffer((__gm__ float*)maskParamInWS, maskParamNum);
        maskGsIdsWS_.SetGlobalBuffer((__gm__ int32_t*)maskParamInWS + maskParamNum, maskParamNum);
        GM_ADDR sortedParamInWS = maskParamInWS + maskParamNum * sizeof(float) * KVFACTOR;
        int64_t gsSortedNum = Ceil<int64_t>(maskParamNum, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        sortedGsWS_.SetGlobalBuffer((__gm__ float*)sortedParamInWS, GetSortLen<float>(gsSortedNum));
    }

    __aicore__ inline void InitMaskUB()
    {
        // 6N
        // resetUB
        pipe_.Reset();
        // input
        pipe_.InitBuffer(inQueueTileMask_, BUFFER_NUM, maskNumPerLoop_ * sizeof(float));  // N
        pipe_.InitBuffer(inQueueDepths_, BUFFER_NUM, maskNumPerLoop_ * sizeof(float));    // N
        // maskWS
        pipe_.InitBuffer(outWsQueueMaskIds_, BUFFER_NUM, maskNumPerLoop_ * sizeof(int32_t));   // N
        pipe_.InitBuffer(outWsQueueMaskDepths_, BUFFER_NUM, maskNumPerLoop_ * sizeof(float));  // N
        // tmp
        pipe_.InitBuffer(indexTmpBuf_, maskNumPerLoop_ * sizeof(int32_t));                        // N
        pipe_.InitBuffer(tileMaskIntTmpBuf_, maskNumPerLoop_ * sizeof(uint8_t) / UINT8_BIT_NUM);  // 0.5N
        // local tensor
        indexLocal_ = indexTmpBuf_.Get<int32_t>();
        tileMaskLocalInt_ = tileMaskIntTmpBuf_.Get<uint8_t>();
    }

    __aicore__ inline void InitSortUB()
    {
        // 6N
        // resetUB
        pipe_.Reset();
        // WS
        pipe_.InitBuffer(wsGsIdsInBuf_, sortProcessNum_ * sizeof(int32_t));  // N
        pipe_.InitBuffer(wsDepthsInBuf_, sortProcessNum_ * sizeof(float));   // N
        // tmp
        pipe_.InitBuffer(sortedTmpBuf_, sortProcessNum_ * sizeof(float) * KVFACTOR);  // 2N
        pipe_.InitBuffer(sortTmpBuf_, sortProcessNum_ * sizeof(float) * KVFACTOR);    // 2N
        // local tensor
        tileMaskDepthsLocal_ = wsDepthsInBuf_.Get<float>();
        tileMaskGsIdsLocal_ = wsGsIdsInBuf_.Get<int32_t>();
        sortedInLocal_ = sortedTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
    }

    __aicore__ inline void InitSortUBOnOnce()
    {
        // 7N
        // resetUB
        pipe_.Reset();
        // out
        pipe_.InitBuffer(outQueueSortedGsIds_, BUFFER_NUM, sortProcessNum_ * sizeof(int32_t));  // N
        // WS
        pipe_.InitBuffer(wsGsIdsInBuf_, sortProcessNum_ * sizeof(int32_t));  // N
        pipe_.InitBuffer(wsDepthsInBuf_, sortProcessNum_ * sizeof(float));   // N
        // tmp
        uint32_t buffSize = sortProcessNum_ * sizeof(float) * KVFACTOR;
        pipe_.InitBuffer(sortTmpBuf_, buffSize);    // 2N
        pipe_.InitBuffer(sortedTmpBuf_, buffSize);  // 2N
        // local tensor
        tileMaskDepthsLocal_ = wsDepthsInBuf_.Get<float>();
        tileMaskGsIdsLocal_ = wsGsIdsInBuf_.Get<int32_t>();
    }

    __aicore__ inline void InitMrgSortUB()
    {
        // 8N
        // resetUB
        pipe_.Reset();
        // out
        pipe_.InitBuffer(outQueueSortedGsIds_, BUFFER_NUM, sortProcessNum_ * sizeof(int32_t));  // N
        // WS
        uint32_t buffSize = sortProcessNum_ * sizeof(float) * KVFACTOR;
        pipe_.InitBuffer(wsSortedInBuf_, buffSize);                                // N
        pipe_.InitBuffer(wsSortedTargetInBuf_, buffSize);                          // N
        pipe_.InitBuffer(wsSortedTargetOutBuf_, buffSize * MRGSORT_OUT_MULT_NUM);  // 2N
        pipe_.InitBuffer(wsSortedOutBuf_, buffSize);                               // N
        // tmp
        pipe_.InitBuffer(sortTmpBuf_, buffSize);                                // N
        pipe_.InitBuffer(wsSortedDepthsBuf_, sortProcessNum_ * sizeof(float));  // 0.5N
        // local tensor
        sortedInLocal_ = wsSortedInBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        sortedTargetInLocal_ = wsSortedTargetInBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        sortedTargetOutLocal_ =
            wsSortedTargetOutBuf_.Get<float>(GetSortLen<float>(sortProcessNum_ * MRGSORT_OUT_MULT_NUM));
    }

    __aicore__ inline void CopyInGatherMask(uint32_t offset, uint32_t loopId)
    {
        LocalTensor<float> depthsLocal = inQueueDepths_.AllocTensor<float>();
        LocalTensor<float> tileMaskLocal = inQueueTileMask_.AllocTensor<float>();
        Duplicate(tileMaskLocal, 0.0f, maskProcessNum_);
        uint32_t tileGsIdOffset = loopId * maskNumPerLoop_;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        if (maskTailNum_ > 0 && loopId == maskLoopNum_ - 1) {
            uint32_t moveData = maskProcessNum_ - maskAlignedNum_;
            uint32_t blockLen = moveData * sizeof(float);
            // compare 接口要求256B对齐，搬运要求32B对齐
            uint32_t alignedMoveData = Align<uint32_t>(moveData, sizeof(float));
            uint32_t alignedMoveDataLength = alignedMoveData - moveData;
            DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, (uint8_t)alignedMoveDataLength, 0};
            DataCopyPad(depthsLocal, depthsGM_[tileGsIdOffset], copyParams, padParams);
            DataCopyPad(tileMaskLocal, allInMaskGM_[offset + tileGsIdOffset], copyParams, padParams);
        } else {
            DataCopy(depthsLocal, depthsGM_[tileGsIdOffset], maskProcessNum_);
            DataCopy(tileMaskLocal, allInMaskGM_[offset + tileGsIdOffset], maskProcessNum_);
        }
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        inQueueDepths_.EnQue(depthsLocal);
        inQueueTileMask_.EnQue(tileMaskLocal);
    }
    __aicore__ inline void CopyInMrgSortGsFromWS(uint32_t targetOffset, uint32_t compareOffset)
    {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        DataCopy(sortedTargetInLocal_, sortedGsWS_[targetOffset], GetSortLen<float>(sortNumPerLoop_));
        DataCopy(sortedInLocal_, sortedGsWS_[compareOffset], GetSortLen<float>(sortProcessNum_));
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void CopyInSortGsFromWS(uint32_t offset, uint32_t loopId)
    {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        if (sortAlignedNum_ > 0 && loopId == sortLoopNum_ - 1) {
            uint32_t blockLen = sortMoveNum_ * sizeof(float);
            DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, (uint8_t)sortAlignedNum_, MAX_FP32};
            DataCopyPad(tileMaskDepthsLocal_, maskDepthsWS_[offset], copyParams, padParams);
            DataCopyPadExtParams<int32_t> params{true, 0, (uint8_t)sortAlignedNum_, 0};
            DataCopyPad(tileMaskGsIdsLocal_, maskGsIdsWS_[offset], copyParams, params);
        } else {
            DataCopy(tileMaskDepthsLocal_, maskDepthsWS_[offset], sortMoveNum_);
            DataCopy(tileMaskGsIdsLocal_, maskGsIdsWS_[offset], sortMoveNum_);
        }
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    }

    __aicore__ inline void CopyOutMaskToWS()
    {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        LocalTensor<int32_t> maskIdsLocal = outWsQueueMaskIds_.DeQue<int32_t>();
        LocalTensor<float> maskDepthsLocal = outWsQueueMaskDepths_.DeQue<float>();

        uint32_t blockLen = maskNum_ * sizeof(float);
        DataCopyExtParams copyParams{1, blockLen, 0, 0, 0};
        DataCopyPad(maskGsIdsWS_[maskOffset_], maskIdsLocal, copyParams);
        DataCopyPad(maskDepthsWS_[maskOffset_], maskDepthsLocal, copyParams);

        outWsQueueMaskIds_.FreeTensor(maskIdsLocal);
        outWsQueueMaskDepths_.FreeTensor(maskDepthsLocal);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        // 更新偏移量
        maskOffset_ += maskNum_;
    }

    __aicore__ inline void CopyOutSortedGsToWS(uint32_t offset)
    {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopy(sortedGsWS_[offset], sortedInLocal_, GetSortLen<float>(sortProcessNum_));
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void CopyOutMrgSortedGsToWS(uint32_t targetOffset, uint32_t compareOffset)
    {
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        DataCopy(sortedGsWS_[targetOffset], sortedTargetOutLocal_[0], GetSortLen<float>(sortNumPerLoop_));
        uint32_t sortOffset = GetSortOffset<float>(sortNumPerLoop_);
        DataCopy(sortedGsWS_[compareOffset], sortedTargetOutLocal_[sortOffset], GetSortLen<float>(sortProcessNum_));
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    __aicore__ inline void CopyOutSortedGsToGM()
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

    __aicore__ inline void SortInUB(uint32_t loopId)
    {
        uint32_t offset = loopId * sortNumPerLoop_;
        uint32_t sortedOffset = GetSortOffset<float>(offset);
        Duplicate(tileMaskDepthsLocal_, MAX_FP32, sortProcessNum_);
        // copy in
        CopyInSortGsFromWS(offset, loopId);
        // sort 仅支持降序，需求为升序
        Muls(tileMaskDepthsLocal_, tileMaskDepthsLocal_, -1.0f, sortProcessNum_);
        // contact
        LocalTensor<float> maskDepthsLocalTmp = tileMaskDepthsLocal_;
        LocalTensor<float> tempTensor = sortTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        Concat(maskDepthsLocalTmp, tileMaskDepthsLocal_, tempTensor, sortProcessNum_ / ONE_REPEAT_CONCAT_NUM);
        // sort
        Sort<float, true>(sortedInLocal_, maskDepthsLocalTmp, tileMaskGsIdsLocal_.ReinterpretCast<uint32_t>(),
                          tempTensor, sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        // copy out
        CopyOutSortedGsToWS(sortedOffset);
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

    __aicore__ inline void SortInGMTiling()
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
            SortInUB(loopId);
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
            CopyOutSortedGsToGM();
        }
    }

    __aicore__ inline void SortInGM()
    {
        // 切分子序列
        SortInGMTiling();
        // 子序列排序
        SubQueSort();
        // 冒泡排序
        BubleMrgSort();
    }

    __aicore__ inline void SortOnOnce()
    {
        InitSortUBOnOnce();
        Duplicate(tileMaskDepthsLocal_, MAX_FP32, sortProcessNum_);
        // copy in
        uint32_t sortedGsOffsetOnTile = 0;
        uint32_t loopId = 0;
        sortLoopNum_ = 1;
        sortMoveNum_ = sortTileNum_;
        CopyInSortGsFromWS(sortedGsOffsetOnTile, loopId);
        // sort 仅支持降序，需求为升序
        Muls(tileMaskDepthsLocal_, tileMaskDepthsLocal_, -1.0f, sortProcessNum_);
        LocalTensor<float> sortedLocal = sortedTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        // contact
        LocalTensor<float> maskDepthsLocalTmp = tileMaskDepthsLocal_;
        LocalTensor<float> tempTensor = sortTmpBuf_.Get<float>(GetSortLen<float>(sortProcessNum_));
        Concat(maskDepthsLocalTmp, tileMaskDepthsLocal_, tempTensor, sortProcessNum_ / ONE_REPEAT_CONCAT_NUM);
        // sort
        Sort<float, true>(sortedLocal, maskDepthsLocalTmp, tileMaskGsIdsLocal_.ReinterpretCast<uint32_t>(), tempTensor,
                          sortProcessNum_ / ONE_REPEAT_SORT_NUM);

        LocalTensor<int32_t> maskSortedGsIdsLocal = outQueueSortedGsIds_.AllocTensor<int32_t>();
        LocalTensor<float> maskSortedDepthsLocal = tileMaskDepthsLocal_;
        // extract
        Extract(maskSortedDepthsLocal, maskSortedGsIdsLocal.ReinterpretCast<uint32_t>(), sortedLocal,
                sortProcessNum_ / ONE_REPEAT_SORT_NUM);
        outQueueSortedGsIds_.EnQue<int32_t>(maskSortedGsIdsLocal);
        // copy out
        CopyOutSortedGsToGM();
    }

    __aicore__ inline void TileSort()
    {
        // 初始化排序高斯球数
        sortTileNum_ = maskTileNum_;
        // 搬运32B对齐
        uint32_t tileSortGsNumAlign = Align<uint32_t>(sortTileNum_, sizeof(float));
        sortAlignedNum_ = tileSortGsNumAlign - sortTileNum_;
        // 排序接口对齐
        sortProcessNum_ = Ceil<uint32_t>(tileSortGsNumAlign, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
        if (sortProcessNum_ <= maxSortNum_) {
            SortOnOnce();
        } else {
            SortInGM();
        }
    }

    __aicore__ inline void BuildTileMask()
    {
        LocalTensor<float> tileMaskLocal = inQueueTileMask_.DeQue<float>();
        CompareScalar(tileMaskLocalInt_, tileMaskLocal, float(1), CMPMODE::EQ, maskProcessNum_);
        inQueueTileMask_.FreeTensor(tileMaskLocal);
    }

    __aicore__ inline void GatherGsIdMask(uint32_t loopId)
    {
        LocalTensor<int32_t> maskIdsLocal = outWsQueueMaskIds_.AllocTensor<int32_t>();
        Duplicate(maskIdsLocal, int32_t(0), maskProcessNum_);
        int32_t gsTileOffset = loopId * maskNumPerLoop_;
        Duplicate(indexLocal_, int32_t(0), maskProcessNum_);
        // CreateVecIndex 仅支持 int16_t/half/int32_t/float
        CreateVecIndex(indexLocal_, gsTileOffset, maskProcessNum_);
        // GatherMask
        uint32_t mask = maskProcessNum_;
        uint64_t rsvdMaskGsIdCnt = 0;
        bool reduceMode = true;
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = REPEAT_STRIDE_SIZE;
        params.src1RepeatStride = REPEAT_STRIDE_SIZE;
        GatherMask(maskIdsLocal, indexLocal_, tileMaskLocalInt_.ReinterpretCast<uint32_t>(), reduceMode, mask, params,
                   rsvdMaskGsIdCnt);
        maskNum_ = rsvdMaskGsIdCnt;
        // 若无相交，则不进行后续处理
        if (rsvdMaskGsIdCnt == 0) {
            outWsQueueMaskIds_.FreeTensor(maskIdsLocal);
            return;
        }
        outWsQueueMaskIds_.EnQue(maskIdsLocal);
    }

    __aicore__ inline void GatherDepthsMask()
    {
        LocalTensor<float> depthsLocal = inQueueDepths_.DeQue<float>();
        if (maskNum_ == 0) {
            inQueueDepths_.FreeTensor(depthsLocal);
            return;
        }

        LocalTensor<float> maskDepthsLocal = outWsQueueMaskDepths_.AllocTensor<float>();
        Duplicate(maskDepthsLocal, 0.0f, maskProcessNum_);
        // GatherMask
        uint32_t mask = maskProcessNum_;
        uint64_t rsvdMaskDepthsCnt = 0;
        bool reduceMode = true;
        GatherMaskParams params;
        params.src0BlockStride = 1;
        params.repeatTimes = 1;
        params.src0RepeatStride = REPEAT_STRIDE_SIZE;
        params.src1RepeatStride = REPEAT_STRIDE_SIZE;
        GatherMask(maskDepthsLocal, depthsLocal, tileMaskLocalInt_.ReinterpretCast<uint32_t>(), reduceMode, mask,
                   params, rsvdMaskDepthsCnt);
        outWsQueueMaskDepths_.EnQue(maskDepthsLocal);
        inQueueDepths_.FreeTensor(depthsLocal);
    }

    __aicore__ inline void ProcessGatherMask(uint32_t loopId)
    {
        BuildTileMask();
        GatherGsIdMask(loopId);
        GatherDepthsMask();
    }

    __aicore__ inline void TileMask(uint32_t gsOffset)
    {
        InitMaskUB();
        maskOffset_ = 0;
        maskProcessNum_ = maskNumPerLoop_;
        for (uint32_t loopId = 0; loopId < maskLoopNum_; loopId++) {
            // 处理尾块
            if (loopId == maskLoopNum_ - 1) {
                maskProcessNum_ = maskTailNum_;
            }
            // 未处理完相交数据，则继续处理，跳过后续相交为0的计算
            if (maskOffset_ < maskTileNum_) {
                CopyInGatherMask(gsOffset, loopId);
                ProcessGatherMask(loopId);
                // 跳过相交为0的场景搬运
                if (maskNum_ != 0) {
                    CopyOutMaskToWS();
                }
            }
        }
    }

    __aicore__ inline void LoopProcess()
    {
        // 初始化vector内偏移量
        sortedOffset_ = 0;
        maskTileNum_ = 0;
        for (int32_t tileId = 0; tileId < tileNum_; tileId++) {
            maskTileNum_ = tileSumsGM_.GetValue(tileId);
            uint32_t gsOffset = tileId * nGauss_;
            // 跳过无相交的Tile
            if (maskTileNum_ > 0) {
                TileMask(gsOffset);
                TileSort();
            }
        }
    }

private:
    TPipe pipe_;

    // input
    TQue<QuePosition::VECIN, QUEUE_DEPTHS_NUM> inQueueTileMask_, inQueueDepths_;
    // workspace
    TQue<QuePosition::VECOUT, QUEUE_DEPTHS_NUM> outWsQueueMaskIds_, outWsQueueMaskDepths_;
    // output
    TQue<QuePosition::VECOUT, QUEUE_DEPTHS_NUM> outQueueSortedGsIds_;

    // mask
    TBuf<TPosition::VECCALC> indexTmpBuf_, tileMaskIntTmpBuf_;
    // sort
    TBuf<TPosition::VECCALC> sortTmpBuf_, sortedTmpBuf_;
    TBuf<TPosition::VECCALC> wsGsIdsInBuf_, wsDepthsInBuf_;
    // MrgSort
    TBuf<TPosition::VECCALC> wsSortedInBuf_, wsSortedTargetInBuf_, wsSortedTargetOutBuf_, wsSortedOutBuf_,
        wsSortedDepthsBuf_;

    // input
    GlobalTensor<float> allInMaskGM_, depthsGM_;
    GlobalTensor<int32_t> tileSumsGM_, tileOffsetsGM_;
    // output
    GlobalTensor<int32_t> sortedGsIdsGM_;
    // workspace
    GlobalTensor<float> maskDepthsWS_, sortedGsWS_;
    GlobalTensor<int32_t> maskGsIdsWS_;

    // mask
    LocalTensor<uint8_t> tileMaskLocalInt_;
    LocalTensor<float> tileMaskDepthsLocal_;
    LocalTensor<int32_t> indexLocal_, tileMaskGsIdsLocal_;
    // sort
    LocalTensor<float> sortedInLocal_, sortedTargetInLocal_, sortedTargetOutLocal_;

    uint32_t blockIndex_;
    // tiling
    uint32_t nGauss_;
    uint32_t tileNum_;
    uint32_t tileNumOffset_;
    uint32_t maskLoopNum_;
    uint32_t maskNumPerLoop_;
    uint32_t maskTailNum_;
    uint32_t maskAlignedNum_;
    uint32_t maxSortNum_;
    // mask
    uint32_t maskProcessNum_;
    uint32_t maskNum_;
    uint32_t maskOffset_;
    uint32_t maskTileNum_;
    // sort
    uint32_t sortedOffset_;
    uint32_t sortTileNum_;
    uint32_t sortLoopNum_;
    uint32_t sortNumPerLoop_;
    uint32_t sortTailSortNum_;
    uint32_t sortTailNum_;
    uint32_t sortAlignedNum_;
    uint32_t sortMoveNum_;
    uint32_t sortProcessNum_;
};

extern "C" __global__ __aicore__ void gaussian_sort(GM_ADDR all_in_mask, GM_ADDR tile_sums, GM_ADDR tile_offsets,
                                                    GM_ADDR depths, GM_ADDR sorted_gs_ids, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);

    GaussianSort op;
    op.Init(all_in_mask, tile_sums, tile_offsets, depths, sorted_gs_ids, userWorkspace, tiling_data);
    op.LoopProcess();
}
