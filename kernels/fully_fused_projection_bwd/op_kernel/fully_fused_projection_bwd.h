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

#ifndef FULLY_FUSED_PROJECTION_BWD_H
#define FULLY_FUSED_PROJECTION_BWD_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fully_fused_projection_bwd_common.h"


namespace FullyFusedProjectionBwdNs {
using namespace AscendC;

class FullyFusedProjectionBwd {
public:
    __aicore__ inline FullyFusedProjectionBwd(){};
    __aicore__ inline void Init(GM_ADDR means, GM_ADDR quats, GM_ADDR scales, GM_ADDR conics, GM_ADDR viewmats,
                                GM_ADDR Ks, GM_ADDR vMeans2d, GM_ADDR vDepths, GM_ADDR vConics, GM_ADDR vColorsCulling,
                                GM_ADDR vOpacitiesCulling, GM_ADDR filter, GM_ADDR compensations, GM_ADDR vPW,
                                GM_ADDR vQuats, GM_ADDR vScales, GM_ADDR vR, GM_ADDR vColors, GM_ADDR vOpacities,
                                GM_ADDR workspace, TPipe *Ppipe, const FullyFusedProjectionBwdTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void QuatNormalize(LocalTensor<float> &quats, int element, int64_t offset);
    __aicore__ inline void QuatToRotmat(LocalTensor<float> &quats, LocalTensor<float> &rotmat, int element);
    __aicore__ inline void InverseVjp(LocalTensor<float> &conics, LocalTensor<float> &vConics,
                                      LocalTensor<float> &vCovar2d, int element);
    __aicore__ inline void QuatScaleToCovarPreci(LocalTensor<float> &rotmat, LocalTensor<float> &scales,
                                                 LocalTensor<float> &covars, int element);
    __aicore__ inline void PosW2C(LocalTensor<float> &means, LocalTensor<float> &R, LocalTensor<float> &t,
                                  LocalTensor<float> &meansC, int element);
    __aicore__ inline void PosW2CVjp(LocalTensor<float> &R, LocalTensor<float> &t, LocalTensor<float> &means,
                                     LocalTensor<float> &vMeansC, LocalTensor<float> &vR, int64_t vPWOffset,
                                     int element, int c);
    __aicore__ inline void CovarW2C(LocalTensor<float> &covars, LocalTensor<float> &R, LocalTensor<float> &covarsC,
                                    int element);
    __aicore__ inline void CovarW2CVjp(LocalTensor<float> &R, LocalTensor<float> &covars, LocalTensor<float> &vCovarC,
                                       LocalTensor<float> &vR, LocalTensor<float> &vCovar, int64_t vROffset,
                                       int element);
    __aicore__ inline void QuatScaleToCovarVjp(LocalTensor<float> &quats, LocalTensor<float> &scales,
                                               LocalTensor<float> &rotmat, LocalTensor<float> &vCovar,
                                               LocalTensor<float> &v_quat, LocalTensor<float> &vScales, int element);
    __aicore__ inline void PerspProjVjp(LocalTensor<float> &vCovar2d, LocalTensor<float> &covarsC,
                                        LocalTensor<float> &meansC, LocalTensor<float> &Ks,
                                        LocalTensor<float> &vMeans2d, LocalTensor<float> &vDepths,
                                        LocalTensor<float> &vMeansC, LocalTensor<float> &vCovarC, int element);

    __aicore__ inline void slice(LocalTensor<float> &viewmatsLocal, LocalTensor<float> &R, LocalTensor<float> &t);

    __aicore__ inline void DataCopyNGm2Local(LocalTensor<float> &local, const GlobalTensor<float> &Gm, int32_t element,
                                             int32_t col);

    __aicore__ inline void DataCopyNLocal2Gm(const GlobalTensor<float> &Gm, LocalTensor<float> &local, int32_t element,
                                             int32_t col);

    __aicore__ inline void DataCopyNLocal2GmNoTrans(const GlobalTensor<float> &Gm, LocalTensor<float> &local,
                                                    int32_t element, int32_t col);

    __aicore__ inline void SubProcess(int64_t i, int64_t k, int64_t element);

    __aicore__ inline void CalcCntPerCore(int32_t i);

    __aicore__ inline int32_t CalcCntPerLoop(LocalTensor<uint8_t> &filter, int32_t element);

    __aicore__ inline void CopyInFilter(int64_t i, int64_t j, int64_t k, int32_t element);

    __aicore__ inline int32_t CalcReverseFilterOffset(int32_t element);

    __aicore__ inline void ReverseFilter(LocalTensor<float> &xlocal, const GlobalTensor<float> &xGm, int32_t i,
                                         int32_t j, int32_t element, int32_t col, int64_t offsetNDim,
                                         int32_t cntPerLoop);

    int64_t B_ = 0;
    int64_t C_ = 0;
    int64_t N_ = 0;
    int64_t perCoreN_ = 0;
    int64_t curCoreN_ = 0;
    int64_t perLoopN_ = 0;
    int64_t lastLoopN_ = 0;
    int64_t perLoopMaxN_ = 0;
    int64_t loopN_ = 0;
    int64_t AlignN = 0;
    int64_t blockIdx = 0;
    int64_t needCoreNum = 0;
    int64_t hasComp_ = 0;
    float width_;
    float height_;
    GlobalTensor<float> meansGm_;
    GlobalTensor<float> quatsGm_;
    GlobalTensor<float> scalesGm_;
    GlobalTensor<float> conicsGm_;
    GlobalTensor<float> viewmatsGm_;
    GlobalTensor<float> KsGm_;
    GlobalTensor<float> vMeans2dGm_;
    GlobalTensor<float> vDepthsGm_;
    GlobalTensor<float> vConicsGm_;
    GlobalTensor<float> vPWGm_;
    GlobalTensor<float> vQuatsGm_;
    GlobalTensor<float> vScalesGm_;
    GlobalTensor<float> vRGm_;

    GlobalTensor<float> vColorsCullingGm_;
    GlobalTensor<float> vOpacitiesCullingGm_;
    GlobalTensor<float> compensationsGm_;
    GlobalTensor<float> vColorsGm_;
    GlobalTensor<float> vOpacitiesGm_;

    GlobalTensor<uint8_t> filterGm_;
    GlobalTensor<int32_t> cntPerCoreGm_;
    GlobalTensor<int32_t> sortIndexGm_;

    LocalTensor<int32_t> offsetFilterTensor;
    LocalTensor<uint8_t> filterLocal;
    LocalTensor<float> copyInTensor;
    LocalTensor<int32_t> orderedIndex;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> copyInQue_;

    TBuf<TPosition::VECCALC> intermediateBuf_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> calBuf_;
    TBuf<TPosition::VECCALC> RtBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> offsetFilterBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
};

__aicore__ inline void FullyFusedProjectionBwd::Init(
    GM_ADDR means, GM_ADDR quats, GM_ADDR scales, GM_ADDR conics, GM_ADDR viewmats, GM_ADDR Ks, GM_ADDR vMeans2d,
    GM_ADDR vDepths, GM_ADDR vConics, GM_ADDR vColorsCulling, GM_ADDR vOpacitiesCulling, GM_ADDR filter,
    GM_ADDR compensations, GM_ADDR vPW, GM_ADDR vQuats, GM_ADDR vScales, GM_ADDR vR, GM_ADDR vColors,
    GM_ADDR vOpacities, GM_ADDR workspace, TPipe *Ppipe, const FullyFusedProjectionBwdTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    // 这里先获取tiling值
    B_ = tilingData->batchNum;
    C_ = tilingData->cameraNum;
    N_ = tilingData->gaussNum;
    width_ = static_cast<float>(tilingData->width);
    height_ = static_cast<float>(tilingData->height);
    perCoreN_ = tilingData->blockLength;
    perLoopMaxN_ = tilingData->perloopNum;
    hasComp_ = tilingData->hasCompensations;
    needCoreNum = tilingData->needCoreNum;
    // 本核需要处理多长的N
    curCoreN_ = perCoreN_;
    if (blockIdx == needCoreNum - 1) {
        // curCoreN_ = 尾核
        curCoreN_ = tilingData->lastcoreNum;
    }

    // 计算一次循环处理多长的N，防止超出curCoreN_
    perLoopN_ = Min(perLoopMaxN_, curCoreN_);
    // 计算需要多少个循环处理curCoreN_
    if (perLoopN_ != 0) {
        loopN_ = curCoreN_ / perLoopN_;
    } else {
        return;
    }
    // 计算最后一次循环需要处理多长的N
    lastLoopN_ = curCoreN_ - perLoopN_ * loopN_;

    meansGm_.SetGlobalBuffer((__gm__ float *)means);
    quatsGm_.SetGlobalBuffer((__gm__ float *)quats);
    scalesGm_.SetGlobalBuffer((__gm__ float *)scales);
    conicsGm_.SetGlobalBuffer((__gm__ float *)conics);
    viewmatsGm_.SetGlobalBuffer((__gm__ float *)viewmats);
    KsGm_.SetGlobalBuffer((__gm__ float *)Ks);
    vMeans2dGm_.SetGlobalBuffer((__gm__ float *)vMeans2d);
    vDepthsGm_.SetGlobalBuffer((__gm__ float *)vDepths);
    vConicsGm_.SetGlobalBuffer((__gm__ float *)vConics);
    vPWGm_.SetGlobalBuffer((__gm__ float *)vPW);
    vQuatsGm_.SetGlobalBuffer((__gm__ float *)vQuats);
    vScalesGm_.SetGlobalBuffer((__gm__ float *)vScales);
    vRGm_.SetGlobalBuffer((__gm__ float *)vR);

    vColorsCullingGm_.SetGlobalBuffer((__gm__ float *)vColorsCulling);
    vOpacitiesCullingGm_.SetGlobalBuffer((__gm__ float *)vOpacitiesCulling);
    compensationsGm_.SetGlobalBuffer((__gm__ float *)compensations);
    vColorsGm_.SetGlobalBuffer((__gm__ float *)vColors);
    vOpacitiesGm_.SetGlobalBuffer((__gm__ float *)vOpacities);

    filterGm_.SetGlobalBuffer((__gm__ uint8_t *)(filter));
    cntPerCoreGm_.SetGlobalBuffer((__gm__ int32_t *)(workspace));
    sortIndexGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + WORKSPACE_CNTPERCORE_OFFSET * C_ +
                                 blockIdx * perLoopMaxN_ * WORKSPACE_SORTEDIDX_OFFSET);

    Ppipe->InitBuffer(copyInQue_, 1, COPYINQUE_LEN * AlignBytes(perLoopN_, sizeof(float)));

    Ppipe->InitBuffer(inputBuf_, INPUTBUF_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(intermediateBuf_, INTERMEDIATEBUF_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(calBuf_, CALBUF_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(maskBuf_, MASKBUF_LEN * Align256(perLoopN_, sizeof(uint16_t)));
    Ppipe->InitBuffer(RtBuf_, RTBUF_LEN * sizeof(float));

    Ppipe->InitBuffer(offsetFilterBuf_, AlignBytes(FILTERBUF_CONSTOFFSET * C_, sizeof(uint32_t)));
    Ppipe->InitBuffer(indexBuf_, AlignBytes(perLoopN_, sizeof(uint32_t)));
}

__aicore__ inline int32_t FullyFusedProjectionBwd::CalcReverseFilterOffset(int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    int32_t sortNum = Ceil(element, SORT_ALIGNCONST) * SORT_ALIGNCONST;

    LocalTensor<int32_t> calTensor = calBuf_.Get<int32_t>();
    LocalTensor<int32_t> filterNotIndexTensor = calTensor[offset * FILTERNOTINDEX_OFFSET];
    LocalTensor<float> filterIndex2Tensor = calTensor[offset * FILTERINDEX2_OFFSET].ReinterpretCast<float>();
    LocalTensor<int32_t> filterIndex1Tensor = calTensor[offset * FILTERINDEX1_OFFSET];

    LocalTensor<int32_t> newIndex = copyInTensor.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> headIndexTensor = newIndex;
    LocalTensor<int32_t> tailIndexTensor = newIndex[sortNum];

    ArithProgression(filterIndex1Tensor, 0, 1, element);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(headIndexTensor, PADVALUE_ZERO, element);
    Duplicate<int32_t>(tailIndexTensor, PADVALUE_ZERO, element);

    uint64_t cnt = 0;
    uint64_t cntNot = 0;
    GatherMaskParams params;
    params.src0BlockStride = SRC0BLOCKSTRIDE;
    params.repeatTimes = REPEATTIMES;
    params.src0RepeatStride = SRC0REPEATSTRIDE;
    params.src1RepeatStride = SRC1REPEATSTRIDE;

    LocalTensor<uint32_t> bitMask0 = filterLocal.ReinterpretCast<uint32_t>();
    Not(filterNotIndexTensor.ReinterpretCast<uint16_t>(), filterLocal.ReinterpretCast<uint16_t>(),
        Ceil(element, FILTER_ALIGNCONST) * INT32_TO_INT16);
    LocalTensor<uint32_t> bitMask1 = filterNotIndexTensor.ReinterpretCast<uint32_t>();

    GatherMask(headIndexTensor, filterIndex1Tensor, bitMask0, true, element, params, cnt);
    PipeBarrier<PIPE_V>();
    GatherMask(tailIndexTensor, filterIndex1Tensor, bitMask1, true, element, params, cntNot);
    PipeBarrier<PIPE_V>();

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    DataCopyExtParams headDataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(cnt * sizeof(uint32_t)), 0, 0,
                                         0};
    DataCopyPad(sortIndexGm_, headIndexTensor, headDataCopyParams);
    DataCopyExtParams tailDataCopyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>((element - cnt) * sizeof(uint32_t)), 0, 0, 0};
    DataCopyPad(sortIndexGm_[cnt], tailIndexTensor, tailDataCopyParams);
    copyInQue_.EnQue<int32_t>(newIndex);
    newIndex = copyInQue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(element * sizeof(uint32_t)), 0, 0,
                                     0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(newIndex, sortIndexGm_, dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    copyInQue_.EnQue<int32_t>(newIndex);
    newIndex = copyInQue_.DeQue<int32_t>();
    LocalTensor<float> concatLocal = calTensor[sortNum].ReinterpretCast<float>();
    LocalTensor<int32_t> descIndex = calTensor[sortNum];
    LocalTensor<float> sortedLocal = calTensor[sortNum * SORTEDLOCAL_OFFSET].ReinterpretCast<float>();
    LocalTensor<float> sortTemp = calTensor[sortNum * SORTEDTEMP_OFFSET].ReinterpretCast<float>();
    LocalTensor<float> dstValueLocal = calTensor[sortNum * DSTVALUELOCAL_OFFSET].ReinterpretCast<float>();
    LocalTensor<float> newIndexFp32 = calTensor[sortNum * NEWINDEXFP32_OFFSET].ReinterpretCast<float>();
    LocalTensor<int32_t> localIndex = calTensor[sortNum * LOCALINDEX_OFFSET];
    Duplicate<float>(newIndexFp32, -1.0f, sortNum);
    PipeBarrier<PIPE_V>();
    Cast(newIndexFp32, newIndex, RoundMode::CAST_NONE, element);

    PipeBarrier<PIPE_V>();
    Concat(concatLocal, newIndexFp32, sortTemp, sortNum / CONCAT_AGLIN_VALUE);
    PipeBarrier<PIPE_V>();
    ArithProgression(localIndex, 0, 1, sortNum);

    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedLocal, concatLocal, localIndex.ReinterpretCast<uint32_t>(), sortTemp,
                      sortNum / BLOCK_BYTES);
    PipeBarrier<PIPE_V>();
    Extract(dstValueLocal, descIndex.ReinterpretCast<uint32_t>(), sortedLocal, sortNum / BLOCK_BYTES);
    PipeBarrier<PIPE_V>();

    ArithProgression(dstValueLocal, static_cast<float>(1 - element), 1.0f, element);
    PipeBarrier<PIPE_V>();
    Abs(dstValueLocal, dstValueLocal, element);
    PipeBarrier<PIPE_V>();
    Cast(localIndex, dstValueLocal, RoundMode::CAST_RINT, element);

    PipeBarrier<PIPE_V>();
    Muls(localIndex, localIndex, INT32_SIZE, element);
    PipeBarrier<PIPE_V>();

    Gather<int32_t>(orderedIndex, descIndex, localIndex.ReinterpretCast<uint32_t>(), 0, element);
    PipeBarrier<PIPE_V>();
    Muls(orderedIndex, orderedIndex, INT32_SIZE, element);
    return cnt;
}

__aicore__ inline void FullyFusedProjectionBwd::ReverseFilter(LocalTensor<float> &xlocal,
                                                              const GlobalTensor<float> &xGm, int32_t i,
                                                              int32_t j, int32_t element, int32_t col,
                                                              int64_t offsetNDim, int32_t cntPerLoop)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    int64_t realElement = Align(cntPerLoop, FLOAT_SIZE);
    int64_t offsetGm = (i * C_ * N_ + j * N_) * col + offsetNDim;

    uint32_t len = cntPerLoop * sizeof(float);
    uint32_t stride = (N_ - cntPerLoop) * sizeof(float);

    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(col), static_cast<uint32_t>(len), stride, 0, 0};
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(copyInTensor, xGm[offsetGm], dataCopyParams, dataCopyPadParams);

    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    for (int32_t k = 0; k < col; k++) {
        LocalTensor<float> calTensor = calBuf_.Get<float>();
        LocalTensor<float> tmpCalTensor = calTensor;
        Duplicate<float>(tmpCalTensor, 0.0f, element);
        PipeBarrier<PIPE_V>();
        Add(tmpCalTensor, copyInTensor[realElement * k], tmpCalTensor, cntPerLoop);

        PipeBarrier<PIPE_V>();
        Gather<float>(xlocal[offset * k], tmpCalTensor, orderedIndex.ReinterpretCast<uint32_t>(), 0, element);

        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
}

__aicore__ inline void FullyFusedProjectionBwd::CopyInFilter(int64_t i, int64_t j, int64_t k, int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    filterLocal = intermediateBuf_.Get<uint8_t>()[offset * COPYINFILTER_OFFSET];
    uint32_t len = Ceil(element, FILTER_ALIGNCONST) * sizeof(uint8_t);
    int64_t offsetGm = i * C_ * (Ceil(N_, FILTER_ALIGNCONST)) + j * (Ceil(N_, FILTER_ALIGNCONST)) +
                       perCoreN_ / FILTER_ALIGNCONST * blockIdx + perLoopN_ / FILTER_ALIGNCONST * k;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(copyInTensor.ReinterpretCast<uint8_t>(), filterGm_[offsetGm], dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    Adds(filterLocal.ReinterpretCast<int32_t>(), copyInTensor.ReinterpretCast<int32_t>(), 0,
         perLoopN_ / FILTER_ALIGNCONST / INT32_SIZE + 1);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::CalcCntPerCore(int32_t i)
{
    for (int32_t j = 0; j < C_; ++j) {
        int32_t k = 0;
        int32_t cntPerCore = 0;
        for (; k < loopN_; ++k) {
            LocalTensor<uint8_t> filterLocalIn = copyInTensor.ReinterpretCast<uint8_t>();
            uint32_t len = Ceil(perLoopN_, FILTER_ALIGNCONST) * sizeof(uint8_t);
            int64_t offsetGm = i * C_ * (Ceil(N_, FILTER_ALIGNCONST)) + j * (Ceil(N_, FILTER_ALIGNCONST)) +
                               perCoreN_ / FILTER_ALIGNCONST * blockIdx + perLoopN_ / FILTER_ALIGNCONST * k;
            DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> dataCopyPadParams{false, 0, 0, 0};
            DataCopyPad(filterLocalIn, filterGm_[offsetGm], dataCopyParams, dataCopyPadParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            int32_t cntPerLoop = CalcCntPerLoop(filterLocalIn, perLoopN_);
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            cntPerCore += cntPerLoop;
        }
        if (lastLoopN_ > 0) {
            LocalTensor<uint8_t> filterLocalIn = copyInTensor.ReinterpretCast<uint8_t>();
            uint32_t len = lastLoopN_ / FILTER_ALIGNCONST * sizeof(uint8_t);
            int64_t offsetGm = i * C_ * (Ceil(N_, FILTER_ALIGNCONST)) + j * (Ceil(N_, FILTER_ALIGNCONST)) +
                               perCoreN_ / FILTER_ALIGNCONST * blockIdx + perLoopN_ / FILTER_ALIGNCONST * k;
            DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> dataCopyPadParams{false, 0, 0, 0};
            DataCopyPad(filterLocalIn, filterGm_[offsetGm], dataCopyParams, dataCopyPadParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            int32_t cntPerLoop = CalcCntPerLoop(filterLocalIn, lastLoopN_);
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            cntPerCore += cntPerLoop;
        }

        LocalTensor<int32_t> cntTensor = copyInTensor.ReinterpretCast<int32_t>();
        cntTensor.SetValue(0, cntPerCore);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(cntPerCoreGm_[j * MAXCORENUM + blockIdx], cntTensor, dataCopyParams);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
    SyncAll();

    LocalTensor<int32_t> cntPerCoreTensor = copyInTensor.ReinterpretCast<int32_t>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(MAXCORENUM * sizeof(int32_t)), 0,
                                     0, 0};
    DataCopyExtParams cntDataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};

    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    for (int32_t j = 0; j < C_; ++j) {
        DataCopyPad(cntPerCoreTensor, cntPerCoreGm_[j * MAXCORENUM], dataCopyParams, dataCopyPadParams);
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        int32_t cntPreSumValue = 0;
        for (int32_t k = 0; k < blockIdx; ++k) {
            cntPreSumValue += cntPerCoreTensor.GetValue(k);
        }
        offsetFilterTensor.SetValue(j + C_, cntPreSumValue);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
    }
}

__aicore__ inline int32_t FullyFusedProjectionBwd::CalcCntPerLoop(LocalTensor<uint8_t> &filter, int32_t element)
{
    uint64_t cnt = 0;
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> tmpTensor0 = calBuf_.Get<float>();
    LocalTensor<float> tmpTensor1 = tmpTensor0[offset];

    GatherMaskParams params;
    params.src0BlockStride = SRC0BLOCKSTRIDE;
    params.repeatTimes = REPEATTIMES;
    params.src0RepeatStride = SRC0REPEATSTRIDE;
    params.src1RepeatStride = SRC1REPEATSTRIDE;

    LocalTensor<uint32_t> bitMask = filter.ReinterpretCast<uint32_t>();
    GatherMask(tmpTensor0, tmpTensor1, bitMask, true, element, params, cnt);
    return static_cast<int32_t>(cnt);
}

__aicore__ inline void FullyFusedProjectionBwd::DataCopyNGm2Local(LocalTensor<float> &local,
                                                                  const GlobalTensor<float> &Gm, int32_t element,
                                                                  int32_t col)
{
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    uint32_t len = element * sizeof(float);
    uint32_t stride = (N_ - element) * sizeof(float);
    copyInQue_.EnQue<float>(copyInTensor);
    copyInTensor = copyInQue_.DeQue<float>();
    uint16_t blockNum = col;
    if (col == 0) {
        blockNum = 1;
        stride = 0;
    }
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(blockNum), static_cast<uint32_t>(len), stride, 0, 0};
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(copyInTensor, Gm, dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    copyInQue_.EnQue<float>(copyInTensor);
    copyInTensor = copyInQue_.DeQue<float>();
    int64_t offset = Align(element, FLOAT_SIZE);
    if (col == 0) {
        Adds(local, copyInTensor, 0.0f, element);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(local, copyInTensor, 0.0f, offset * col);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void FullyFusedProjectionBwd::DataCopyNLocal2Gm(const GlobalTensor<float> &Gm,
                                                                  LocalTensor<float> &local, int32_t element,
                                                                  int32_t col)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<uint8_t> transposeTmpBufferLocal = calBuf_.Get<uint8_t>();

    TransposeParamsExt transposeParams;
    transposeParams.nSize = 1;
    transposeParams.cSize = col;
    transposeParams.hSize = offset;
    transposeParams.wSize = 1;
    transposeParams.transposeType = TransposeType::TRANSPOSE_NCHW2NHWC;
    PipeBarrier<PIPE_V>();

    Transpose(copyInTensor, local, transposeTmpBufferLocal, transposeParams);
    PipeBarrier<PIPE_V>();

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(col * element * sizeof(float)), 0,
                                     0, 0};
    DataCopyPad(Gm, copyInTensor, dataCopyParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
}

__aicore__ inline void FullyFusedProjectionBwd::DataCopyNLocal2GmNoTrans(const GlobalTensor<float> &Gm,
                                                                         LocalTensor<float> &local,
                                                                         int32_t element, int32_t col)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    uint32_t stride = (N_ - element) * sizeof(float);
    PipeBarrier<PIPE_V>();
    Adds(copyInTensor, local, 0.0f, offset * col);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(col), static_cast<uint32_t>(element * sizeof(float)), 0,
                                     stride, 0};
    DataCopyPad(Gm, copyInTensor, dataCopyParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

__aicore__ inline void FullyFusedProjectionBwd::slice(LocalTensor<float> &viewmatsLocal, LocalTensor<float> &R,
                                                      LocalTensor<float> &t)
{
    R.SetValue(RMAT_INDEX11, viewmatsLocal.GetValue(VIEWMAT_INDEX11));
    R.SetValue(RMAT_INDEX12, viewmatsLocal.GetValue(VIEWMAT_INDEX12));
    R.SetValue(RMAT_INDEX13, viewmatsLocal.GetValue(VIEWMAT_INDEX13));
    R.SetValue(RMAT_INDEX21, viewmatsLocal.GetValue(VIEWMAT_INDEX21));
    R.SetValue(RMAT_INDEX22, viewmatsLocal.GetValue(VIEWMAT_INDEX22));
    R.SetValue(RMAT_INDEX23, viewmatsLocal.GetValue(VIEWMAT_INDEX23));
    R.SetValue(RMAT_INDEX31, viewmatsLocal.GetValue(VIEWMAT_INDEX31));
    R.SetValue(RMAT_INDEX32, viewmatsLocal.GetValue(VIEWMAT_INDEX32));
    R.SetValue(RMAT_INDEX33, viewmatsLocal.GetValue(VIEWMAT_INDEX33));

    t.SetValue(TMAT_INDEX11, viewmatsLocal.GetValue(VIEWMAT_INDEX14));
    t.SetValue(TMAT_INDEX12, viewmatsLocal.GetValue(VIEWMAT_INDEX24));
    t.SetValue(TMAT_INDEX13, viewmatsLocal.GetValue(VIEWMAT_INDEX34));
}

__aicore__ inline void FullyFusedProjectionBwd::QuatNormalize(LocalTensor<float> &quats, int element,
                                                              int64_t offset)
{
    // R、X、Y、Z分量
    LocalTensor<float> RVec = quats[offset * RVEC_OFFSET];
    LocalTensor<float> XVec = quats[offset * XVEC_OFFSET];
    LocalTensor<float> YVec = quats[offset * YVEC_OFFSET];
    LocalTensor<float> ZVec = quats[offset * ZVEC_OFFSET];

    // 存储临时中间变量
    LocalTensor<float> Share = intermediateBuf_.Get<float>();
    LocalTensor<float> TempBuf = Share[offset * QUANTNORM_TMPBUF_OFFSET];
    LocalTensor<float> TempR2 = TempBuf[offset * QUANTNORM_TMPR2_OFFSET];
    LocalTensor<float> TempX2 = TempBuf[offset * QUANTNORM_TMPX2_OFFSET];
    LocalTensor<float> TempY2 = TempBuf[offset * QUANTNORM_TMPY2_OFFSET];
    LocalTensor<float> TempZ2 = TempBuf[offset * QUANTNORM_TMPZ2_OFFSET];
    LocalTensor<float> TempInvNorm = TempBuf[offset * QUANTNORM_TMPINVNORM_OFFSET];
    LocalTensor<float> RNorm = TempBuf[offset * QUANTNORM_RNORM_OFFSET];
    LocalTensor<float> XNorm = TempBuf[offset * QUANTNORM_XNORM_OFFSET];
    LocalTensor<float> YNorm = TempBuf[offset * QUANTNORM_YNORM_OFFSET];
    LocalTensor<float> ZNorm = TempBuf[offset * QUANTNORM_ZNORM_OFFSET];
    Mul(TempR2, RVec, RVec, element);
    Mul(TempX2, XVec, XVec, element);
    Mul(TempY2, YVec, YVec, element);
    Mul(TempZ2, ZVec, ZVec, element);
    PipeBarrier<PIPE_V>();

    Add(TempInvNorm, TempR2, TempX2, element);
    PipeBarrier<PIPE_V>();
    Add(TempInvNorm, TempInvNorm, TempY2, element);
    PipeBarrier<PIPE_V>();
    Add(TempInvNorm, TempInvNorm, TempZ2, element);
    PipeBarrier<PIPE_V>();

    // 防止除零
    PipeBarrier<PIPE_V>();
    Sqrt(TempInvNorm, TempInvNorm, element);
    Maxs(TempInvNorm, TempInvNorm, AVOID_ZERO, element);
    PipeBarrier<PIPE_V>();

    // 归一化
    Div(RNorm, RVec, TempInvNorm, element);
    Div(XNorm, XVec, TempInvNorm, element);
    Div(YNorm, YVec, TempInvNorm, element);
    Div(ZNorm, ZVec, TempInvNorm, element);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void FullyFusedProjectionBwd::QuatToRotmat(LocalTensor<float> &quats,
                                                             LocalTensor<float> &rotmat, int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    QuatNormalize(quats, element, offset);
    LocalTensor<float> Share = intermediateBuf_.Get<float>();

    LocalTensor<float> RNorm = Share[offset * QUANTOROT_RNORM_OFFSET];
    LocalTensor<float> XNorm = Share[offset * QUANTOROT_XNORM_OFFSET];
    LocalTensor<float> YNorm = Share[offset * QUANTOROT_YNORM_OFFSET];
    LocalTensor<float> ZNorm = Share[offset * QUANTOROT_ZNORM_OFFSET];

    LocalTensor<float> TempBuf = Share[QUANTOROT_TMPBUF_OFFSET * offset];

    // 存储复用的中间乘积项
    LocalTensor<float> TempR2 = TempBuf[offset * QUANTOROT_TMPR2_OFFSET];
    LocalTensor<float> TempX2 = TempBuf[offset * QUANTOROT_TMPX2_OFFSET];
    LocalTensor<float> TempY2 = TempBuf[offset * QUANTOROT_TMPY2_OFFSET];
    LocalTensor<float> TempZ2 = TempBuf[offset * QUANTOROT_TMPZ2_OFFSET];
    LocalTensor<float> TempRx = TempBuf[offset * QUANTOROT_TMPRX_OFFSET];
    LocalTensor<float> TempRy = TempBuf[offset * QUANTOROT_TMPRY_OFFSET];
    LocalTensor<float> TempRz = TempBuf[offset * QUANTOROT_TMPRZ_OFFSET];
    LocalTensor<float> TempXy = TempBuf[offset * QUANTOROT_TMPXY_OFFSET];
    LocalTensor<float> TempXz = TempBuf[offset * QUANTOROT_TMPXZ_OFFSET];
    LocalTensor<float> TempYz = TempBuf[offset * QUANTOROT_TMPYZ_OFFSET];

    // 4个平方项计算
    Mul(TempR2, RNorm, RNorm, element);
    Mul(TempX2, XNorm, XNorm, element);
    Mul(TempY2, YNorm, YNorm, element);
    Mul(TempZ2, ZNorm, ZNorm, element);

    // 6个交叉乘积项计算
    Mul(TempRx, RNorm, XNorm, element);
    Mul(TempRy, RNorm, YNorm, element);
    Mul(TempRz, RNorm, ZNorm, element);
    Mul(TempXy, XNorm, YNorm, element);
    Mul(TempXz, XNorm, ZNorm, element);
    Mul(TempYz, YNorm, ZNorm, element);
    PipeBarrier<PIPE_V>();

    // 计算旋转矩阵的9个元素

    // R00 =1-2*(Y^2+Z^2)
    Add(rotmat, TempY2, TempZ2, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat, rotmat, TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();
    Duplicate<float>(TempBuf[offset * QUANTOROT_R00_OFFSET], ONE_FLOAT_VALUE, element);
    Sub(rotmat, TempBuf[offset * QUANTOROT_R00_OFFSET], rotmat, element);
    PipeBarrier<PIPE_V>();

    // R01=2*(X*Y-R*Z)
    Sub(rotmat[offset * QUANTOROT_R01_OFFSET], TempXy, TempRz, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R01_OFFSET], rotmat[offset * QUANTOROT_R01_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // R02=2*(X*Z+R*Y)
    Add(rotmat[offset * QUANTOROT_R02_OFFSET], TempXz, TempRy, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R02_OFFSET], rotmat[offset * QUANTOROT_R02_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // R10=2*(X*Y+R*Z)
    Add(rotmat[offset * QUANTOROT_R10_OFFSET], TempXy, TempRz, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R10_OFFSET], rotmat[offset * QUANTOROT_R10_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // R11 =1-2*(X^2+Z^2)
    Add(rotmat[offset * QUANTOROT_R11_OFFSET], TempX2, TempZ2, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R11_OFFSET], rotmat[offset * QUANTOROT_R11_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();
    Sub(rotmat[offset * QUANTOROT_R11_OFFSET], TempBuf[offset * QUANTOROT_R00_OFFSET],
        rotmat[offset * QUANTOROT_R11_OFFSET], element);
    PipeBarrier<PIPE_V>();

    // R12=2*(Y*Z-R*X)
    Sub(rotmat[offset * QUANTOROT_R12_OFFSET], TempYz, TempRx, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R12_OFFSET], rotmat[offset * QUANTOROT_R12_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // R20=2*(X*Z-R*Y)
    Sub(rotmat[offset * QUANTOROT_R20_OFFSET], TempXz, TempRy, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R20_OFFSET], rotmat[offset * QUANTOROT_R20_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // R21=2*(X*Y+R*Z)
    Add(rotmat[offset * QUANTOROT_R21_OFFSET], TempYz, TempRx, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R21_OFFSET], rotmat[offset * QUANTOROT_R21_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // R22 =1-2*(X^2+Y^2)
    Add(rotmat[offset * QUANTOROT_R22_OFFSET], TempX2, TempY2, element);
    PipeBarrier<PIPE_V>();
    Muls(rotmat[offset * QUANTOROT_R22_OFFSET], rotmat[offset * QUANTOROT_R22_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();
    Sub(rotmat[offset * QUANTOROT_R22_OFFSET], TempBuf[offset * QUANTOROT_R00_OFFSET],
        rotmat[offset * QUANTOROT_R22_OFFSET], element);

    PipeBarrier<PIPE_V>();
}
__aicore__ inline void FullyFusedProjectionBwd::InverseVjp(LocalTensor<float> &conics,
                                                           LocalTensor<float> &vConics,
                                                           LocalTensor<float> &vCovar2d, int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);

    LocalTensor<float> Share = intermediateBuf_.Get<float>();
    LocalTensor<float> TempBuf = Share[INVJP_TMPBUF_OFFSET * offset];
    // 存储0.5v_C1
    Muls(TempBuf, vConics[offset * C1_OFFSET], HALFVC1_VALUE, element);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> InterCal = Share[INVJP_INTERCAL_OFFSET * offset];
    // S_00
    Mul(vCovar2d, conics[offset * C0_OFFSET], conics[offset * C0_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(vCovar2d, vCovar2d, vConics[offset * C0_OFFSET], element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics[offset * C0_OFFSET], conics[offset * C1_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, TempBuf, element);
    PipeBarrier<PIPE_V>();
    Muls(InterCal, InterCal, TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d, vCovar2d, InterCal, element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics[offset * C1_OFFSET], conics[offset * C1_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, vConics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d, vCovar2d, InterCal, element);
    PipeBarrier<PIPE_V>();
    Muls(vCovar2d, vCovar2d, MINUS_ONE_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // S_01
    Mul(vCovar2d[offset * C1_OFFSET], conics, conics[offset * C1_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(vCovar2d[offset * C1_OFFSET], vCovar2d[offset * C1_OFFSET], vConics, element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics, conics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, TempBuf, element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d[offset * C1_OFFSET], vCovar2d[offset * C1_OFFSET], InterCal, element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics[offset * C1_OFFSET], conics[offset * C1_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, TempBuf, element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d[offset * C1_OFFSET], vCovar2d[offset * C1_OFFSET], InterCal, element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics[offset * C1_OFFSET], conics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, vConics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d[offset * C1_OFFSET], vCovar2d[offset * C1_OFFSET], InterCal, element);
    PipeBarrier<PIPE_V>();
    Muls(vCovar2d[offset * C1_OFFSET], vCovar2d[offset * C1_OFFSET], MINUS_ONE_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // S_11
    Mul(vCovar2d[offset * C2_OFFSET], conics[offset * C1_OFFSET], conics[offset * C1_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(vCovar2d[offset * C2_OFFSET], vCovar2d[offset * C2_OFFSET], vConics, element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics[offset * C1_OFFSET], conics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, TempBuf, element);
    PipeBarrier<PIPE_V>();
    Muls(InterCal, InterCal, INTERCAL_VALUE, element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d[offset * C2_OFFSET], vCovar2d[offset * C2_OFFSET], InterCal, element);
    PipeBarrier<PIPE_V>();

    Mul(InterCal, conics[offset * C2_OFFSET], conics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(InterCal, InterCal, vConics[offset * C2_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(vCovar2d[offset * C2_OFFSET], vCovar2d[offset * C2_OFFSET], InterCal, element);
    PipeBarrier<PIPE_V>();
    Muls(vCovar2d[offset * C2_OFFSET], vCovar2d[offset * C2_OFFSET], MINUS_ONE_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void FullyFusedProjectionBwd::QuatScaleToCovarPreci(LocalTensor<float> &rotmat,
                                                                      LocalTensor<float> &scales,
                                                                      LocalTensor<float> &covars, int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> Share = intermediateBuf_.Get<float>();
    LocalTensor<float> Temp1Buffer = Share[QUANTSCALE_TMP1BUF_OFFSET * offset];
    LocalTensor<float> Temp2Buffer = Share[QUANTSCALE_TMP2BUF_OFFSET * offset];
    Duplicate<float>(Temp1Buffer, PADVALUE_ZERO_FLOAT, QUANTSCALE_TMP1BUF_LEN * offset);
    Duplicate<float>(Temp2Buffer, PADVALUE_ZERO_FLOAT, QUANTSCALE_TMP2BUF_LEN * offset);

    for (int i = 0; i < ROTMAT_DIM; ++i) {
        for (int j = 0; j < ROTMAT_DIM; ++j) {
            PipeBarrier<PIPE_V>();
            Mul(Temp1Buffer[offset * (i * ROTMAT_DIM + j)], rotmat[offset * (i * ROTMAT_DIM + j)], scales[offset * j],
                element);
        }
    }
    PipeBarrier<PIPE_V>();
    Mul(covars[offset * COVMAT_INDEX11], Temp1Buffer, Temp1Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM1_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX12],
        Temp1Buffer[offset * LEFTMAT_INDEX12], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX11], covars[offset * COVMAT_INDEX11], Temp2Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM2_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX13],
        Temp1Buffer[offset * LEFTMAT_INDEX13], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX11], covars[offset * COVMAT_INDEX11], Temp2Buffer[offset * TMP2SUM2_OFFSET],
        element);
    PipeBarrier<PIPE_V>();

    Mul(covars[offset * COVMAT_INDEX12], Temp1Buffer, Temp1Buffer[offset * LEFTMAT_INDEX21], element);
    Mul(Temp2Buffer[offset * TMP2SUM1_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX12],
        Temp1Buffer[offset * LEFTMAT_INDEX22], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX12], covars[offset * COVMAT_INDEX12], Temp2Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM2_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX13],
        Temp1Buffer[offset * LEFTMAT_INDEX23], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX12], covars[offset * COVMAT_INDEX12], Temp2Buffer[offset * TMP2SUM2_OFFSET],
        element);
    PipeBarrier<PIPE_V>();

    Mul(covars[offset * COVMAT_INDEX13], Temp1Buffer, Temp1Buffer[offset * LEFTMAT_INDEX31], element);
    Mul(Temp2Buffer[offset * TMP2SUM1_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX12],
        Temp1Buffer[offset * LEFTMAT_INDEX32], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX13], covars[offset * COVMAT_INDEX13], Temp2Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM2_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX13],
        Temp1Buffer[offset * LEFTMAT_INDEX33], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX13], covars[offset * COVMAT_INDEX13], Temp2Buffer[offset * TMP2SUM2_OFFSET],
        element);
    PipeBarrier<PIPE_V>();

    Mul(covars[offset * COVMAT_INDEX22], Temp1Buffer[offset * LEFTMAT_INDEX21], Temp1Buffer[offset * LEFTMAT_INDEX21],
        element);
    Mul(Temp2Buffer[offset * TMP2SUM1_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX22],
        Temp1Buffer[offset * LEFTMAT_INDEX22], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX22], covars[offset * COVMAT_INDEX22], Temp2Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM2_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX23],
        Temp1Buffer[offset * LEFTMAT_INDEX23], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX22], covars[offset * COVMAT_INDEX22], Temp2Buffer[offset * TMP2SUM2_OFFSET],
        element);
    PipeBarrier<PIPE_V>();

    Mul(covars[offset * COVMAT_INDEX23], Temp1Buffer[offset * LEFTMAT_INDEX21], Temp1Buffer[offset * LEFTMAT_INDEX31],
        element);
    Mul(Temp2Buffer[offset * TMP2SUM1_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX22],
        Temp1Buffer[offset * LEFTMAT_INDEX32], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX23], covars[offset * COVMAT_INDEX23], Temp2Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM2_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX23],
        Temp1Buffer[offset * LEFTMAT_INDEX33], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX23], covars[offset * COVMAT_INDEX23], Temp2Buffer[offset * TMP2SUM2_OFFSET],
        element);
    PipeBarrier<PIPE_V>();

    Mul(covars[offset * COVMAT_INDEX33], Temp1Buffer[offset * LEFTMAT_INDEX31], Temp1Buffer[offset * LEFTMAT_INDEX31],
        element);
    Mul(Temp2Buffer[offset * TMP2SUM1_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX32],
        Temp1Buffer[offset * LEFTMAT_INDEX32], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX33], covars[offset * COVMAT_INDEX33], Temp2Buffer, element);
    Mul(Temp2Buffer[offset * TMP2SUM2_OFFSET], Temp1Buffer[offset * LEFTMAT_INDEX33],
        Temp1Buffer[offset * LEFTMAT_INDEX33], element);
    PipeBarrier<PIPE_V>();
    Add(covars[offset * COVMAT_INDEX33], covars[offset * COVMAT_INDEX33], Temp2Buffer[offset * TMP2SUM2_OFFSET],
        element);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline void FullyFusedProjectionBwd::PosW2C(LocalTensor<float> &means, LocalTensor<float> &R,
                                                       LocalTensor<float> &t, LocalTensor<float> &meansC,
                                                       int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    float a0 = R.GetValue(RMAT_INDEX11);
    float a1 = R.GetValue(RMAT_INDEX12);
    float a2 = R.GetValue(RMAT_INDEX13);
    float a3 = R.GetValue(RMAT_INDEX21);
    float a4 = R.GetValue(RMAT_INDEX22);
    float a5 = R.GetValue(RMAT_INDEX23);
    float a6 = R.GetValue(RMAT_INDEX31);
    float a7 = R.GetValue(RMAT_INDEX32);
    float a8 = R.GetValue(RMAT_INDEX33);
    Muls(meansC[offset * MEANS1_OFFSET], means[offset * MEANS1_OFFSET], a0, element);
    Muls(meansC[offset * MEANS2_OFFSET], means[offset * MEANS1_OFFSET], a3, element);
    Muls(meansC[offset * MEANS3_OFFSET], means[offset * MEANS1_OFFSET], a6, element);
    PipeBarrier<PIPE_V>();

    Axpy(meansC[offset * MEANS1_OFFSET], means[offset * MEANS2_OFFSET], a1, element);
    Axpy(meansC[offset * MEANS2_OFFSET], means[offset * MEANS2_OFFSET], a4, element);
    Axpy(meansC[offset * MEANS3_OFFSET], means[offset * MEANS2_OFFSET], a7, element);
    PipeBarrier<PIPE_V>();

    Axpy(meansC[offset * MEANS1_OFFSET], means[offset * MEANS3_OFFSET], a2, element);
    Axpy(meansC[offset * MEANS2_OFFSET], means[offset * MEANS3_OFFSET], a5, element);
    Axpy(meansC[offset * MEANS3_OFFSET], means[offset * MEANS3_OFFSET], a8, element);
    PipeBarrier<PIPE_V>();

    Adds(meansC[offset * MEANS1_OFFSET], meansC[offset * MEANS1_OFFSET], t.GetValue(MEANS1_OFFSET), element);
    Adds(meansC[offset * MEANS2_OFFSET], meansC[offset * MEANS2_OFFSET], t.GetValue(MEANS2_OFFSET), element);
    Adds(meansC[offset * MEANS3_OFFSET], meansC[offset * MEANS3_OFFSET], t.GetValue(MEANS3_OFFSET), element);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::CovarW2C(LocalTensor<float> &covars, LocalTensor<float> &R,
                                                         LocalTensor<float> &covarsC, int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    float a0 = R.GetValue(RMAT_INDEX11);
    float a1 = R.GetValue(RMAT_INDEX12);
    float a2 = R.GetValue(RMAT_INDEX13);
    float a3 = R.GetValue(RMAT_INDEX21);
    float a4 = R.GetValue(RMAT_INDEX22);
    float a5 = R.GetValue(RMAT_INDEX23);
    float a6 = R.GetValue(RMAT_INDEX31);
    float a7 = R.GetValue(RMAT_INDEX32);
    float a8 = R.GetValue(RMAT_INDEX33);
    // 计算covarsC00 = A0*a0^2 + 2A1a0a1 + 2A2a0a2 + A4a1^2 + 2A5a1a2 + A8a2^2
    // 计算covarsC01 = A0a0a3 + A1(a0a4+a1a3) + A2(a0a5+a2a3) + A4a1a4 +
    // A5(a1a5+a2a4) + A8a2a5 计算covarsC02 = A0a0a6 + A1(a0a7+a1a6) +
    // A2(a0a8+a2a6) + A4a1a7 + A5(a1a8+a2a7) + A8a2a8 计算covarsC11 = A0a3^2 +
    // 2A1a3a4 + 2A2a3a5 + A4a4^2 + 2A5a4a5 + A8a5^2 计算covarsC12 = A0a3a6 +
    // A1(a3a7+a4a6) + A2(a3a8+a5a6) + A4a4a7 + A5(a4a8+a6a7) + A8a5a8
    // 计算covarsC22 = A0a6^2 + 2A1a6a7 + 2A2a6a8 + A4a7^2 + 2A5a7a8 + A8a8^2
    LocalTensor<float> covars00 = covars[offset * COVMAT_INDEX11];
    LocalTensor<float> covars01 = covars[offset * COVMAT_INDEX12];
    LocalTensor<float> covars02 = covars[offset * COVMAT_INDEX13];
    LocalTensor<float> covars11 = covars[offset * COVMAT_INDEX22];
    LocalTensor<float> covars12 = covars[offset * COVMAT_INDEX23];
    LocalTensor<float> covars22 = covars[offset * COVMAT_INDEX33];

    LocalTensor<float> covarsC00 = covarsC[offset * COVMAT_INDEX11];
    LocalTensor<float> covarsC01 = covarsC[offset * COVMAT_INDEX12];
    LocalTensor<float> covarsC02 = covarsC[offset * COVMAT_INDEX13];
    LocalTensor<float> covarsC11 = covarsC[offset * COVMAT_INDEX22];
    LocalTensor<float> covarsC12 = covarsC[offset * COVMAT_INDEX23];
    LocalTensor<float> covarsC22 = covarsC[offset * COVMAT_INDEX33];
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

    Muls(covarsC00, covars00, a0 * a0, element);
    Muls(covarsC01, covars00, a0 * a3, element);
    Muls(covarsC02, covars00, a0 * a6, element);
    Muls(covarsC11, covars00, a3 * a3, element);
    Muls(covarsC12, covars00, a3 * a6, element);
    Muls(covarsC22, covars00, a6 * a6, element);

    PipeBarrier<PIPE_V>();
    Axpy(covarsC00, covars01, a0 * a1 + a0 * a1, element);
    Axpy(covarsC01, covars01, a0 * a4 + a1 * a3, element);
    Axpy(covarsC02, covars01, a0 * a7 + a1 * a6, element);
    Axpy(covarsC11, covars01, a3 * a4 + a3 * a4, element);
    Axpy(covarsC12, covars01, a3 * a7 + a4 * a6, element);
    Axpy(covarsC22, covars01, a6 * a7 + a6 * a7, element);

    PipeBarrier<PIPE_V>();
    Axpy(covarsC00, covars02, a0 * a2 + a0 * a2, element);
    Axpy(covarsC01, covars02, a0 * a5 + a2 * a3, element);
    Axpy(covarsC02, covars02, a0 * a8 + a2 * a6, element);
    Axpy(covarsC11, covars02, a3 * a5 + a3 * a5, element);
    Axpy(covarsC12, covars02, a3 * a8 + a5 * a6, element);
    Axpy(covarsC22, covars02, a6 * a8 + a6 * a8, element);

    PipeBarrier<PIPE_V>();
    Axpy(covarsC00, covars11, a1 * a1, element);
    Axpy(covarsC01, covars11, a1 * a4, element);
    Axpy(covarsC02, covars11, a1 * a7, element);
    Axpy(covarsC11, covars11, a4 * a4, element);
    Axpy(covarsC12, covars11, a4 * a7, element);
    Axpy(covarsC22, covars11, a7 * a7, element);

    PipeBarrier<PIPE_V>();
    Axpy(covarsC00, covars12, a1 * a2 + a1 * a2, element);
    Axpy(covarsC01, covars12, a1 * a5 + a2 * a4, element);
    Axpy(covarsC02, covars12, a1 * a8 + a2 * a7, element);
    Axpy(covarsC11, covars12, a4 * a5 + a4 * a5, element);
    Axpy(covarsC12, covars12, a4 * a8 + a5 * a7, element);
    Axpy(covarsC22, covars12, a7 * a8 + a7 * a8, element);

    PipeBarrier<PIPE_V>();
    Axpy(covarsC00, covars22, a2 * a2, element);
    Axpy(covarsC01, covars22, a2 * a5, element);
    Axpy(covarsC02, covars22, a2 * a8, element);
    Axpy(covarsC11, covars22, a5 * a5, element);
    Axpy(covarsC12, covars22, a5 * a8, element);
    Axpy(covarsC22, covars22, a8 * a8, element);

    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::PosW2CVjp(LocalTensor<float> &R, LocalTensor<float> &t,
                                                          LocalTensor<float> &means, LocalTensor<float> &vMeansC,
                                                          LocalTensor<float> &vR, int64_t vPWOffset, int element,
                                                          int c)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> vRTmp = calBuf_.Get<float>();
    LocalTensor<float> reduceSumResult = vRTmp[offset * REDUCESUMRESULT_OFFSET];
    LocalTensor<float> reduceSumTmp = vRTmp[offset * REDUCESUMTMP_OFFSET];
    for (int i = 0; i < VR_DIM; ++i) {
        for (int j = 0; j < VR_DIM; ++j) {
            Mul(vRTmp[offset * (i * VR_DIM + j)], vMeansC[offset * i], means[offset * j], element);
        }
    }
    PipeBarrier<PIPE_V>();
    for (int i = 0; i < VR_DIM; ++i) {
        for (int j = 0; j < VR_DIM; ++j) {
            ReduceSum<float>(reduceSumResult[(i * VR_DIM + j) * FLOAT_ONE_BLOCK_NUM], vRTmp[offset * (i * VR_DIM + j)],
                             reduceSumTmp, element);
            PipeBarrier<PIPE_V>();
        }
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (int i = 0; i < VR_DIM; ++i) {
        for (int j = 0; j < VR_DIM; ++j) {
            float v = reduceSumResult[(i * VR_DIM + j) * FLOAT_ONE_BLOCK_NUM].GetValue(0);
            vR.SetValue(i * VR_DIM + j, v);
        }
    }

    float a0 = R.GetValue(RMAT_INDEX11);
    float a1 = R.GetValue(RMAT_INDEX12);
    float a2 = R.GetValue(RMAT_INDEX13);
    float a3 = R.GetValue(RMAT_INDEX21);
    float a4 = R.GetValue(RMAT_INDEX22);
    float a5 = R.GetValue(RMAT_INDEX23);
    float a6 = R.GetValue(RMAT_INDEX31);
    float a7 = R.GetValue(RMAT_INDEX32);
    float a8 = R.GetValue(RMAT_INDEX33);

    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    // 修改转置
    LocalTensor<float> vPW = copyInTensor;
    LocalTensor<float> vPWtmp = vRTmp[VPWTMP_OFFSET * offset];
    Muls(vPWtmp[offset * VPWTMP1_OFFSET], vMeansC[offset * VMEANSC1_OFFSET], a0, element);
    Muls(vPWtmp[offset * VPWTMP2_OFFSET], vMeansC[offset * VMEANSC1_OFFSET], a1, element);
    Muls(vPWtmp[offset * VPWTMP3_OFFSET], vMeansC[offset * VMEANSC1_OFFSET], a2, element);
    PipeBarrier<PIPE_V>();
    Axpy(vPWtmp[offset * VPWTMP1_OFFSET], vMeansC[offset * VMEANSC2_OFFSET], a3, element);
    Axpy(vPWtmp[offset * VPWTMP2_OFFSET], vMeansC[offset * VMEANSC2_OFFSET], a4, element);
    Axpy(vPWtmp[offset * VPWTMP3_OFFSET], vMeansC[offset * VMEANSC2_OFFSET], a5, element);
    PipeBarrier<PIPE_V>();
    Axpy(vPWtmp[offset * VPWTMP1_OFFSET], vMeansC[offset * VMEANSC3_OFFSET], a6, element);
    Axpy(vPWtmp[offset * VPWTMP2_OFFSET], vMeansC[offset * VMEANSC3_OFFSET], a7, element);
    Axpy(vPWtmp[offset * VPWTMP3_OFFSET], vMeansC[offset * VMEANSC3_OFFSET], a8, element);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> indexTensor = calBuf_.Get<int32_t>();

    for (int i = 0; i < element; ++i) {
        indexTensor.SetValue(i * VPW_ELEMENT + VPWTMP1_OFFSET, (i + offset * VPWTMP1_OFFSET) * FLOAT_SIZE);
        indexTensor.SetValue(i * VPW_ELEMENT + VPWTMP2_OFFSET, (i + offset * VPWTMP2_OFFSET) * FLOAT_SIZE);
        indexTensor.SetValue(i * VPW_ELEMENT + VPWTMP3_OFFSET, (i + offset * VPWTMP3_OFFSET) * FLOAT_SIZE);
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    AscendC::Gather<float>(vPW, vPWtmp, indexTensor.ReinterpretCast<uint32_t>(), 0, VPW_ELEMENT * element);

    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(VPW_ELEMENT * element * sizeof(float)), 0, 0, 0};

    // 原子累加
    if (c != 0) {
        SetAtomicAdd<float>();
    }
    DataCopyPad(vPWGm_[vPWOffset], vPW, dataCopyParams);
    if (c != 0) {
        SetAtomicNone();
    }

    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::CovarW2CVjp(LocalTensor<float> &R, LocalTensor<float> &covars,
                                                            LocalTensor<float> &vCovarC, LocalTensor<float> &vR,
                                                            LocalTensor<float> &vCovar, int64_t vROffset,
                                                            int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> Share = intermediateBuf_.Get<float>();
    LocalTensor<float> buffer1 = Share[offset * COVARSW2C_BUF1_OFFSET];
    LocalTensor<float> buffer2 = buffer1[offset * COVARSW2C_BUF2_OFFSET];
    LocalTensor<float> buffer3 = calBuf_.Get<float>();
    Duplicate<float>(buffer1, 0, COVARSW2C_BUFLEN * offset);
    Duplicate<float>(buffer2, 0, COVARSW2C_BUFLEN * offset);
    Duplicate<float>(buffer3, 0, COVARSW2C_BUFLEN * offset);
    for (int i = 0; i < VR_DIM; ++i) {
        for (int k = 0; k < VR_DIM; ++k) {
            for (int j = 0; j < VR_DIM; ++j) {
                Axpy(buffer1[offset * (i * VR_DIM + j)], vCovarC[offset * GetSymmetricIndex(i, k)],
                     R.GetValue(k * VR_DIM + j), element);
                PipeBarrier<PIPE_V>();
            }
        }
    }
    for (int i = 0; i < VR_DIM; ++i) {
        for (int j = 0; j < VR_DIM; ++j) {
            for (int k = 0; k < VR_DIM; ++k) {
                PipeBarrier<PIPE_V>();
                // covars有转置，偏移有变化
                Mul(buffer2[offset * (i * VR_DIM + j)], buffer1[offset * (i * VR_DIM + k)],
                    covars[offset * GetSymmetricIndex(k, j)], element);
                PipeBarrier<PIPE_V>();
                Add(buffer3[offset * (i * VR_DIM + j)], buffer3[offset * (i * VR_DIM + j)],
                    buffer2[offset * (i * VR_DIM + j)], element);
            }
        }
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float> vROut = copyInTensor;
    for (int i = 0; i < VR_DIM; ++i) {
        for (int j = 0; j < VR_DIM; ++j) {
            ReduceSum<float>(buffer1, buffer3[offset * (i * VR_DIM + j)], buffer2, element);
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            float value = vR.GetValue(i * VR_DIM + j) + buffer1.GetValue(0) * TWO_FLOAT_VALUE;
            // vCovarC * R * covar.T 和 vCovarC.T * R * covar 相同，直接乘2
            vROut.SetValue(i * VR_DIM + j, value);
            SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        }
    }
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(9 * sizeof(float)), 0, 0, 0};
    // 原子累加
    SetAtomicAdd<float>();
    DataCopyPad(vRGm_[vROffset], vROut, dataCopyParams);
    SetAtomicNone();
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    PipeBarrier<PIPE_V>();
    float a0 = R.GetValue(RMAT_INDEX11);
    float a1 = R.GetValue(RMAT_INDEX12);
    float a2 = R.GetValue(RMAT_INDEX13);
    float a3 = R.GetValue(RMAT_INDEX21);
    float a4 = R.GetValue(RMAT_INDEX22);
    float a5 = R.GetValue(RMAT_INDEX23);
    float a6 = R.GetValue(RMAT_INDEX31);
    float a7 = R.GetValue(RMAT_INDEX32);
    float a8 = R.GetValue(RMAT_INDEX33);
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    LocalTensor<float> vCovarC00 = vCovarC[offset * COVMAT_INDEX11];
    LocalTensor<float> vCovarC01 = vCovarC[offset * COVMAT_INDEX12];
    LocalTensor<float> vCovarC02 = vCovarC[offset * COVMAT_INDEX13];
    LocalTensor<float> vCovarC11 = vCovarC[offset * COVMAT_INDEX22];
    LocalTensor<float> vCovarC12 = vCovarC[offset * COVMAT_INDEX23];
    LocalTensor<float> vCovarC22 = vCovarC[offset * COVMAT_INDEX33];

    LocalTensor<float> vCovar00 = vCovar[offset * COVMAT_INDEX11];
    LocalTensor<float> vCovar01 = vCovar[offset * COVMAT_INDEX12];
    LocalTensor<float> vCovar02 = vCovar[offset * COVMAT_INDEX13];
    LocalTensor<float> vCovar11 = vCovar[offset * COVMAT_INDEX22];
    LocalTensor<float> vCovar12 = vCovar[offset * COVMAT_INDEX23];
    LocalTensor<float> vCovar22 = vCovar[offset * COVMAT_INDEX33];

    Axpy(vCovar00, vCovarC00, a0 * a0, element);
    Axpy(vCovar01, vCovarC00, a0 * a1, element);
    Axpy(vCovar02, vCovarC00, a0 * a2, element);
    Axpy(vCovar11, vCovarC00, a1 * a1, element);
    Axpy(vCovar12, vCovarC00, a1 * a2, element);
    Axpy(vCovar22, vCovarC00, a2 * a2, element);

    PipeBarrier<PIPE_V>();
    Axpy(vCovar00, vCovarC01, a0 * a3 + a0 * a3, element);
    Axpy(vCovar01, vCovarC01, a0 * a4 + a1 * a3, element);
    Axpy(vCovar02, vCovarC01, a0 * a5 + a2 * a3, element);
    Axpy(vCovar11, vCovarC01, a1 * a4 + a1 * a4, element);
    Axpy(vCovar12, vCovarC01, a1 * a5 + a2 * a4, element);
    Axpy(vCovar22, vCovarC01, a2 * a5 + a2 * a5, element);

    PipeBarrier<PIPE_V>();
    Axpy(vCovar00, vCovarC02, a0 * a6 + a0 * a6, element);
    Axpy(vCovar01, vCovarC02, a0 * a7 + a1 * a6, element);
    Axpy(vCovar02, vCovarC02, a0 * a8 + a2 * a6, element);
    Axpy(vCovar11, vCovarC02, a1 * a7 + a1 * a7, element);
    Axpy(vCovar12, vCovarC02, a1 * a8 + a2 * a7, element);
    Axpy(vCovar22, vCovarC02, a2 * a8 + a2 * a8, element);

    PipeBarrier<PIPE_V>();
    Axpy(vCovar00, vCovarC11, a3 * a3, element);
    Axpy(vCovar01, vCovarC11, a3 * a4, element);
    Axpy(vCovar02, vCovarC11, a3 * a5, element);
    Axpy(vCovar11, vCovarC11, a4 * a4, element);
    Axpy(vCovar12, vCovarC11, a4 * a5, element);
    Axpy(vCovar22, vCovarC11, a5 * a5, element);

    PipeBarrier<PIPE_V>();
    Axpy(vCovar00, vCovarC12, a3 * a6 + a3 * a6, element);
    Axpy(vCovar01, vCovarC12, a3 * a7 + a4 * a6, element);
    Axpy(vCovar02, vCovarC12, a3 * a8 + a5 * a6, element);
    Axpy(vCovar11, vCovarC12, a4 * a7 + a4 * a7, element);
    Axpy(vCovar12, vCovarC12, a4 * a8 + a5 * a7, element);
    Axpy(vCovar22, vCovarC12, a5 * a8 + a5 * a8, element);

    PipeBarrier<PIPE_V>();
    Axpy(vCovar00, vCovarC22, a6 * a6, element);
    Axpy(vCovar01, vCovarC22, a6 * a7, element);
    Axpy(vCovar02, vCovarC22, a6 * a8, element);
    Axpy(vCovar11, vCovarC22, a7 * a7, element);
    Axpy(vCovar12, vCovarC22, a7 * a8, element);
    Axpy(vCovar22, vCovarC22, a8 * a8, element);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::PerspProjVjp(LocalTensor<float> &vCovar2d,
    LocalTensor<float> &covarsC,
    LocalTensor<float> &meansC, LocalTensor<float> &Ks,
    LocalTensor<float> &vMeans2d, LocalTensor<float> &vDepths,
    LocalTensor<float> &vMeansC, LocalTensor<float> &vCovarC, int element)
{
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    int64_t offset = Align(element, FLOAT_SIZE);
    float fx = Ks.GetValue(FX_INDEX);
    float fy = Ks.GetValue(FY_INDEX);
    float cx = Ks.GetValue(CX_INDEX);
    float cy = Ks.GetValue(CY_INDEX);
    ASSERT(fx != 0.0f && "fx can not be Zero!");
    ASSERT(fy != 0.0f && "fy can not be Zero!");
    float tanFovx = TANFOV_COE * width_ / fx;
    float tanFovy = TANFOV_COE * height_ / fy;
    float limXPos = (width_ - cx) / fx + LIM_COE * tanFovx;
    float limX = -LIM_COE * tanFovx - cx / fx;
    float limYPos = (height_ - cy) / fy + LIM_COE * tanFovy;
    float limY = -LIM_COE * tanFovy - cy / fy;
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

    LocalTensor<float> tx = meansC[offset * VMEANSC1_OFFSET];
    LocalTensor<float> ty = meansC[offset * VMEANSC2_OFFSET];
    LocalTensor<float> tz = meansC[offset * VMEANSC3_OFFSET];

    LocalTensor<float> vMeans2d0 = vMeans2d[offset * VMEANSC2D1_OFFSET];
    LocalTensor<float> vMeans2d1 = vMeans2d[offset * VMEANSC2D2_OFFSET];

    LocalTensor<float> vCovar2d00 = vCovar2d[offset * VCOV2D1_OFFSET];
    LocalTensor<float> vCovar2d01 = vCovar2d[offset * VCOV2D2_OFFSET];
    LocalTensor<float> vCovar2d11 = vCovar2d[offset * VCOV2D3_OFFSET];

    LocalTensor<float> TempBuf = calBuf_.Get<float>();
    LocalTensor<float> txDivTz = TempBuf[offset * TXDIVTZ_OFFSET];
    LocalTensor<float> tyDivTz = TempBuf[offset * TYDIVTZ_OFFSET]; // TempBuf[56] 56 * 4 / 32 = 7
    LocalTensor<float> fxDivTz = TempBuf[offset * FXDIVTZ_OFFSET];
    LocalTensor<float> fyDivTz = TempBuf[offset * FYDIVTZ_OFFSET];
    LocalTensor<float> tmpBuf0 = TempBuf[offset * PERSP_TMPBUF0_OFFSET];
    LocalTensor<float> tmpBuf1 = TempBuf[offset * PERSP_TMPBUF1_OFFSET];
    LocalTensor<float> tmpBuf2 = TempBuf[offset * PERSP_TMPBUF2_OFFSET];
    LocalTensor<float> tmpBuf3 = TempBuf[offset * PERSP_TMPBUF3_OFFSET];
    LocalTensor<float> tmpBuf4 = TempBuf[offset * PERSP_TMPBUF4_OFFSET];

    /* calc tx/tz, ty/tz, fx/tz, fy/tz, for later reuse */
    Div(txDivTz, tx, tz, element);
    Div(tyDivTz, ty, tz, element);
    Duplicate<float>(tmpBuf0, fx, element);
    PipeBarrier<PIPE_V>();
    Div(fxDivTz, tmpBuf0, tz, element);

    PipeBarrier<PIPE_V>();
    Duplicate<float>(tmpBuf0, fy, element);
    PipeBarrier<PIPE_V>();
    Div(fyDivTz, tmpBuf0, tz, element);
    PipeBarrier<PIPE_V>();
    /* calc v_mean_c */
    LocalTensor<float> vMeanC0 = vMeansC[offset * VMEANSC1_OFFSET];
    LocalTensor<float> vMeanC1 = vMeansC[offset * VMEANSC2_OFFSET];
    LocalTensor<float> vMeanC2 = vMeansC[offset * VMEANSC3_OFFSET];
    Mul(vMeanC0, fxDivTz, vMeans2d0, element);
    Mul(vMeanC1, fyDivTz, vMeans2d1, element);
    PipeBarrier<PIPE_V>();
    Mul(vMeanC2, vMeanC0, txDivTz, element);
    Mul(tmpBuf0, vMeanC1, tyDivTz, element);
    PipeBarrier<PIPE_V>();
    Add(vMeanC2, vMeanC2, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Muls(vMeanC2, vMeanC2, MINUS_ONE_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    int64_t offset16 = Align256(element, INT16_SIZE);
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

    LocalTensor<uint16_t> xClippingMask0 = maskBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> xClippingMask1 = xClippingMask0[offset16 * XCLIPMASK1_OFFSET];
    LocalTensor<uint16_t> yClippingMask0 = xClippingMask0[offset16 * YCLIPMASK0_OFFSET];
    LocalTensor<uint16_t> yClippingMask1 = xClippingMask0[offset16 * YCLIPMASK1_OFFSET];

    LocalTensor<uint8_t> xClippingMask0uint8 = xClippingMask0.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> xClippingMask1uint8 = xClippingMask1.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> yClippingMask0uint8 = yClippingMask0.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> yClippingMask1uint8 = yClippingMask1.ReinterpretCast<uint8_t>();

    CompareScalar(xClippingMask0uint8, txDivTz, limX, CMPMODE::GE, offset16);
    CompareScalar(xClippingMask1uint8, txDivTz, limXPos, CMPMODE::LE, offset16);
    CompareScalar(yClippingMask0uint8, tyDivTz, limY, CMPMODE::GE, offset16);
    CompareScalar(yClippingMask1uint8, tyDivTz, limYPos, CMPMODE::LE, offset16);

    PipeBarrier<PIPE_V>();
    And(xClippingMask0, xClippingMask0, xClippingMask1, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));
    And(yClippingMask0, yClippingMask0, yClippingMask1, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));
    /* calc clamped tx and ty */

    Maxs(txDivTz, txDivTz, limX, element);
    Maxs(tyDivTz, tyDivTz, limY, element);
    PipeBarrier<PIPE_V>();
    Mins(txDivTz, txDivTz, limXPos, element);
    Mins(tyDivTz, tyDivTz, limYPos, element);
    PipeBarrier<PIPE_V>();
    Mul(tx, txDivTz, tz, element);
    Mul(ty, tyDivTz, tz, element);
    PipeBarrier<PIPE_V>();
    /* calc -tx/tz and -ty/tz */
    Div(txDivTz, tx, tz, element);
    Div(tyDivTz, ty, tz, element);
    PipeBarrier<PIPE_V>();
    Muls(txDivTz, txDivTz, static_cast<float>(-1.0), element);
    Muls(tyDivTz, tyDivTz, static_cast<float>(-1.0), element);
    PipeBarrier<PIPE_V>();

    /* calc vCovarC,  it's a 3x3 symmetric matrix, only need to store
     * C00/C01/C02/C11/C12/C22 */
    LocalTensor<float> vCovarC00 = vCovarC[offset * VCOVARC00_OFFSET];
    LocalTensor<float> vCovarC01 = vCovarC[offset * VCOVARC01_OFFSET];
    LocalTensor<float> vCovarC02 = vCovarC[offset * VCOVARC02_OFFSET];
    LocalTensor<float> vCovarC11 = vCovarC[offset * VCOVARC11_OFFSET];
    LocalTensor<float> vCovarC12 = vCovarC[offset * VCOVARC12_OFFSET];
    LocalTensor<float> vCovarC22 = vCovarC[offset * VCOVARC22_OFFSET];
    Mul(tmpBuf0, tz, tz, element);
    Mul(vCovarC00, vCovar2d00, fxDivTz, element);
    Mul(vCovarC01, vCovar2d01, fxDivTz, element);
    Mul(vCovarC11, vCovar2d11, fyDivTz, element);
    PipeBarrier<PIPE_V>();
    Mul(vCovarC00, vCovarC00, fxDivTz, element);
    Mul(vCovarC01, vCovarC01, fyDivTz, element);
    Mul(vCovarC11, vCovarC11, fyDivTz, element);
    PipeBarrier<PIPE_V>();

    Mul(vCovarC02, vCovarC00, txDivTz, element);
    Mul(vCovarC12, vCovarC01, txDivTz, element);
    Mul(tmpBuf0, vCovarC01, tyDivTz, element);
    Mul(tmpBuf1, vCovarC11, tyDivTz, element);
    PipeBarrier<PIPE_V>();
    Add(vCovarC02, vCovarC02, tmpBuf0, element);
    Add(vCovarC12, vCovarC12, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    Mul(tmpBuf0, vCovarC02, txDivTz, element);
    Mul(tmpBuf1, vCovarC12, tyDivTz, element);
    PipeBarrier<PIPE_V>();
    Add(vCovarC22, tmpBuf0, tmpBuf1, element);
    PipeBarrier<PIPE_V>();
    /* v_cov2d @ J @ covar_c */
    LocalTensor<float> covarC00 = covarsC[offset * COVARC00_OFFSET];
    LocalTensor<float> covarC01 = covarsC[offset * COVARC01_OFFSET];
    LocalTensor<float> covarC02 = covarsC[offset * COVARC02_OFFSET];
    LocalTensor<float> covarC11 = covarsC[offset * COVARC11_OFFSET];
    LocalTensor<float> covarC12 = covarsC[offset * COVARC12_OFFSET];
    LocalTensor<float> covarC22 = covarsC[offset * COVARC22_OFFSET];

    /* J00 = fxDivTz, J11 = fyDivTz, J01 = J10 = 0 */
    LocalTensor<float> j00 = fxDivTz;
    LocalTensor<float> j11 = fyDivTz;
    LocalTensor<float> j02 = tmpBuf2;
    LocalTensor<float> j12 = tmpBuf3;
    Mul(j02, fxDivTz, txDivTz, element);
    Mul(j12, fyDivTz, tyDivTz, element);
    PipeBarrier<PIPE_V>();

    /* A = vCovar2d @ J, [2, 2] @ [2, 3] = [2, 3] */
    LocalTensor<float> A00 = tx;
    LocalTensor<float> A01 = ty;
    LocalTensor<float> A02 = vMeans2d;
    LocalTensor<float> A10 = TempBuf[offset * A10_OFFSET];
    LocalTensor<float> A11 = TempBuf[offset * A11_OFFSET];
    LocalTensor<float> A12 = TempBuf[offset * A12_OFFSET];
    Mul(A00, vCovar2d00, j00, element);
    Mul(A01, vCovar2d01, j11, element);
    Mul(A02, vCovar2d00, j02, element);
    Mul(tmpBuf0, vCovar2d01, j12, element);
    PipeBarrier<PIPE_V>();
    Add(A02, A02, tmpBuf0, element);
    Mul(A10, vCovar2d01, j00, element);
    Mul(A11, vCovar2d11, j11, element);
    Mul(A12, vCovar2d01, j02, element);
    Mul(tmpBuf1, vCovar2d11, j12, element);
    PipeBarrier<PIPE_V>();
    Add(A12, A12, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    /* v_J = A @ covar_c, [2, 3] @ [3, 3] = [2, 3] */
    LocalTensor<float> vJ00 = vCovar2d[offset * VJ00_OFFSET];
    LocalTensor<float> vJ01 = vCovar2d[offset * VJ01_OFFSET];
    LocalTensor<float> vJ02 = vCovar2d[offset * VJ02_OFFSET];
    LocalTensor<float> vJ10 = A00;
    LocalTensor<float> vJ11 = A01;
    LocalTensor<float> vJ12 = A02;
    Mul(vJ00, A00, covarC00, element);
    Mul(tmpBuf0, A01, covarC01, element);
    Mul(tmpBuf1, A02, covarC02, element);
    PipeBarrier<PIPE_V>();
    Add(vJ00, vJ00, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Add(vJ00, vJ00, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    Mul(vJ01, A00, covarC01, element);
    Mul(tmpBuf0, A01, covarC11, element);
    Mul(tmpBuf1, A02, covarC12, element);
    PipeBarrier<PIPE_V>();
    Add(vJ01, vJ01, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Add(vJ01, vJ01, tmpBuf1, element);

    Mul(vJ02, A00, covarC02, element);
    Mul(tmpBuf0, A01, covarC12, element);
    Mul(tmpBuf1, A02, covarC22, element);
    PipeBarrier<PIPE_V>();
    Add(vJ02, vJ02, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Add(vJ02, vJ02, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    Mul(vJ10, A10, covarC00, element);
    Mul(tmpBuf0, A11, covarC01, element);
    Mul(tmpBuf1, A12, covarC02, element);
    PipeBarrier<PIPE_V>();
    Add(vJ10, vJ10, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Add(vJ10, vJ10, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    Mul(vJ11, A10, covarC01, element);
    Mul(tmpBuf0, A11, covarC11, element);
    Mul(tmpBuf1, A12, covarC12, element);
    PipeBarrier<PIPE_V>();
    Add(vJ11, vJ11, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Add(vJ11, vJ11, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    Mul(vJ12, A10, covarC02, element);
    Mul(tmpBuf0, A11, covarC12, element);
    Mul(tmpBuf1, A12, covarC22, element);
    PipeBarrier<PIPE_V>();
    Add(vJ12, vJ12, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Add(vJ12, vJ12, tmpBuf1, element);
    PipeBarrier<PIPE_V>();

    Add(vJ00, vJ00, vJ00, element);
    Add(vJ01, vJ01, vJ01, element);
    Add(vJ02, vJ02, vJ02, element);
    Add(vJ10, vJ10, vJ10, element);
    Add(vJ11, vJ11, vJ11, element);
    Add(vJ12, vJ12, vJ12, element);

    PipeBarrier<PIPE_V>();

    /* update v_mean_c */
    LocalTensor<float> fxDivTz2 = TempBuf[offset * FXDIVTZ2_OFFSET];
    LocalTensor<float> fyDivTz2 = TempBuf[offset * FYDIVTZ2_OFFSET];
    Div(fxDivTz2, fxDivTz, tz, element);
    Div(fyDivTz2, fyDivTz, tz, element);
    PipeBarrier<PIPE_V>();
    Mul(tmpBuf0, fxDivTz2, vJ02, element);
    PipeBarrier<PIPE_V>();
    Sub(tmpBuf1, vMeanC0, tmpBuf0, element);
    PipeBarrier<PIPE_V>();
    Select(vMeanC0, xClippingMask0, tmpBuf1, vMeanC0, SELMODE::VSEL_TENSOR_TENSOR_MODE, element);

    Mul(tmpBuf2, fyDivTz2, vJ12, element);
    PipeBarrier<PIPE_V>();
    Sub(tmpBuf3, vMeanC1, tmpBuf2, element);
    PipeBarrier<PIPE_V>();
    Select(vMeanC1, yClippingMask0, tmpBuf3, vMeanC1, SELMODE::VSEL_TENSOR_TENSOR_MODE, element);

    Not(xClippingMask1, xClippingMask0, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));
    Mul(tmpBuf0, tmpBuf0, txDivTz, element);
    PipeBarrier<PIPE_V>();
    Add(tmpBuf1, vMeanC2, tmpBuf0, element);
    PipeBarrier<PIPE_V>();

    Select(vMeanC2, xClippingMask1, tmpBuf1, vMeanC2, SELMODE::VSEL_TENSOR_TENSOR_MODE, element);

    Not(yClippingMask1, yClippingMask0, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));
    Mul(tmpBuf2, tmpBuf2, tyDivTz, element);
    PipeBarrier<PIPE_V>();
    Add(tmpBuf3, vMeanC2, tmpBuf2, element);
    PipeBarrier<PIPE_V>();
    Select(vMeanC2, yClippingMask1, tmpBuf3, vMeanC2, SELMODE::VSEL_TENSOR_TENSOR_MODE, element);

    Mul(tmpBuf1, fxDivTz2, vJ00, element);
    PipeBarrier<PIPE_V>();
    Sub(vMeanC2, vMeanC2, tmpBuf1, element);
    Mul(tmpBuf3, fyDivTz2, vJ11, element);
    PipeBarrier<PIPE_V>();
    Sub(vMeanC2, vMeanC2, tmpBuf3, element);
    PipeBarrier<PIPE_V>();
    Axpy(vMeanC2, tmpBuf0, VMEANC2_VALUE, element);
    PipeBarrier<PIPE_V>();
    Axpy(vMeanC2, tmpBuf2, VMEANC2_VALUE, element);
    PipeBarrier<PIPE_V>();
    Add(vMeanC2, vMeanC2, vDepths, element);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::QuatScaleToCovarVjp(LocalTensor<float> &quats,
    LocalTensor<float> &scales, LocalTensor<float> &rotmat, LocalTensor<float> &vCovar,
    LocalTensor<float> &v_quat, LocalTensor<float> &vScales, int element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> M = calBuf_.Get<float>();
    // M = scales * rotmat
    Mul(M[offset * M_FISRTLINE_OFFSET], rotmat[offset * M_FISRTLINE_OFFSET], scales, offset * M_DIM);
    Mul(M[offset * M_SECONDLINE_OFFSET], rotmat[offset * M_SECONDLINE_OFFSET], scales, offset * M_DIM);
    Mul(M[offset * M_THIRDLINE_OFFSET], rotmat[offset * M_THIRDLINE_OFFSET], scales, offset * M_DIM);

    // vCovar = vCovar + vCovar.T 对称矩阵，直接乘2
    Muls(vCovar, vCovar, TWO_FLOAT_VALUE, offset * VMMAT_THIRDLINE_OFFSET);
    PipeBarrier<PIPE_V>();
    // v_M = temp * M
    LocalTensor<float> TempBuf = intermediateBuf_.Get<float>();
    LocalTensor<float> v_M = TempBuf[offset * VM_OFFSET];
    LocalTensor<float> temp = TempBuf[offset * QUATSCALETOCO_TEMP_OFFSET];
    Duplicate<float>(v_M, 0, VM_LEN * offset);
    Duplicate<float>(temp, 0, VM_LEN * offset);
    for (int i = 0; i < M_DIM; ++i) {
        for (int j = 0; j < M_DIM; ++j) {
            for (int k = 0; k < M_DIM; ++k) {
                PipeBarrier<PIPE_V>();
                Mul(temp, vCovar[offset * GetSymmetricIndex(i, k)], M[offset * (k * M_DIM + j)], element);
                PipeBarrier<PIPE_V>();
                Add(v_M[offset * (i * M_DIM + j)], v_M[offset * (i * M_DIM + j)], temp, element);
            }
        }
    }
    PipeBarrier<PIPE_V>();
    // vScales 为v_M和rotmat逐元素相乘，在倒数第二维上求和
    Duplicate<float>(vScales, 0, M_DIM * offset);
    for (int j = 0; j < M_DIM; ++j) {
        for (int i = 0; i < M_DIM; ++i) {
            PipeBarrier<PIPE_V>();
            Mul(temp, v_M[offset * (i * M_DIM + j)], rotmat[offset * (i * M_DIM + j)], element);
            PipeBarrier<PIPE_V>();
            Add(vScales[offset * j], vScales[offset * j], temp, element);
        }
    }
    // vR复用M
    LocalTensor<float> vR = calBuf_.Get<float>();
    Mul(vR[offset * VRMAT_FIRSTLINE_OFFSET], v_M[offset * VMMAT_FIRSTLINE_OFFSET], scales, offset * M_DIM);
    Mul(vR[offset * VRMAT_SECONDLINE_OFFSET], v_M[offset * VMMAT_SECONDLINE_OFFSET], scales, offset * M_DIM);
    Mul(vR[offset * VRMAT_THIRDLINE_OFFSET], v_M[offset * VMMAT_THIRDLINE_OFFSET], scales, offset * M_DIM);

    // 存储norm结果
    LocalTensor<float> quats_n = TempBuf[offset * QUATSN_OFFSET];
    LocalTensor<float> norm = TempBuf[offset * NORM_OFFSET];
    Mul(norm, quats, quats, offset * QUAT_ELEMENT);
    PipeBarrier<PIPE_V>();
    Add(norm, norm, norm[offset * X_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(norm, norm, norm[offset * Y_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(norm, norm, norm[offset * Z_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Sqrt(norm, norm, element);
    PipeBarrier<PIPE_V>();
    Maxs(norm, norm, 1e-12f, element);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> w = quats_n[offset * W_OFFSET];
    LocalTensor<float> x = quats_n[offset * X_OFFSET];
    LocalTensor<float> y = quats_n[offset * Y_OFFSET];
    LocalTensor<float> z = quats_n[offset * Z_OFFSET];

    Div(w, quats[offset * W_OFFSET], norm, element);
    Div(x, quats[offset * X_OFFSET], norm, element);
    Div(y, quats[offset * Y_OFFSET], norm, element);
    Div(z, quats[offset * Z_OFFSET], norm, element);

    LocalTensor<float> vR00 = vR[offset * VR00_OFFSET];
    LocalTensor<float> vR01 = vR[offset * VR01_OFFSET];
    LocalTensor<float> vR02 = vR[offset * VR02_OFFSET];
    LocalTensor<float> vR10 = vR[offset * VR10_OFFSET];
    LocalTensor<float> vR11 = vR[offset * VR11_OFFSET];
    LocalTensor<float> vR12 = vR[offset * VR12_OFFSET];
    LocalTensor<float> vR20 = vR[offset * VR20_OFFSET];
    LocalTensor<float> vR21 = vR[offset * VR21_OFFSET];
    LocalTensor<float> vR22 = vR[offset * VR22_OFFSET];

    LocalTensor<float> v_quat_n = TempBuf[offset * VQUATN_OFFSET];
    temp = TempBuf[offset * QUATSCALCO_TMP_OFFSET];
    Duplicate<float>(v_quat_n, 0, QUAT_ELEMENT * offset);
    // x * vR12_vR21
    Sub(temp, vR21, vR12, element);
    PipeBarrier<PIPE_V>();
    Mul(v_quat_n[offset * W_OFFSET], temp, x, element);
    PipeBarrier<PIPE_V>();
    // y * vR20_vR02
    Sub(temp, vR02, vR20, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, y, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * W_OFFSET], v_quat_n[offset * W_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // z * vR01_vR10
    Sub(temp, vR10, vR01, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, z, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * W_OFFSET], v_quat_n[offset * W_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // 上述3项之和再乘2，算出第1部分
    Muls(v_quat_n[offset * W_OFFSET], v_quat_n[offset * W_OFFSET], TWO_FLOAT_VALUE, element);

    // -2.0 * x * vR11_add_vR22
    Add(temp, vR11, vR22, element);
    PipeBarrier<PIPE_V>();
    Mul(v_quat_n[offset * X_OFFSET], temp, x, element);
    PipeBarrier<PIPE_V>();
    Muls(v_quat_n[offset * X_OFFSET], v_quat_n[offset * X_OFFSET], -2.0f, element);
    PipeBarrier<PIPE_V>();
    // y * vR01_add_vR10
    Add(temp, vR01, vR10, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, y, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * X_OFFSET], v_quat_n[offset * X_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // z * vR02_add_vR20
    Add(temp, vR02, vR20, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, z, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * X_OFFSET], v_quat_n[offset * X_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // w * vR12_vR21
    Sub(temp, vR21, vR12, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, w, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * X_OFFSET], v_quat_n[offset * X_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // 上述4项之和再乘2，算出第2部分
    Muls(v_quat_n[offset * X_OFFSET], v_quat_n[offset * X_OFFSET], TWO_FLOAT_VALUE, element);

    // x * vR01_add_vR10
    Add(temp, vR01, vR10, element);
    PipeBarrier<PIPE_V>();
    Mul(v_quat_n[offset * Y_OFFSET], temp, x, element);
    PipeBarrier<PIPE_V>();
    // -2.0 * y * vR00_add_vR22
    Add(temp, vR00, vR22, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, y, element);
    PipeBarrier<PIPE_V>();
    Muls(temp, temp, -2.0f, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * Y_OFFSET], v_quat_n[offset * Y_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // z * vR12_add_vR21
    Add(temp, vR12, vR21, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, z, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * Y_OFFSET], v_quat_n[offset * Y_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // w * vR20_vR02
    Sub(temp, vR02, vR20, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, w, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * Y_OFFSET], v_quat_n[offset * Y_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // 上述4项之和再乘2，算出第3部分
    Muls(v_quat_n[offset * Y_OFFSET], v_quat_n[offset * Y_OFFSET], TWO_FLOAT_VALUE, element);

    // x * vR02_add_vR20
    Add(temp, vR02, vR20, element);
    PipeBarrier<PIPE_V>();
    Mul(v_quat_n[offset * Z_OFFSET], temp, x, element);
    PipeBarrier<PIPE_V>();
    // y * vR12_add_vR21
    Add(temp, vR12, vR21, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, y, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * Z_OFFSET], v_quat_n[offset * Z_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // -2.0 * z * vR00_add_vR11
    Add(temp, vR00, vR11, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, z, element);
    PipeBarrier<PIPE_V>();
    Muls(temp, temp, -2.0f, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * Z_OFFSET], v_quat_n[offset * Z_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // w * vR01_vR10
    Sub(temp, vR10, vR01, element);
    PipeBarrier<PIPE_V>();
    Mul(temp, temp, w, element);
    PipeBarrier<PIPE_V>();
    Add(v_quat_n[offset * Z_OFFSET], v_quat_n[offset * Z_OFFSET], temp, element);
    PipeBarrier<PIPE_V>();
    // 上述4项之和再乘2，算出第4部分
    Muls(v_quat_n[offset * Z_OFFSET], v_quat_n[offset * Z_OFFSET], TWO_FLOAT_VALUE, element);
    PipeBarrier<PIPE_V>();

    // quats_n * v_quat_n
    Mul(v_quat[offset * W_OFFSET], quats_n[offset * W_OFFSET], v_quat_n[offset * W_OFFSET], offset * QUAT_ELEMENT);
    PipeBarrier<PIPE_V>();
    Add(temp, v_quat, v_quat[offset * X_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(temp, temp, v_quat[offset * Y_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Add(temp, temp, v_quat[offset * Z_OFFSET], element);
    PipeBarrier<PIPE_V>();
    Mul(quats_n[offset * W_OFFSET], quats_n[offset * W_OFFSET], temp, element);
    Mul(quats_n[offset * X_OFFSET], quats_n[offset * X_OFFSET], temp, element);
    Mul(quats_n[offset * Y_OFFSET], quats_n[offset * Y_OFFSET], temp, element);
    Mul(quats_n[offset * Z_OFFSET], quats_n[offset * Z_OFFSET], temp, element);

    PipeBarrier<PIPE_V>();
    // v_quat_n - quats_n * v_quat_n
    Sub(v_quat, v_quat_n, quats_n, offset * QUAT_ELEMENT);
    PipeBarrier<PIPE_V>();
    Div(v_quat[offset * W_OFFSET], v_quat[offset * W_OFFSET], norm, element);
    Div(v_quat[offset * X_OFFSET], v_quat[offset * X_OFFSET], norm, element);
    Div(v_quat[offset * Y_OFFSET], v_quat[offset * Y_OFFSET], norm, element);
    Div(v_quat[offset * Z_OFFSET], v_quat[offset * Z_OFFSET], norm, element);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::SubProcess(int64_t i, int64_t k, int64_t element)
{
    int64_t offsetGmWithoutC = i * N_ + perCoreN_ * blockIdx + k * perLoopN_;
    int64_t offsetLocal = Align(element, FLOAT_SIZE);
    LocalTensor<float> intermediateTensor = intermediateBuf_.Get<float>();
    LocalTensor<float> inputTensor = inputBuf_.Get<float>();
    LocalTensor<float> vCovar = intermediateTensor;
    Duplicate<float>(vCovar, 0.0f, offsetLocal * VCOVAR_LEN);
    LocalTensor<float> meansLocal = inputTensor;
    DataCopyNGm2Local(meansLocal, meansGm_[i * MEANS_ELEMENT * N_ + perLoopN_ * k + perCoreN_ * blockIdx], element,
                      MEANS_ELEMENT);

    LocalTensor<float> quatsLocal = inputTensor[QUATS_OFFSET * offsetLocal];
    LocalTensor<float> rotmat = intermediateTensor[ROT_OFFSET * offsetLocal];
    DataCopyNGm2Local(quatsLocal, quatsGm_[i * QUAT_ELEMENT * N_ + perLoopN_ * k + perCoreN_ * blockIdx], element,
                      QUAT_ELEMENT);

    LocalTensor<float> scalesLocal = inputTensor[SCALES_OFFSET * offsetLocal];
    LocalTensor<float> covars = intermediateTensor[COVARS_OFFSET * offsetLocal];
    DataCopyNGm2Local(scalesLocal, scalesGm_[i * SCALES_ELEMENT * N_ + perLoopN_ * k + perCoreN_ * blockIdx], element,
                      SCALES_ELEMENT);

    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
    QuatToRotmat(quatsLocal, rotmat, element);
    Duplicate<float>(covars, 0, offsetLocal * COVAR_LEN);
    PipeBarrier<PIPE_V>();
    QuatScaleToCovarPreci(rotmat, scalesLocal, covars, element);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> vColorsCullingSumLocal = intermediateTensor[VCOLORSCCULLINGSUM_OFFSET * offsetLocal];
    LocalTensor<float> vOpacitiesCullingSumLocal = intermediateTensor[VOPACITIESCCULLINGSUM_OFFSET * offsetLocal];

    Duplicate<float>(vColorsCullingSumLocal, 0.0f, offsetLocal * VCOLORSCCULLINGSUM_LEN);

    for (int64_t j = 0; j < C_; ++j) {
        orderedIndex = indexBuf_.Get<int32_t>();

        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        int64_t offsetNDim = offsetFilterTensor.GetValue(j + C_) + offsetFilterTensor.GetValue(j);

        CopyInFilter(i, j, k, element);

        int32_t cntPerLoop = CalcReverseFilterOffset(element);
        offsetFilterTensor.SetValue(j, offsetFilterTensor.GetValue(j) + cntPerLoop);

        int64_t offsetGm = i * C_ * N_ + j * N_ + perCoreN_ * blockIdx + k * perLoopN_;
        LocalTensor<float> conicsLocal = inputTensor[CONICS_OFFSET * offsetLocal];
        LocalTensor<float> vConicsLocal = inputTensor[VCONICS_OFFSET * offsetLocal];
        LocalTensor<float> vCovar2d = intermediateTensor[VCOVARS2D_OFFSET * offsetLocal];
        DataCopyNGm2Local(conicsLocal,
                          conicsGm_[(i * C_ * N_ + j * N_) * CONICS_ELEMENT + perLoopN_ * k + perCoreN_ * blockIdx],
                          element, CONICS_ELEMENT);
        ReverseFilter(vConicsLocal, vConicsGm_, i, j, element, CONICS_ELEMENT, offsetNDim, cntPerLoop);
        // 计算InverseVjp
        InverseVjp(conicsLocal, vConicsLocal, vCovar2d, element);
        LocalTensor<float> viewmatsLocal = inputTensor[VIEWMAT_OFFSET * offsetLocal];
        LocalTensor<float> Rt = RtBuf_.Get<float>();
        LocalTensor<float> R = Rt;
        LocalTensor<float> t = Rt[RMAT_LEN];
        DataCopyNGm2Local(viewmatsLocal, viewmatsGm_[(i * C_ + j) * VIEWMATS_ELEMENT], VIEWMATS_ELEMENT, 0);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        slice(viewmatsLocal, R, t);
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

        LocalTensor<float> meansC = intermediateTensor[MEANSC_OFFSET * offsetLocal];
        PosW2C(meansLocal, R, t, meansC, element);

        LocalTensor<float> covarsC = intermediateTensor[COVARSC_OFFSET * offsetLocal];

        CovarW2C(covars, R, covarsC, element);

        LocalTensor<float> KsLocal = inputTensor[KS_OFFSET * offsetLocal];
        LocalTensor<float> vMeans2dLocal = inputTensor[VMEANS2D_OFFSET * offsetLocal];
        LocalTensor<float> vDepthsLocal = inputTensor[VDEPTHS_OFFSET * offsetLocal];
        LocalTensor<float> vMeansC = intermediateTensor[VMEANSC_OFFSET * offsetLocal];
        LocalTensor<float> vCovarC = intermediateTensor[VCOVARC_OFFSET * offsetLocal];
        DataCopyNGm2Local(KsLocal, KsGm_[(i * C_ + j) * KS_ELEMENT], KS_ELEMENT, 0);

        ReverseFilter(vMeans2dLocal, vMeans2dGm_, i, j, element, VMEANS_CONSTDIM, offsetNDim, cntPerLoop);
        ReverseFilter(vDepthsLocal, vDepthsGm_, i, j, element, VDEPTHS_CONSTDIM, offsetNDim, cntPerLoop);

        // 计算PerspProjVjp
        PerspProjVjp(vCovar2d, covarsC, meansC, KsLocal, vMeans2dLocal, vDepthsLocal, vMeansC, vCovarC, element);
        LocalTensor<float> vR = inputTensor[VR_OFFSET * offsetLocal];
        // PosW2CVjp不输出vPW，在函数内直接用累加copyout到gm
        PosW2CVjp(R, t, meansLocal, vMeansC, vR, offsetGmWithoutC * VPW_ELEMENT, element, j);
        // CovarW2CVjp不输出vR，在函数内直接用累加copyout到gm
        CovarW2CVjp(R, covars, vCovarC, vR, vCovar, (i * C_ + j) * VR_ELEMENT, element);

        LocalTensor<float> vColorsCullingLocal = intermediateTensor[VCOLORSCULLING_OFFSET * offsetLocal];
        LocalTensor<float> vOpacitiesCullingLocal = intermediateTensor[VOPACITIESCULLING_OFFSET * offsetLocal];
        ReverseFilter(vColorsCullingLocal, vColorsCullingGm_, i, j, element, VCOLORSCULLING_ELEMENT, offsetNDim,
                      cntPerLoop);
        ReverseFilter(vOpacitiesCullingLocal, vOpacitiesCullingGm_, i, j, element, VOPACITIESCULLING_ELEMENT,
                      offsetNDim, cntPerLoop);
        PipeBarrier<PIPE_V>();

        if (hasComp_ == 1) {
            LocalTensor<float> compensationsLocal = intermediateTensor[COMPENSATION_OFFSET * offsetLocal];
            DataCopyNGm2Local(
                compensationsLocal,
                compensationsGm_[(i * C_ * N_ + j * N_) * COMPENSATIONS_ELEMENT + perLoopN_ * k + perCoreN_ * blockIdx],
                element, COMPENSATIONS_ELEMENT);
            PipeBarrier<PIPE_V>();
            Mul(vOpacitiesCullingLocal, compensationsLocal, vOpacitiesCullingLocal, element);
            PipeBarrier<PIPE_V>();
        }
        Add(vColorsCullingSumLocal, vColorsCullingSumLocal, vColorsCullingLocal, offsetLocal * VCOLORSCULLING_ELEMENT);
        Add(vOpacitiesCullingSumLocal, vOpacitiesCullingSumLocal, vOpacitiesCullingLocal, offsetLocal);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> vQuatsLocal = inputTensor[VQUATS_OFFSET * offsetLocal];
    LocalTensor<float> vScalesLocal = inputTensor[VSCALES_OFFSET * offsetLocal];
    QuatScaleToCovarVjp(quatsLocal, scalesLocal, rotmat, vCovar, vQuatsLocal, vScalesLocal, element);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    DataCopyNLocal2Gm(vQuatsGm_[offsetGmWithoutC * QUAT_ELEMENT], vQuatsLocal, element, QUAT_ELEMENT);
    DataCopyNLocal2Gm(vScalesGm_[offsetGmWithoutC * SCALES_ELEMENT], vScalesLocal, element, SCALES_ELEMENT);
    DataCopyNLocal2GmNoTrans(vColorsGm_[i * VCOLORSCULLING_ELEMENT * N_ + perLoopN_ * k + perCoreN_ * blockIdx],
                             vColorsCullingSumLocal, element, VCOLORSCULLING_ELEMENT);
    DataCopyNLocal2GmNoTrans(vOpacitiesGm_[i * N_ + perLoopN_ * k + perCoreN_ * blockIdx], vOpacitiesCullingSumLocal,
                             element, VOPACITIESCULLING_ELEMENT);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FullyFusedProjectionBwd::Process()
{
    copyInTensor = copyInQue_.AllocTensor<float>();
    offsetFilterTensor = offsetFilterBuf_.Get<int32_t>();
    for (int64_t i = 0; i < B_; ++i) {
        int64_t k = 0;
        Duplicate(offsetFilterTensor, 0, FILTERBUF_CONSTOFFSET * C_);
        CalcCntPerCore(i);
        for (; k < loopN_; ++k) {
            SubProcess(i, k, perLoopN_);
        }
        // 计算尾块
        if (lastLoopN_ > 0) {
            SubProcess(i, k, lastLoopN_);
        }
    }
    copyInQue_.FreeTensor(copyInTensor);
}

} // namespace FullyFusedProjectionBwdNs
#endif // FULLY_FUSED_PROJECTION_BWD_H