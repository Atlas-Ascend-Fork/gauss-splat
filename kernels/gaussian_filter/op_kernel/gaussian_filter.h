/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GAUSSIAN_FILTER_H
#define GAUSSIAN_FILTER_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "gaussian_filter_common.h"

namespace GaussianFilterNs {
using namespace AscendC;

template <const bool ISCOMPEXIST> class GaussianFilter {
public:
    __aicore__ inline GaussianFilter(){};
    __aicore__ inline void Init(GM_ADDR means, GM_ADDR colors, GM_ADDR det, GM_ADDR opacities, GM_ADDR means2d,
                                GM_ADDR depths, GM_ADDR radius, GM_ADDR conics, GM_ADDR covars2d, GM_ADDR compensations,
                                GM_ADDR meansCulling, GM_ADDR colorsCulling, GM_ADDR means2dCulling,
                                GM_ADDR depthsCulling, GM_ADDR radiusCulling, GM_ADDR covars2dCulling,
                                GM_ADDR conicsCulling, GM_ADDR opacitiesCulling, GM_ADDR filter, GM_ADDR cnt,
                                GM_ADDR workspace, TPipe *Ppipe, const GaussianFilterTilingData *tilingData);
    __aicore__ inline void Process();

private:
    template <typename T>
    __aicore__ inline void DataCopyIn(LocalTensor<T> local, GlobalTensor<T> gm, int64_t b, int64_t c, int64_t n,
                                      int64_t col, int32_t element);
    template <typename T>
    __aicore__ inline void DataCopyOut(GlobalTensor<T> gm, LocalTensor<T> local, int64_t col, int64_t offset);
    __aicore__ inline void CopyInFilterSource(int64_t b, int64_t c, int64_t n, int32_t element);
    __aicore__ inline void CalcFilter(int32_t element);
    __aicore__ inline int32_t CalcCntPerLoop(int32_t element);
    __aicore__ inline void CopyOutFilter(int64_t b, int64_t c, int64_t n, int32_t element);
    __aicore__ inline void CopyInFilter(int64_t b, int64_t c, int64_t n, int32_t element);
    __aicore__ inline void ProcessMeans2dAndRadius(int64_t b, int64_t c, int64_t n, int32_t element,
                                                   int64_t offsetNDim);
    __aicore__ inline void ProcessMeansAndDepth(int64_t b, int64_t c, int64_t n, int32_t element, int64_t offsetNDim);
    __aicore__ inline void ProcessCovars2d(int64_t b, int64_t c, int64_t n, int32_t element, int64_t offsetNDim);
    __aicore__ inline void ProcessColors(int64_t b, int64_t c, int64_t n, int32_t element, int64_t offsetNDim);
    __aicore__ inline void ProcessConics(int64_t b, int64_t c, int64_t n, int32_t element, int64_t offsetNDim);
    __aicore__ inline void ProcessOpacities(int64_t b, int64_t c, int64_t n, int32_t element, int64_t offsetNDim);
    __aicore__ inline void ProcessOpacitiesAndCompensations(int64_t b, int64_t c, int64_t n, int32_t element,
                                                            int64_t offsetNDim);
    __aicore__ inline void CopyOutCnt(GlobalTensor<int32_t> gm, int32_t cnt);
    __aicore__ inline void SubProcess(int64_t i, int64_t j);

    int64_t B_ = 0;
    int64_t C_ = 0;
    int64_t N_ = 0;
    int64_t width_ = 0;
    int64_t height_ = 0;
    float nearPlane_ = 0;
    float farPlane_ = 0;
    int64_t perCoreN_ = 0;
    int64_t curCoreN_ = 0;
    int64_t perLoopN_ = 0;
    int64_t lastLoopN_ = 0;
    int64_t perLoopMaxN_ = 0;
    int64_t loopN_ = 0;
    int64_t AlignN_ = 0;
    int64_t blockIdx_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t bufferLen_ = 0;
    int32_t offsetFilterCore_ = 0;
    int64_t offsetGm_ = 0;
    int64_t copyOutLen_ = 0;

    LocalTensor<int32_t> cntPreSum_;
    LocalTensor<int32_t> cntPerLoops_;

    GlobalTensor<float> meansGm_;
    GlobalTensor<float> colorsGm_;
    GlobalTensor<float> detGm_;
    GlobalTensor<float> opacitiesGm_;
    GlobalTensor<float> compensationsGm_;
    GlobalTensor<float> means2dGm_;
    GlobalTensor<float> depthsGm_;
    GlobalTensor<float> radiusGm_;
    GlobalTensor<float> conicsGm_;
    GlobalTensor<float> covars2dGm_;
    GlobalTensor<float> meansCullingGm_;
    GlobalTensor<float> colorsCullingGm_;
    GlobalTensor<float> means2dCullingGm_;
    GlobalTensor<float> depthsCullingGm_;
    GlobalTensor<float> radiusCullingGm_;
    GlobalTensor<float> covars2dCullingGm_;
    GlobalTensor<float> conicsCullingGm_;
    GlobalTensor<float> opacitiesCullingGm_;
    GlobalTensor<uint8_t> filterGm_;
    GlobalTensor<int32_t> cntGm_;
    GlobalTensor<int32_t> cntPerCoreGm_;
    LocalTensor<uint8_t> filter_;

    TQue<QuePosition::VECIN, 1> inQue0_;
    TQue<QuePosition::VECIN, 1> inQue1_;
    TQue<QuePosition::VECIN, 1> inQue2_;
    TQue<QuePosition::VECIN, 1> filterQue_;

    TQue<QuePosition::VECOUT, 1> outQue0_;
    TQue<QuePosition::VECOUT, 1> outQue1_;
    TQue<QuePosition::VECOUT, 1> outQue2_;

    TBuf<TPosition::VECCALC> calBuf_;
};

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::Init(
    GM_ADDR means, GM_ADDR colors, GM_ADDR det, GM_ADDR opacities, GM_ADDR means2d, GM_ADDR depths, GM_ADDR radius,
    GM_ADDR conics, GM_ADDR covars2d, GM_ADDR compensations, GM_ADDR meansCulling, GM_ADDR colorsCulling,
    GM_ADDR means2dCulling, GM_ADDR depthsCulling, GM_ADDR radiusCulling, GM_ADDR covars2dCulling,
    GM_ADDR conicsCulling, GM_ADDR opacitiesCulling, GM_ADDR filter, GM_ADDR cnt, GM_ADDR workspace, TPipe *Ppipe,
    const GaussianFilterTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    B_ = tilingData->batchNum;
    C_ = tilingData->cameraNum;
    N_ = tilingData->gaussNum;

    width_ = tilingData->width;
    height_ = tilingData->height;
    nearPlane_ = tilingData->nearPlane;
    farPlane_ = tilingData->farPlane;

    perCoreN_ = tilingData->blockLength;
    perLoopMaxN_ = tilingData->perloopNum;
    needCoreNum_ = tilingData->needCoreNum;
    // 本核需要处理多长的N
    curCoreN_ = perCoreN_;
    if (blockIdx_ == needCoreNum_ - 1) {
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
    colorsGm_.SetGlobalBuffer((__gm__ float *)colors);
    detGm_.SetGlobalBuffer((__gm__ float *)det);
    opacitiesGm_.SetGlobalBuffer((__gm__ float *)opacities);
    compensationsGm_.SetGlobalBuffer((__gm__ float *)compensations);
    means2dGm_.SetGlobalBuffer((__gm__ float *)means2d);
    depthsGm_.SetGlobalBuffer((__gm__ float *)depths);
    radiusGm_.SetGlobalBuffer((__gm__ float *)radius);
    conicsGm_.SetGlobalBuffer((__gm__ float *)conics);
    covars2dGm_.SetGlobalBuffer((__gm__ float *)covars2d);

    meansCullingGm_.SetGlobalBuffer((__gm__ float *)meansCulling);
    colorsCullingGm_.SetGlobalBuffer((__gm__ float *)colorsCulling);
    means2dCullingGm_.SetGlobalBuffer((__gm__ float *)means2dCulling);
    depthsCullingGm_.SetGlobalBuffer((__gm__ float *)depthsCulling);
    radiusCullingGm_.SetGlobalBuffer((__gm__ float *)radiusCulling);
    covars2dCullingGm_.SetGlobalBuffer((__gm__ float *)covars2dCulling);
    conicsCullingGm_.SetGlobalBuffer((__gm__ float *)conicsCulling);
    opacitiesCullingGm_.SetGlobalBuffer((__gm__ float *)opacitiesCulling);
    filterGm_.SetGlobalBuffer((__gm__ uint8_t *)filter);
    cntGm_.SetGlobalBuffer((__gm__ int32_t *)cnt);
    cntPerCoreGm_.SetGlobalBuffer((__gm__ int32_t *)(workspace));

    Ppipe->InitBuffer(inQue0_, 1, QUE_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(inQue1_, 1, QUE_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(inQue2_, 1, 1 * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(filterQue_, 1, 1 * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(outQue0_, 1, QUE_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(outQue1_, 1, QUE_LEN * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(outQue2_, 1, 1 * AlignBytes(perLoopN_, sizeof(float)));

    Ppipe->InitBuffer(calBuf_,
                      CALBUF_LEN1 * AlignBytes(perLoopN_, sizeof(float)) + CALBUF_LEN2 * MAX_CORE_NUM * INT32_SIZE);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::SubProcess(int64_t i, int64_t j)
{
    int32_t cntPerCore = 0;
    int64_t k = 0;
    int64_t offset16 = Align256(perLoopN_, INT16_SIZE);
    LocalTensor calTensor = calBuf_.Get<int32_t>();
    Duplicate(calTensor, 0, CALBUF_LEN1 * Align(perLoopN_, sizeof(float)) + CALBUF_LEN2 * MAX_CORE_NUM);
    cntPerLoops_ = calTensor[offset16 * CNTPERLOOP];
    cntPreSum_ = cntPerLoops_[MAX_CORE_NUM];
    offsetFilterCore_ = 0;
    for (; k < loopN_; ++k) {
        CopyInFilterSource(i, j, k, perLoopN_);
        CalcFilter(perLoopN_);
        int32_t cntPerLoop = CalcCntPerLoop(perLoopN_);
        CopyOutFilter(i, j, k, perLoopN_);
        cntPerCore += cntPerLoop;
        cntPerLoops_.SetValue(k, cntPerLoop);
        cntPreSum_.SetValue(k, cntPerCore);
    }
    if (lastLoopN_ != 0) {
        CopyInFilterSource(i, j, k, lastLoopN_);
        CalcFilter(lastLoopN_);
        int32_t cntPerLoop = CalcCntPerLoop(lastLoopN_);
        CopyOutFilter(i, j, k, lastLoopN_);
        cntPerCore += cntPerLoop;
        cntPerLoops_.SetValue(k, cntPerLoop);
        cntPreSum_.SetValue(k, cntPerCore);
    }
    CopyOutCnt(cntPerCoreGm_[blockIdx_], cntPerCore);
    SyncAll();

    LocalTensor cntPerCoreLocal = calTensor[0];
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(MAX_CORE_NUM * sizeof(int32_t)), 0,
                                     0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(cntPerCoreLocal, cntPerCoreGm_, dataCopyParams, dataCopyPadParams);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    for (k = 0; k < blockIdx_; ++k) {
        offsetFilterCore_ += cntPerCoreLocal.GetValue(k);
    }
    if (blockIdx_ == needCoreNum_ - 1) {
        cntGm_.SetValue(i * C_ + j, offsetFilterCore_ + cntPerCoreLocal.GetValue(blockIdx_));
    }

    int64_t offsetFilterLoop = 0;
    for (k = 0; k < loopN_; ++k) {
        if (k > 0) {
            offsetFilterLoop = cntPreSum_.GetValue(k - 1);
        }
        copyOutLen_ = cntPerLoops_.GetValue(k);
        int64_t offsetNDim = offsetFilterCore_ + offsetFilterLoop;

        CopyInFilter(i, j, k, perLoopN_);
        ProcessMeans2dAndRadius(i, j, k, perLoopN_, offsetNDim);
        ProcessMeansAndDepth(i, j, k, perLoopN_, offsetNDim);
        ProcessCovars2d(i, j, k, perLoopN_, offsetNDim);
        ProcessColors(i, j, k, perLoopN_, offsetNDim);
        ProcessConics(i, j, k, perLoopN_, offsetNDim);
        if constexpr (ISCOMPEXIST) {
            ProcessOpacitiesAndCompensations(i, j, k, perLoopN_, offsetNDim);
        } else {
            ProcessOpacities(i, j, k, perLoopN_, offsetNDim);
        }
        filterQue_.EnQue<uint8_t>(filter_);
        filter_ = filterQue_.DeQue<uint8_t>();
        filterQue_.FreeTensor(filter_);
    }
    if (lastLoopN_ != 0) {
        if (k > 0) {
            offsetFilterLoop = cntPreSum_.GetValue(k - 1);
        }
        copyOutLen_ = cntPerLoops_.GetValue(k);
        int64_t offsetNDim = offsetFilterCore_ + offsetFilterLoop;
        CopyInFilter(i, j, k, lastLoopN_);
        ProcessMeans2dAndRadius(i, j, k, lastLoopN_, offsetNDim);
        ProcessMeansAndDepth(i, j, k, lastLoopN_, offsetNDim);
        ProcessCovars2d(i, j, k, lastLoopN_, offsetNDim);
        ProcessColors(i, j, k, lastLoopN_, offsetNDim);
        ProcessConics(i, j, k, lastLoopN_, offsetNDim);
        if constexpr (ISCOMPEXIST) {
            ProcessOpacitiesAndCompensations(i, j, k, lastLoopN_, offsetNDim);
        } else {
            ProcessOpacities(i, j, k, lastLoopN_, offsetNDim);
        }
        filterQue_.EnQue<uint8_t>(filter_);
        filter_ = filterQue_.DeQue<uint8_t>();
        filterQue_.FreeTensor(filter_);
    }
}

template <const bool ISCOMPEXIST> __aicore__ inline void GaussianFilter<ISCOMPEXIST>::Process()
{
    for (int64_t i = 0; i < B_; ++i) {
        for (int64_t j = 0; j < C_; ++j) {
            SubProcess(i, j);
        }
    }
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::CopyOutCnt(GlobalTensor<int32_t> gm, int32_t cnt)
{
    LocalTensor<int32_t> cntTensor = outQue0_.AllocTensor<int32_t>();
    cntTensor.SetValue(0, cnt);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(gm, cntTensor, dataCopyParams);
    outQue0_.FreeTensor(cntTensor);
}

template <const bool ISCOMPEXIST>
template <typename T>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::DataCopyIn(LocalTensor<T> local, GlobalTensor<T> gm,
                                                                     int64_t b, int64_t c, int64_t n, int64_t col,
                                                                     int32_t element)
{
    uint32_t len = element * sizeof(T);
    uint32_t stride = (N_ - element) * sizeof(T);
    int64_t offsetGm = (b * C_ * N_ + c * N_) * col + perCoreN_ * blockIdx_ + n * perLoopN_;
    if (c == -1) {
        offsetGm = b * N_ * col + perCoreN_ * blockIdx_ + n * perLoopN_;
    }
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(col), static_cast<uint32_t>(len), stride, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(local, gm[offsetGm], dataCopyParams, dataCopyPadParams);
}

template <const bool ISCOMPEXIST>
template <typename T>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::DataCopyOut(GlobalTensor<T> gm, LocalTensor<T> local,
                                                                      int64_t col, int64_t offset)
{
    uint32_t len = copyOutLen_ * sizeof(T);
    uint32_t stride = (N_ - copyOutLen_) * sizeof(T);
    if (col == 1) {
        stride = 0;
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = static_cast<uint16_t>(col);
    dataCopyParams.blockLen = static_cast<uint32_t>(len);
    dataCopyParams.srcStride = (offset - Align(copyOutLen_, FLOAT_SIZE)) / ALIGN_8;
    dataCopyParams.dstStride = stride;

    DataCopyPad(gm, local, dataCopyParams);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::CopyInFilterSource(int64_t b, int64_t c, int64_t n,
                                                                             int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> in0Local = inQue0_.AllocTensor<float>();
    LocalTensor<float> means2dLocal = in0Local;
    LocalTensor<float> radiusLocal = in0Local[RADIUS_OFFSET * offset];

    LocalTensor<float> in1Local = inQue1_.AllocTensor<float>();
    LocalTensor<float> detLocal = in1Local;
    LocalTensor<float> depthsLocal = in1Local[DPETHS_OFFSET * offset];

    /* datacopy means2d radius depths */
    DataCopyIn(means2dLocal, means2dGm_, b, c, n, MEANS2D_DIM, element);
    DataCopyIn(radiusLocal, radiusGm_, b, c, n, MEANS2D_DIM, element);
    DataCopyIn(detLocal, detGm_, b, c, n, DET_DIM, element);
    DataCopyIn(depthsLocal, depthsGm_, b, c, n, DEPTHS_DIM, element);

    inQue0_.EnQue<float>(in0Local);
    inQue1_.EnQue<float>(in1Local);
}

template <const bool ISCOMPEXIST> __aicore__ inline void GaussianFilter<ISCOMPEXIST>::CalcFilter(int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    int64_t offset16 = Align256(element, INT16_SIZE);
    LocalTensor<float> in0Local = inQue0_.DeQue<float>();
    LocalTensor<float> means2dLocal = in0Local;
    LocalTensor<float> radiusLocal = in0Local[RADIUS_OFFSET * offset];
    LocalTensor<float> in1Local = inQue1_.DeQue<float>();
    LocalTensor<float> detLocal = in1Local;
    LocalTensor<float> depthsLocal = in1Local[DPETHS_OFFSET * offset];

    LocalTensor<uint8_t> filter = outQue0_.AllocTensor<uint8_t>();

    LocalTensor<float> tempTensor = calBuf_.Get<float>()[offset16];

    LocalTensor<float> means2dX = means2dLocal;
    LocalTensor<float> means2dY = means2dLocal[offset];
    LocalTensor<float> radiusX = radiusLocal;
    LocalTensor<float> radiusY = radiusLocal[offset];

    /* calc filter */
    LocalTensor<uint16_t> mask = calBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> maskTmp = mask[offset16];
    LocalTensor<uint8_t> maskUINT8 = mask.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskTmpUINT8 = maskTmp.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_V>();
    Add(tempTensor, means2dX, radiusX, element);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskUINT8, tempTensor, 0.0f, CMPMODE::GT, offset16);
    PipeBarrier<PIPE_V>();

    Sub(tempTensor, means2dX, radiusX, element);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTmpUINT8, tempTensor, static_cast<float>(width_), CMPMODE::LT, offset16);
    PipeBarrier<PIPE_V>();
    And(mask, mask, maskTmp, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));

    PipeBarrier<PIPE_V>();
    Add(tempTensor, means2dY, radiusY, element);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTmpUINT8, tempTensor, 0.0f, CMPMODE::GT, offset16);
    PipeBarrier<PIPE_V>();
    And(mask, mask, maskTmp, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));

    PipeBarrier<PIPE_V>();
    Sub(tempTensor, means2dY, radiusY, element);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTmpUINT8, tempTensor, static_cast<float>(height_), CMPMODE::LT, offset16);
    PipeBarrier<PIPE_V>();
    And(mask, mask, maskTmp, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));

    PipeBarrier<PIPE_V>();
    CompareScalar(maskTmpUINT8, detLocal, 0.0f, CMPMODE::GT, offset16);
    PipeBarrier<PIPE_V>();
    And(mask, mask, maskTmp, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));

    PipeBarrier<PIPE_V>();
    CompareScalar(maskTmpUINT8, depthsLocal, nearPlane_, CMPMODE::GT, offset16);
    PipeBarrier<PIPE_V>();
    And(mask, mask, maskTmp, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));

    PipeBarrier<PIPE_V>();
    CompareScalar(maskTmpUINT8, depthsLocal, farPlane_, CMPMODE::LT, offset16);
    PipeBarrier<PIPE_V>();
    And(filter.ReinterpretCast<uint16_t>(), mask, maskTmp, Ceil(offset16, AscendCUtils::GetBitSize(sizeof(uint16_t))));

    outQue0_.EnQue<uint8_t>(filter);
    inQue0_.FreeTensor(in0Local);
    inQue1_.FreeTensor(in1Local);
}

template <const bool ISCOMPEXIST>
__aicore__ inline int32_t GaussianFilter<ISCOMPEXIST>::CalcCntPerLoop(int32_t element)
{
    uint64_t cnt = 0;
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<uint8_t> filterLocal = outQue0_.DeQue<uint8_t>();
    LocalTensor<float> tmpTensor0 = calBuf_.Get<float>();
    LocalTensor<float> tmpTensor1 = tmpTensor0[offset];

    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;

    LocalTensor<uint32_t> bitMask = filterLocal.ReinterpretCast<uint32_t>();
    GatherMask(tmpTensor0, tmpTensor1, bitMask, true, element, params, cnt);
    outQue0_.EnQue<uint8_t>(filterLocal);
    return static_cast<int32_t>(cnt);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::CopyOutFilter(int64_t b, int64_t c, int64_t n,
                                                                        int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<uint8_t> filterLocal = outQue0_.DeQue<uint8_t>();
    /* copyout filter */
    uint32_t len = Ceil(element, ALIGN_8) * sizeof(uint8_t);
    int64_t offsetGm = b * C_ * (Ceil(N_, ALIGN_8)) + c * (Ceil(N_, ALIGN_8)) + perCoreN_ / ALIGN_8 * blockIdx_ +
                       perLoopN_ / ALIGN_8 * n;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len), 0, 0, 0};
    DataCopyPad(filterGm_[offsetGm], filterLocal, dataCopyParams);
    outQue0_.FreeTensor(filterLocal);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::CopyInFilter(int64_t b, int64_t c, int64_t n, int32_t element)
{
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    int64_t offset = Align(element, FLOAT_SIZE);
    filter_ = filterQue_.AllocTensor<uint8_t>();
    /* copyout filter */
    uint32_t len = Ceil(element, ALIGN_8) * sizeof(uint8_t);
    int64_t offsetGm = b * C_ * (Ceil(N_, ALIGN_8)) + c * (Ceil(N_, ALIGN_8)) + perCoreN_ / ALIGN_8 * blockIdx_ +
                       perLoopN_ / ALIGN_8 * n;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(len), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(filter_, filterGm_[offsetGm], dataCopyParams, dataCopyPadParams);
    filterQue_.EnQue<uint8_t>(filter_);
    filter_ = filterQue_.DeQue<uint8_t>();
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessMeans2dAndRadius(int64_t b, int64_t c, int64_t n,
                                                                                  int32_t element, int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> in0Local = inQue0_.AllocTensor<float>();
    LocalTensor<float> means2dLocal = in0Local;
    LocalTensor<float> radiusLocal = in0Local[RADIUS_OFFSET * offset];

    DataCopyIn(means2dLocal, means2dGm_, b, c, n, MEANS2D_DIM, element);
    DataCopyIn(radiusLocal, radiusGm_, b, c, n, RADIUS_DIM, element);

    inQue0_.EnQue<float>(in0Local);

    /* calc */
    in0Local = inQue0_.DeQue<float>();
    means2dLocal = in0Local;
    radiusLocal = in0Local[RADIUS_OFFSET * offset];
    LocalTensor<float> out0Local = outQue0_.AllocTensor<float>();
    LocalTensor<float> means2dCullingLocal = out0Local;
    LocalTensor<float> radiusCullingLocal = out0Local[RADIUS_OFFSET * offset];
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(means2dCullingLocal[offset * 0], means2dLocal[offset * 0], bitMask, true, element, params, cnt);
    GatherMask(means2dCullingLocal[offset * 1], means2dLocal[offset * 1], bitMask, true, element, params, cnt);
    GatherMask(radiusCullingLocal[offset * 0], radiusLocal[offset * 0], bitMask, true, element, params, cnt);
    GatherMask(radiusCullingLocal[offset * 1], radiusLocal[offset * 1], bitMask, true, element, params, cnt);
    outQue0_.EnQue<float>(out0Local);
    inQue0_.FreeTensor(in0Local);

    /* copyout */
    out0Local = outQue0_.DeQue<float>();
    means2dCullingLocal = out0Local;
    radiusCullingLocal = out0Local[RADIUS_OFFSET * offset];

    DataCopyOut(means2dCullingGm_[(b * C_ * N_ + c * N_) * MEANS2D_OFFSET + offsetNDim], means2dCullingLocal,
                MEANS2D_CULLING_DIM, offset);
    DataCopyOut(radiusCullingGm_[(b * C_ * N_ + c * N_) * RADIUS_OFFSET + offsetNDim], radiusCullingLocal,
                RADIUS_CULLING_DIM, offset);
    outQue0_.FreeTensor(out0Local);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessMeansAndDepth(int64_t b, int64_t c, int64_t n,
                                                                               int32_t element, int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> in1Local = inQue1_.AllocTensor<float>();
    LocalTensor<float> meansLocal = in1Local;
    LocalTensor<float> depthsLocal = in1Local[PROCESS_DPETHS_OFFSET * offset];

    DataCopyIn(meansLocal, meansGm_, b, -1, n, MEANS_DIM, element);
    DataCopyIn(depthsLocal, depthsGm_, b, c, n, DEPTHS_DIM, element);

    inQue1_.EnQue<float>(in1Local);

    /* calc */
    in1Local = inQue1_.DeQue<float>();
    meansLocal = in1Local;
    depthsLocal = in1Local[PROCESS_DPETHS_OFFSET * offset];
    LocalTensor<float> out1Local = outQue1_.AllocTensor<float>();
    LocalTensor<float> meansCullingLocal = out1Local;
    LocalTensor<float> depthsCullingLocal = out1Local[PROCESS_DPETHS_OFFSET * offset];
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(meansCullingLocal[offset * 0], meansLocal[offset * 0], bitMask, true, element, params, cnt);
    GatherMask(meansCullingLocal[offset * 1], meansLocal[offset * 1], bitMask, true, element, params, cnt);
    GatherMask(meansCullingLocal[offset * MEANSCULLING_OFFSET], meansLocal[offset * MEANSCULLING_OFFSET], bitMask, true,
               element, params, cnt);
    GatherMask(depthsCullingLocal[offset * 0], depthsLocal[offset * 0], bitMask, true, element, params, cnt);
    outQue1_.EnQue<float>(out1Local);
    inQue1_.FreeTensor(in1Local);

    /* copyout */
    out1Local = outQue1_.DeQue<float>();
    meansCullingLocal = out1Local;
    depthsCullingLocal = out1Local[PROCESS_DPETHS_OFFSET * offset];

    DataCopyOut(meansCullingGm_[(b * C_ * N_ + c * N_) * PROCESS_DPETHS_OFFSET + offsetNDim], meansCullingLocal,
                MEANS_CULLING_DIM, offset);
    DataCopyOut(depthsCullingGm_[(b * C_ * N_ + c * N_) * 1 + offsetNDim], depthsCullingLocal, DEPTHS_CULLING_DIM,
                offset);
    outQue1_.FreeTensor(out1Local);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessCovars2d(int64_t b, int64_t c, int64_t n,
                                                                          int32_t element, int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> covars2dLocal = inQue0_.AllocTensor<float>();

    DataCopyIn(covars2dLocal, covars2dGm_, b, c, n, COVARS2D_DIM, element);

    inQue0_.EnQue<float>(covars2dLocal);

    /* calc */
    covars2dLocal = inQue0_.DeQue<float>();

    LocalTensor<float> covars2dCullingLocal = outQue0_.AllocTensor<float>();
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(covars2dCullingLocal[offset * 0], covars2dLocal[offset * 0], bitMask, true, element, params, cnt);
    GatherMask(covars2dCullingLocal[offset * 1], covars2dLocal[offset * 1], bitMask, true, element, params, cnt);
    GatherMask(covars2dCullingLocal[offset * COVARS2DCULLING_OFFSET], covars2dLocal[offset * COVARS2DCULLING_OFFSET],
               bitMask, true, element, params, cnt);
    outQue0_.EnQue<float>(covars2dCullingLocal);
    inQue0_.FreeTensor(covars2dLocal);

    /* copyout */
    covars2dCullingLocal = outQue0_.DeQue<float>();

    DataCopyOut(covars2dCullingGm_[(b * C_ * N_ + c * N_) * COVARS2D_OFFSET + offsetNDim], covars2dCullingLocal,
                COVARS2D_CULLING_DIM, offset);
    outQue0_.FreeTensor(covars2dCullingLocal);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessColors(int64_t b, int64_t c, int64_t n,
                                                                        int32_t element, int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> colorsLocal = inQue1_.AllocTensor<float>();

    DataCopyIn(colorsLocal, colorsGm_, b, -1, n, COLORS_DIM, element);

    inQue1_.EnQue<float>(colorsLocal);

    /* calc */
    colorsLocal = inQue1_.DeQue<float>();

    LocalTensor<float> colorsCullingLocal = outQue1_.AllocTensor<float>();
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(colorsCullingLocal[offset * 0], colorsLocal[offset * 0], bitMask, true, element, params, cnt);
    GatherMask(colorsCullingLocal[offset * 1], colorsLocal[offset * 1], bitMask, true, element, params, cnt);
    GatherMask(colorsCullingLocal[offset * COLORCULLING_OFFSET], colorsLocal[offset * COLORCULLING_OFFSET], bitMask,
               true, element, params, cnt);
    outQue1_.EnQue<float>(colorsCullingLocal);
    inQue1_.FreeTensor(colorsLocal);

    /* copyout */
    colorsCullingLocal = outQue1_.DeQue<float>();

    DataCopyOut(colorsCullingGm_[(b * C_ * N_ + c * N_) * 3 + offsetNDim], colorsCullingLocal, COLORS_CULLING_DIM,
                offset);
    outQue1_.FreeTensor(colorsCullingLocal);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessConics(int64_t b, int64_t c, int64_t n,
                                                                        int32_t element, int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> conicsLocal = inQue0_.AllocTensor<float>();

    DataCopyIn(conicsLocal, conicsGm_, b, c, n, CONICS_DIM, element);

    inQue0_.EnQue<float>(conicsLocal);

    /* calc */
    conicsLocal = inQue0_.DeQue<float>();

    LocalTensor<float> conicsCullingLocal = outQue0_.AllocTensor<float>();
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(conicsCullingLocal[offset * 0], conicsLocal[offset * 0], bitMask, true, element, params, cnt);
    GatherMask(conicsCullingLocal[offset * 1], conicsLocal[offset * 1], bitMask, true, element, params, cnt);
    GatherMask(conicsCullingLocal[offset * CONICSCULLING_OFFSET], conicsLocal[offset * CONICSCULLING_OFFSET], bitMask,
               true, element, params, cnt);
    outQue0_.EnQue<float>(conicsCullingLocal);
    inQue0_.FreeTensor(conicsLocal);

    /* copyout */
    conicsCullingLocal = outQue0_.DeQue<float>();

    DataCopyOut(conicsCullingGm_[(b * C_ * N_ + c * N_) * CONICSCULLINGGM_OFFSET + offsetNDim], conicsCullingLocal,
                CONICS_CULLING_DIM, offset);
    outQue0_.FreeTensor(conicsCullingLocal);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessOpacities(int64_t b, int64_t c, int64_t n,
                                                                           int32_t element, int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> opacitiesLocal = inQue2_.AllocTensor<float>();

    DataCopyIn(opacitiesLocal, opacitiesGm_, b, -1, n, OPACITIES_DIM, element);

    inQue2_.EnQue<float>(opacitiesLocal);

    /* calc */
    opacitiesLocal = inQue2_.DeQue<float>();

    LocalTensor<float> opacitiesCullingLocal = outQue2_.AllocTensor<float>();
    PipeBarrier<PIPE_V>();
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(opacitiesCullingLocal[offset * 0], opacitiesLocal[offset * 0], bitMask, true, element, params, cnt);
    outQue2_.EnQue<float>(opacitiesCullingLocal);
    inQue2_.FreeTensor(opacitiesLocal);

    /* copyout */
    opacitiesCullingLocal = outQue2_.DeQue<float>();

    DataCopyOut(opacitiesCullingGm_[(b * C_ * N_ + c * N_) * 1 + offsetNDim], opacitiesCullingLocal,
                OPACITIES_CULLING_DIM, offset);
    outQue2_.FreeTensor(opacitiesCullingLocal);
}

template <const bool ISCOMPEXIST>
__aicore__ inline void GaussianFilter<ISCOMPEXIST>::ProcessOpacitiesAndCompensations(int64_t b, int64_t c,
                                                                                           int64_t n, int32_t element,
                                                                                           int64_t offsetNDim)
{
    /* copyin */
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> opacitiesLocal = inQue2_.AllocTensor<float>();
    LocalTensor<float> compensationsLocal = inQue1_.AllocTensor<float>();

    DataCopyIn(opacitiesLocal, opacitiesGm_, b, -1, n, OPACITIES_DIM, element);
    DataCopyIn(compensationsLocal, compensationsGm_, b, c, n, COMPENSATIONS_DIM, element);

    inQue2_.EnQue<float>(opacitiesLocal);
    inQue1_.EnQue<float>(compensationsLocal);

    /* calc */
    opacitiesLocal = inQue2_.DeQue<float>();
    compensationsLocal = inQue1_.DeQue<float>();

    LocalTensor<float> opacitiesCullingLocal = outQue2_.AllocTensor<float>();
    Mul(opacitiesLocal, opacitiesLocal, compensationsLocal, element);
    PipeBarrier<PIPE_V>();
    uint64_t cnt = 0;
    GatherMaskParams params;
    params.src0BlockStride = 1;
    params.repeatTimes = 1;
    params.src0RepeatStride = ALIGN_8;
    params.src1RepeatStride = 0;
    LocalTensor<uint32_t> bitMask = filter_.ReinterpretCast<uint32_t>();
    GatherMask(opacitiesCullingLocal[offset * 0], opacitiesLocal[offset * 0], bitMask, true, element, params, cnt);
    outQue2_.EnQue<float>(opacitiesCullingLocal);
    inQue2_.FreeTensor(opacitiesLocal);
    inQue1_.FreeTensor(compensationsLocal);

    /* copyout */
    opacitiesCullingLocal = outQue2_.DeQue<float>();

    DataCopyOut(opacitiesCullingGm_[(b * C_ * N_ + c * N_) * 1 + offsetNDim], opacitiesCullingLocal,
                OPACITIES_CULLING_DIM, offset);
    outQue2_.FreeTensor(opacitiesCullingLocal);
}

} // namespace GaussianFilterNs
#endif // GAUSSIAN_FILTER_H