/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "simt_api/common_functions.h"
#include "apply_grid_exposure_vf.h"
using namespace AscendC;
using namespace std;

constexpr uint32_t SF_USE_CLAMP = (1 << 0);
constexpr uint32_t SF_USE_GRID = (1 << 2);

constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t FLOAT_SIZE = 4;
static constexpr Reg::CastTrait castTruncTrait = {Reg::RegLayout::ZERO,
    Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

class ColorExposureFwd {
public:
    __aicore__ inline ColorExposureFwd()
    {}

    __aicore__ inline void GetTilingData(const ColorExposureFwdTilingData *tiling_data)
    {
        batchSize_ = tiling_data->batchSize;
        cameraNum_ = tiling_data->cameraNum;
        gaussNum_ = tiling_data->gaussNum;
        width_ = static_cast<float>(tiling_data->width);
        height_ = static_cast<float>(tiling_data->height);
        gridWidth_ = tiling_data->gridWidth;
        gridHeight_ = tiling_data->gridHeight;
        gridDepth_ = tiling_data->gridDepth;
        clamp_ = ((tiling_data->supportFeatures & SF_USE_CLAMP) != 0);
        useGrid_ = ((tiling_data->supportFeatures & SF_USE_GRID) != 0);
        totalTaskNum_ = tiling_data->totalTaskNum;
        tailNum_ = tiling_data->tailNum;
        taskNumPerScore_ = tiling_data->taskNumPerScore;
        taskNumPerLcore_ = tiling_data->taskNumPerLcore;
        numScore_ = tiling_data->numScore;
        numLcore_ = tiling_data->numLcore;
        blockDim_ = tiling_data->blockDim;
        taskNumPerLoop_ = tiling_data->taskNumPerLoop;
        ubTotalSize_ = tiling_data->ubTotalSize;
    }

    __aicore__ inline void Init(GM_ADDR colors, GM_ADDR means2d, GM_ADDR exposure,
        GM_ADDR cnt, GM_ADDR colorsOut, GM_ADDR gridIndex, GM_ADDR gridWeight,
        const ColorExposureFwdTilingData *tiling_data, TPipe* pipe)
    {
        PreInit(tiling_data);
        this->pipe_ = pipe;
        
        uint64_t bcnSize = batchSize_ * cameraNum_ * gaussNum_;
        uint32_t exposureSize = gridWidth_ * gridHeight_ * gridDepth_;
        colorsGM_.SetGlobalBuffer((__gm__ float *)colors, bcnSize * 3);
        if (useGrid_) {
            means2dGM_.SetGlobalBuffer((__gm__ float *)means2d, bcnSize * 2);
            gridIndexGM_.SetGlobalBuffer((__gm__ int32_t *)gridIndex, bcnSize * 3);
            gridWeightGM_.SetGlobalBuffer((__gm__ float *)gridWeight, bcnSize * 3);
        }
        exposureGM_.SetGlobalBuffer((__gm__ float *)exposure, exposureSize * 12);
        cntGM_.SetGlobalBuffer((__gm__ int32_t *)cnt, batchSize_ * cameraNum_);
        colorsOutGM_.SetGlobalBuffer((__gm__ float *)colorsOut, bcnSize * 3);

        uint32_t sizePerLoop = taskNumPerLoop_ * FLOAT_SIZE;
        this->pipe_->InitBuffer(colorsQue_, 2, sizePerLoop * 3);
        this->pipe_->InitBuffer(colorsOutQue_, 2, sizePerLoop * 3);
        this->pipe_->InitBuffer(cntBuf_, batchSize_ * cameraNum_ * FLOAT_SIZE);
        if (useGrid_) {
            this->pipe_->InitBuffer(means2dQue_, 2, sizePerLoop * 2);
            this->pipe_->InitBuffer(gridExpBuf_, exposureSize * FLOAT_SIZE * 12);
            this->pipe_->InitBuffer(gridIndexBuf_, sizePerLoop * 3);
            this->pipe_->InitBuffer(greyBuf_, sizePerLoop);
            gridExpTensor_ = gridExpBuf_.Get<float>();
            gridIndexTensor_ = gridIndexBuf_.Get<int32_t>();
            greyTensor_ = greyBuf_.Get<float>();
        } else {
            this->pipe_->InitBuffer(expBuf_, exposureSize * FLOAT_SIZE * 12);
            expTensor_ = expBuf_.Get<float>();
        }
        cntTensor_ = cntBuf_.Get<int32_t>();
    }

    __aicore__ inline void Process()
    {
        CopyInGlobal();
        for (uint32_t batchIdx = 0; batchIdx < batchSize_; batchIdx++) {
            for (uint32_t cameraIdx = 0; cameraIdx < cameraNum_; cameraIdx++) {
                cnt_ = cntTensor_[batchIdx * cameraNum_ + cameraIdx].GetValue(0);
                for (uint32_t taskLoopIdx = 0; taskLoopIdx < taskLoop_; taskLoopIdx++) {
                    if (!ComputingTaskNum(taskLoopIdx)) {
                        break;
                    }
                    CopyIn(batchIdx, cameraIdx, taskLoopIdx);
                    ApplyExposure();
                    CopyOut(batchIdx, cameraIdx, taskLoopIdx);
                }
            }
        }
    }

private:
    __aicore__ inline void PreInit(const ColorExposureFwdTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex_ = GetBlockIdx();

        GetTilingData(tiling_data);
        if (this->blockIndex_ < numLcore_) {
            taskNumPerCore_ = taskNumPerLcore_;
            taskStartIndex_ = this->blockIndex_ * taskNumPerCore_;
        } else {
            taskNumPerCore_ = taskNumPerScore_;
            taskStartIndex_ = numLcore_ * taskNumPerLcore_ + (this->blockIndex_ - numLcore_) * taskNumPerCore_;
        }
        taskLoop_ = static_cast<int32_t>((taskNumPerCore_ + taskNumPerLoop_ - 1) / taskNumPerLoop_);
    }

    __aicore__ inline void CopyInGlobal()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        uint32_t expSize = gridWidth_ * gridHeight_ * gridDepth_;
        DataCopyExtParams copyParams{1, expSize * FLOAT_SIZE * 12, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        if (useGrid_) {
            DataCopyPad(gridExpTensor_, exposureGM_, copyParams, padParams);
        } else {
            DataCopyPad(expTensor_, exposureGM_, copyParams, padParams);
        }
        uint32_t cntSize = batchSize_ * cameraNum_ * FLOAT_SIZE;
        DataCopyExtParams copyCntParams{1, cntSize, 0, 0, 0};
        DataCopyPadExtParams<int32_t> padCntParams{false, 0, 0, 0};
        DataCopyPad(cntTensor_, cntGM_, copyCntParams, padCntParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline bool ComputingTaskNum(int32_t taskLoopIdx)
    {
        if (taskLoopIdx == taskLoop_ - 1) {
            taskNumPerCurLoop_ = taskNumPerCore_ - taskLoopIdx * taskNumPerLoop_;
        } else {
            taskNumPerCurLoop_ = taskNumPerLoop_;
        }
        isTailBlock_ = (taskLoopIdx == taskLoop_ - 1) && tailNum_ != 0 &&
            (this->blockIndex_ == blockDim_ - 1);

        uint64_t taskStart = taskStartIndex_ + taskLoopIdx * taskNumPerLoop_;
        uint64_t taskEnd = isTailBlock_ ? taskStart + taskNumPerCurLoop_ - tailNum_
            : taskStart + taskNumPerCurLoop_;
        if (taskEnd > cnt_ && cnt_ > taskStart) {
            uint32_t num = static_cast<uint32_t>(cnt_ - taskStart);
            validTaskNum_ = isTailBlock_ ? min(taskNumPerCurLoop_ - tailNum_, num) : num;
            isTailBlock_ = true; // force to tail block.
        } else if (cnt_ >= taskEnd) {
            validTaskNum_ = isTailBlock_ ? taskNumPerCurLoop_ - tailNum_ : taskNumPerCurLoop_;
        }
        return taskStart < cnt_;
    }

    __aicore__ inline void CopyIn(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t taskCopyinIndex = taskStartIndex_ + taskLoopIndex * taskNumPerLoop_;
        uint64_t baseOffset = batchIdx * cameraNum_ * gaussNum_ + cameraIdx * gaussNum_;

        LocalTensor<float> colorsTensor = colorsQue_.AllocTensor<float>();
        const uint32_t colorDim = 3;
        for (uint32_t i = 0; i < colorDim; i++) {
            uint64_t offset = baseOffset * colorDim + i * gaussNum_ + taskCopyinIndex;
            if (isTailBlock_) {
                DataCopyExtParams copyParams {
                    1, static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE), 0, 0, 0
                };
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                DataCopyPad(colorsTensor[i * taskNumPerCurLoop_], colorsGM_[offset], copyParams, padParams);
            } else {
                DataCopy(colorsTensor[i * taskNumPerCurLoop_], colorsGM_[offset], taskNumPerCurLoop_);
            }
        }
        colorsQue_.EnQue(colorsTensor);

        if (useGrid_) {
            LocalTensor<float> means2dTensor = means2dQue_.AllocTensor<float>();
            const uint32_t means2dDim = 2;
            for (uint32_t i = 0; i < means2dDim; i++) {
                uint64_t offset = baseOffset * means2dDim + i * gaussNum_ + taskCopyinIndex;
                if (isTailBlock_) {
                    DataCopyExtParams copyParams {
                        1, static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE), 0, 0, 0
                    };
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                    DataCopyPad(means2dTensor[i * taskNumPerCurLoop_], means2dGM_[offset], copyParams, padParams);                    
                } else {
                    DataCopy(means2dTensor[i * taskNumPerCurLoop_], means2dGM_[offset], taskNumPerCurLoop_);
                }
            }
            means2dQue_.EnQue(means2dTensor);
        }
    }

    __aicore__ inline void PrepareCalculate(LocalTensor<float> &colorsTensor,
        LocalTensor<float> &means2dTensor)
    {
        uint32_t len = taskNumPerCurLoop_;
        uint16_t tiles = (uint16_t)((len + 63u) >> 6);
        uint32_t remain = isTailBlock_ ? validTaskNum_ : taskNumPerCurLoop_;
        __ubuf__ float *colors = (__ubuf__ float *)colorsTensor.GetPhyAddr();
        __ubuf__ float *means2d = (__ubuf__ float *)means2dTensor.GetPhyAddr();
        __ubuf__ float *grey = (__ubuf__ float *)greyTensor_.GetPhyAddr();
        __ubuf__ int32_t *index = (__ubuf__ int32_t *)gridIndexTensor_.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> r, g, b, rx, ry, rz, tmp, one;
            Reg::RegTensor<int32_t> idx, idy, idz, tmpi;
            Reg::MaskReg mask, cmpMask;
            int32_t gw = static_cast<int32_t>(gridWidth_);
            int32_t gh = static_cast<int32_t>(gridHeight_);
            int32_t gd = static_cast<int32_t>(gridDepth_);
            float wf = gridWidth_ * 1.0f;
            float hf = gridHeight_ * 1.0f;
            float df = gridDepth_ * 1.0f;
            float invW = 1.0f / width_;
            float invH = 1.0f / height_;
            for (uint16_t t = 0; t < tiles; t++) {
                uint32_t off = (uint32_t)t * 64u;
                mask = Reg::UpdateMask<float>(remain); // auto update remain

                // load from ub to register
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(r, colors + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(g, colors + len + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(b, colors + 2u * len + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(rx, means2d + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(ry, means2d + len + off);

                // calculate grey
                Reg::Muls(rz, r, 0.299f, mask);
                Reg::Muls(tmp, g, 0.587f, mask);
                Reg::Add(rz, rz, tmp, mask);
                Reg::Muls(tmp, b, 0.114f, mask);
                Reg::Add(rz, rz, tmp, mask);
                
                // map grid
                Reg::Muls(rx, rx, invW, mask);
                Reg::Maxs(rx, rx, 0.0f, mask);
                Reg::Mins(rx, rx, 1.0f, mask);
                Reg::Muls(rx, rx, wf - 1.0f, mask);
                Reg::Muls(ry, ry, invH, mask);
                Reg::Maxs(ry, ry, 0.0f, mask);
                Reg::Mins(ry, ry, 1.0f, mask);
                Reg::Muls(ry, ry, hf - 1.0f, mask);
                Reg::Maxs(rz, rz, 0.0f, mask);
                Reg::Mins(rz, rz, 1.0f, mask);
                Reg::Muls(rz, rz, df - 1.0f, mask);

                // calculate weights
                Reg::Cast<int32_t, float, castTruncTrait>(idx, rx, mask);
                Reg::Cast<float, int32_t, castTruncTrait>(tmp, idx, mask);
                Reg::Sub(rx, rx, tmp, mask);
                Reg::Cast<int32_t, float, castTruncTrait>(idy, ry, mask);
                Reg::Cast<float, int32_t, castTruncTrait>(tmp, idy, mask);
                Reg::Sub(ry, ry, tmp, mask);
                Reg::Cast<int32_t, float, castTruncTrait>(idz, rz, mask);
                Reg::Cast<float, int32_t, castTruncTrait>(tmp, idz, mask);
                Reg::Sub(rz, rz, tmp, mask);

                // boundary adjust
                Reg::Duplicate(one, 1.0f, mask);
                Reg::Compares<int32_t, CMPMODE::EQ>(cmpMask, idx, gw - 1, mask);
                Reg::Adds(tmpi, idx, -1, mask);
                Reg::Select(rx, one, rx, cmpMask);
                Reg::Select(idx, tmpi, idx, cmpMask);
                Reg::Compares<int32_t, CMPMODE::EQ>(cmpMask, idy, gh - 1, mask);
                Reg::Adds(tmpi, idy, -1, mask);
                Reg::Select(ry, one, ry, cmpMask);
                Reg::Select(idy, tmpi, idy, cmpMask);
                Reg::Compares<int32_t, CMPMODE::EQ>(cmpMask, idz, gd - 1, mask);
                Reg::Adds(tmpi, idz, -1, mask);
                Reg::Select(rz, one, rz, cmpMask);
                Reg::Select(idz, tmpi, idz, cmpMask);

                // store from register to ub
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(means2d + off, rx, mask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(means2d + len + off, ry, mask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(grey + off, rz, mask);
                Reg::StoreAlign<int32_t>(index + off, idx, mask);
                Reg::StoreAlign<int32_t>(index + len + off, idy, mask);
                Reg::StoreAlign<int32_t>(index + 2 * len + off, idz, mask);
            }
        }
    }

    __aicore__ inline void ApplyGridExposure(LocalTensor<float> &colorsOutTensor)
    {
        LocalTensor<float> colorsTensor = colorsQue_.DeQue<float>();
        LocalTensor<float> means2dTensor = means2dQue_.DeQue<float>();
        PrepareCalculate(colorsTensor, means2dTensor);
        asc_vf_call<ApplyGridExposureVF<THREAD_NUM>>(
            dim3{THREAD_NUM, 1, 1},
            (__ubuf__ float *)means2dTensor.GetPhyAddr(),
            (__ubuf__ float *)greyTensor_.GetPhyAddr(),
            (__ubuf__ int32_t *)gridIndexTensor_.GetPhyAddr(),
            (__ubuf__ float *)gridExpTensor_.GetPhyAddr(),
            (__ubuf__ float *)colorsTensor.GetPhyAddr(),
            (__ubuf__ float *)colorsOutTensor.GetPhyAddr(),
            clamp_,
            isTailBlock_ ? validTaskNum_ : taskNumPerCurLoop_,
            taskNumPerCurLoop_,
            static_cast<int32_t>(gridHeight_),
            static_cast<int32_t>(gridDepth_)
        );
        colorsQue_.FreeTensor(colorsTensor);
        means2dQue_.EnQue(means2dTensor);
    }

    __aicore__ inline void ApplyGlobalExposure(LocalTensor<float> &colorsOutTensor)
    {
        uint32_t len = taskNumPerCurLoop_;
        uint16_t tiles = (uint16_t)((len + 63u) >> 6);
        uint32_t remain = isTailBlock_ ? validTaskNum_ : taskNumPerCurLoop_;
        LocalTensor<float> colorsTensor = colorsQue_.DeQue<float>();
        __ubuf__ float *colors = (__ubuf__ float *)colorsTensor.GetPhyAddr();
        __ubuf__ float *exposure = (__ubuf__ float *)expTensor_.GetPhyAddr();
        __ubuf__ float *output = (__ubuf__ float *)colorsOutTensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> r, g, b, r0, g0, b0, tmp;
            Reg::RegTensor<float> e0, e1, e2, e3;
            Reg::MaskReg mask;
            for (uint16_t t = 0; t < tiles; t++) {
                uint32_t off = (uint32_t)t * 64u;
                mask = Reg::UpdateMask<float>(remain); // auto update remain

                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(r, colors + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(g, colors + len + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(b, colors + 2 * len + off);

                // r = e0 * r + e1 * g + e2 * b + e3
                Reg::Duplicate(e0, exposure[0], mask);
                Reg::Duplicate(e1, exposure[1], mask);
                Reg::Duplicate(e2, exposure[2], mask);
                Reg::Duplicate(e3, exposure[3], mask);
                Reg::Mul(r0, e0, r, mask);
                Reg::Mul(tmp, e1, g, mask);
                Reg::Add(r0, r0, tmp, mask);
                Reg::Mul(tmp, e2, b, mask);
                Reg::Add(r0, r0, tmp, mask);
                Reg::Add(r0, e3, r0, mask);

                // g = e0 * r + e1 * g + e2 * b + e3
                Reg::Duplicate(e0, exposure[4], mask);
                Reg::Duplicate(e1, exposure[5], mask);
                Reg::Duplicate(e2, exposure[6], mask);
                Reg::Duplicate(e3, exposure[7], mask);
                Reg::Mul(g0, e0, r, mask);
                Reg::Mul(tmp, e1, g, mask);
                Reg::Add(g0, g0, tmp, mask);
                Reg::Mul(tmp, e2, b, mask);
                Reg::Add(g0, g0, tmp, mask);
                Reg::Add(g0, e3, g0, mask);

                // b = e0 * r + e1 * g + e2 * b + e3
                Reg::Duplicate(e0, exposure[8], mask);
                Reg::Duplicate(e1, exposure[9], mask);
                Reg::Duplicate(e2, exposure[10], mask);
                Reg::Duplicate(e3, exposure[11], mask);
                Reg::Mul(b0, e0, r, mask);
                Reg::Mul(tmp, e1, g, mask);
                Reg::Add(b0, b0, tmp, mask);
                Reg::Mul(tmp, e2, b, mask);
                Reg::Add(b0, b0, tmp, mask);
                Reg::Add(b0, e3, b0, mask);

                if (clamp_) {
                    Reg::Maxs(r0, r0, 0.0f, mask);
                    Reg::Maxs(g0, g0, 0.0f, mask);
                    Reg::Maxs(b0, b0, 0.0f, mask);
                }
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(output + off, r0, mask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(output + len + off, g0, mask);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(output + 2 * len + off, b0, mask);
            }
        }
        colorsQue_.FreeTensor(colorsTensor);
    }

    __aicore__ inline void ApplyExposure()
    {
        LocalTensor<float> colorsOutTensor = colorsOutQue_.AllocTensor<float>();
        if (useGrid_) {
            ApplyGridExposure(colorsOutTensor);
        } else {
            ApplyGlobalExposure(colorsOutTensor);
        }
        colorsOutQue_.EnQue(colorsOutTensor);
    }

    __aicore__ inline void CopyOutIndexAndWeight(uint64_t baseOffset, uint64_t taskCopyinIndex)
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        
        LocalTensor<float> means2dTensor = means2dQue_.DeQue<float>();
        const uint32_t channelDim = 3;
        const uint32_t weightDim = 3;
        uint64_t offset = 0;
        if (isTailBlock_) {
            DataCopyExtParams copyParams {
                1, static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE), 0, 0, 0
            };
            for (uint32_t i = 0; i < channelDim; i++) {
                offset = baseOffset * channelDim + i * gaussNum_ + taskCopyinIndex;
                DataCopyPad(gridIndexGM_[offset], gridIndexTensor_[i * taskNumPerCurLoop_], copyParams);
                if (i == channelDim - 1) {
                    DataCopyPad(gridWeightGM_[offset], greyTensor_, copyParams);
                } else {
                    DataCopyPad(gridWeightGM_[offset], means2dTensor[i * taskNumPerCurLoop_], copyParams);
                }
            }
        } else {
            for (uint32_t i = 0; i < channelDim; i++) {
                offset = baseOffset * channelDim + i * gaussNum_ + taskCopyinIndex;
                DataCopy(gridIndexGM_[offset], gridIndexTensor_[i * taskNumPerCurLoop_], taskNumPerCurLoop_);
                if (i == channelDim - 1) {
                    DataCopy(gridWeightGM_[offset], greyTensor_, taskNumPerCurLoop_);
                } else {
                    DataCopy(gridWeightGM_[offset], means2dTensor[i * taskNumPerCurLoop_], taskNumPerCurLoop_);                    
                }
            }
        }
        means2dQue_.FreeTensor(means2dTensor);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx, uint32_t cameraIdx, int32_t taskLoopIndex)
    {
        LocalTensor<float> colorsOutTensor = colorsOutQue_.DeQue<float>();
        uint64_t taskCopyinIndex = taskStartIndex_ + taskLoopIndex * taskNumPerLoop_;
        uint64_t baseOffset = batchIdx * cameraNum_ * gaussNum_ + cameraIdx * gaussNum_;
        const uint32_t colorDim = 3;
        uint64_t offset = 0;
        if (isTailBlock_) {
            DataCopyExtParams copyParams {
                1, static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE), 0, 0, 0
            };
            for (uint32_t i = 0; i < colorDim; i++) {
                offset = baseOffset * colorDim + i * gaussNum_ + taskCopyinIndex;
                DataCopyPad(colorsOutGM_[offset], colorsOutTensor[i * taskNumPerCurLoop_], copyParams);
            }
        } else {
            for (uint32_t i = 0; i < colorDim; i++) {
                offset = baseOffset * colorDim + i * gaussNum_ + taskCopyinIndex;
                DataCopy(colorsOutGM_[offset], colorsOutTensor[i * taskNumPerCurLoop_], taskNumPerCurLoop_);
            }
        }
        colorsOutQue_.FreeTensor(colorsOutTensor);

        if (useGrid_) {
            CopyOutIndexAndWeight(baseOffset, taskCopyinIndex);
        }
    }

private:
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> expBuf_, gridExpBuf_, cntBuf_;
    TBuf<TPosition::VECCALC> gridIndexBuf_, greyBuf_;
    TQue<QuePosition::VECIN, 1> colorsQue_, means2dQue_;
    TQue<QuePosition::VECOUT, 1> colorsOutQue_;
    LocalTensor<float> expTensor_, gridExpTensor_, greyTensor_;
    LocalTensor<int32_t> gridIndexTensor_, cntTensor_;

    GlobalTensor<float> colorsGM_;
    GlobalTensor<float> means2dGM_;
    GlobalTensor<float> exposureGM_;
    GlobalTensor<int32_t> cntGM_;
    GlobalTensor<float> colorsOutGM_;
    GlobalTensor<int32_t> gridIndexGM_;
    GlobalTensor<float> gridWeightGM_;

    uint32_t batchSize_, cameraNum_, gaussNum_, gridWidth_, gridHeight_, gridDepth_;
    float width_, height_;
    bool useGrid_, clamp_, isTailBlock_;
    uint32_t totalTaskNum_, tailNum_, taskNumPerScore_, taskNumPerLcore_, cnt_, validTaskNum_;
    uint32_t numScore_, numLcore_, blockDim_, ubTotalSize_;
    uint64_t blockIndex_;
    uint32_t taskNumPerLoop_, taskNumPerCurLoop_, taskNumPerCore_, taskStartIndex_, taskLoop_;
};

extern "C" __global__ __aicore__ void color_exposure_fwd(GM_ADDR colors, GM_ADDR means2d, GM_ADDR exposure, GM_ADDR cnt,
    GM_ADDR colorsOut, GM_ADDR gridIndex, GM_ADDR gridWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        ColorExposureFwd op;
        op.Init(colors, means2d, exposure, cnt, colorsOut, gridIndex, gridWeight,
            &tiling_data, &pipe);
        op.Process();
    }
}