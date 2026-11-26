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
#include "calc_color_exp_grad_vf.h"
using namespace AscendC;
using namespace std;

constexpr uint32_t SF_USE_CLAMP = (1 << 0);
constexpr uint32_t SF_CALC_EXP_GRAD = (1 << 1);
constexpr uint32_t SF_USE_GRID = (1 << 2);

constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t FLOAT_SIZE = 4;
class ColorExposureBwd {
public:
    __aicore__ inline ColorExposureBwd()
    {}

    __aicore__ inline void GetTilingData(const ColorExposureBwdTilingData *tiling_data)
    {
        batchSize_ = tiling_data->batchSize;
        cameraNum_ = tiling_data->cameraNum;
        gaussNum_ = tiling_data->gaussNum;
        gridWidth_ = tiling_data->gridWidth;
        gridHeight_ = tiling_data->gridHeight;
        gridDepth_ = tiling_data->gridDepth;
        clamp_ = ((tiling_data->supportFeatures & SF_USE_CLAMP) != 0);
        calcExpGrad_ = ((tiling_data->supportFeatures & SF_CALC_EXP_GRAD) != 0);
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

    __aicore__ inline void Init(GM_ADDR colors, GM_ADDR vColorsIn, GM_ADDR exposure,
        GM_ADDR cnt, GM_ADDR gridIndex, GM_ADDR gridWeight, GM_ADDR vColorsOut, GM_ADDR vExposure,
        const ColorExposureBwdTilingData *tiling_data, TPipe* pipe)
    {
        PreInit(tiling_data);
        this->pipe_ = pipe;

        uint64_t bcnSize = batchSize_ * cameraNum_ * gaussNum_;
        uint32_t exposureSize = gridWidth_ * gridHeight_ * gridDepth_;
        colorsGM_.SetGlobalBuffer((__gm__ float *)colors, bcnSize * 3);
        vColorsInGM_.SetGlobalBuffer((__gm__ float *)vColorsIn, bcnSize * 3);
        exposureGM_.SetGlobalBuffer((__gm__ float *)exposure, exposureSize * 12);
        vColorsOutGM_.SetGlobalBuffer((__gm__ float *)vColorsOut, bcnSize * 3);
        cntGM_.SetGlobalBuffer((__gm__ int32_t *)cnt, batchSize_ * cameraNum_);
        if (useGrid_) {
            gridIndexGM_.SetGlobalBuffer((__gm__ int32_t *)gridIndex, bcnSize * 3);
            gridWeightGM_.SetGlobalBuffer((__gm__ float *)gridWeight, bcnSize * 3);
        }
        if (calcExpGrad_) {
            vExposureGM_.SetGlobalBuffer((__gm__ float *)vExposure, exposureSize * 12);
        }

        uint32_t sizePerLoop = taskNumPerLoop_ * FLOAT_SIZE;
        this->pipe_->InitBuffer(colorsQue_, 2, sizePerLoop * 3);
        this->pipe_->InitBuffer(vColorsInQue_, 2, sizePerLoop * 3);
        this->pipe_->InitBuffer(vColorsOutQue_, 2, sizePerLoop * 3);
        this->pipe_->InitBuffer(expBuf_, exposureSize * FLOAT_SIZE * 12);
        this->pipe_->InitBuffer(cntBuf_, batchSize_ * cameraNum_ * FLOAT_SIZE);
        if (useGrid_) {
            this->pipe_->InitBuffer(gridIndexQue_, 2, sizePerLoop * 3);
            this->pipe_->InitBuffer(gridWeightQue_, 2, sizePerLoop * 3);
        }
        if (calcExpGrad_) {
            this->pipe_->InitBuffer(vExpBuf_, exposureSize * FLOAT_SIZE * 12);
            vExpTensor_ = vExpBuf_.Get<float>();
        }
        expTensor_ = expBuf_.Get<float>();
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
                    CalculateGrad();
                    CopyOut(batchIdx, cameraIdx, taskLoopIdx);
                }
            }
        }
    }

private:
    __aicore__ inline void PreInit(const ColorExposureBwdTilingData *tiling_data)
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
        uint32_t exposureSize = gridWidth_ * gridHeight_ * gridDepth_;
        DataCopyExtParams copyParams{1, exposureSize * FLOAT_SIZE * 12, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(expTensor_, exposureGM_, copyParams, padParams);
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

        // copy in colors and vColorsIn
        LocalTensor<float> colorsTensor = colorsQue_.AllocTensor<float>();
        LocalTensor<float> vColorsInTensor = vColorsInQue_.AllocTensor<float>();
        const uint32_t colorDim = 3;
        uint64_t offset = 0;
        for (uint32_t i = 0; i < colorDim; i++) {
            offset = baseOffset * colorDim + i * gaussNum_ + taskCopyinIndex;
            if (isTailBlock_) {
                DataCopyExtParams copyParams {
                    1, static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE), 0, 0, 0
                };
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                DataCopyPad(colorsTensor[i * taskNumPerCurLoop_], colorsGM_[offset], copyParams, padParams);
                DataCopyPad(vColorsInTensor[i * taskNumPerCurLoop_], vColorsInGM_[offset], copyParams, padParams);
            } else {
                DataCopy(colorsTensor[i * taskNumPerCurLoop_], colorsGM_[offset], taskNumPerCurLoop_);
                DataCopy(vColorsInTensor[i * taskNumPerCurLoop_], vColorsInGM_[offset], taskNumPerCurLoop_);
            }
        }
        colorsQue_.EnQue(colorsTensor);
        vColorsInQue_.EnQue(vColorsInTensor);

        // copy in index and weight for useGrid
        if (useGrid_) {
            LocalTensor<int32_t> indexTensor = gridIndexQue_.AllocTensor<int32_t>();
            LocalTensor<float> weightTensor = gridWeightQue_.AllocTensor<float>();
            const uint32_t channelDim = 3;
            for (uint32_t i = 0; i < channelDim; i++) {
                offset = baseOffset * channelDim + i * gaussNum_ + taskCopyinIndex;
                if (isTailBlock_) {
                    DataCopyExtParams copyParams {
                        1, static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE), 0, 0, 0
                    };
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                    DataCopyPadExtParams<int32_t> padIntParams{false, 0, 0, 0};
                    DataCopyPad(indexTensor[i * taskNumPerCurLoop_], gridIndexGM_[offset], copyParams, padIntParams);
                    DataCopyPad(weightTensor[i * taskNumPerCurLoop_], gridWeightGM_[offset], copyParams, padParams);
                } else {
                    DataCopy(indexTensor[i * taskNumPerCurLoop_], gridIndexGM_[offset], taskNumPerCurLoop_);
                    DataCopy(weightTensor[i * taskNumPerCurLoop_], gridWeightGM_[offset], taskNumPerCurLoop_);
                }
            }
            gridIndexQue_.EnQue(indexTensor);
            gridWeightQue_.EnQue(weightTensor);
        }
    }

    __aicore__ inline void CalculateGrad()
    {
        LocalTensor<float> colorsTensor = colorsQue_.DeQue<float>();
        LocalTensor<float> vColorsInTensor = vColorsInQue_.DeQue<float>();
        LocalTensor<float> vColorsOutTensor = vColorsOutQue_.AllocTensor<float>();
        uint32_t expSize = gridWidth_ * gridHeight_ * gridDepth_;
        if (calcExpGrad_) {
            Duplicate(vExpTensor_, 0.0f, expSize * 12); // reset to zeros
        }
        if (useGrid_) {
            LocalTensor<int32_t> indexTensor = gridIndexQue_.DeQue<int32_t>();
            LocalTensor<float> weightTensor = gridWeightQue_.DeQue<float>();
            asc_vf_call<CalcGradUseGrid<THREAD_NUM>>(
                dim3{THREAD_NUM, 1, 1},
                (__ubuf__ float *)colorsTensor.GetPhyAddr(),
                (__ubuf__ float *)vColorsInTensor.GetPhyAddr(),
                (__ubuf__ float *)vColorsOutTensor.GetPhyAddr(),
                (__ubuf__ float *)vExpTensor_.GetPhyAddr(),
                (__ubuf__ float *)expTensor_.GetPhyAddr(),
                (__ubuf__ float *)weightTensor.GetPhyAddr(),
                (__ubuf__ int32_t *)indexTensor.GetPhyAddr(),
                isTailBlock_ ? validTaskNum_ : taskNumPerCurLoop_,
                taskNumPerCurLoop_,
                static_cast<int32_t>(gridHeight_),
                static_cast<int32_t>(gridDepth_),
                clamp_, calcExpGrad_
            );
            gridIndexQue_.FreeTensor(indexTensor);
            gridWeightQue_.FreeTensor(weightTensor);
        } else {
            asc_vf_call<CalcGradGlobal<THREAD_NUM>>(
                dim3{THREAD_NUM, 1, 1},
                (__ubuf__ float *)colorsTensor.GetPhyAddr(),
                (__ubuf__ float *)vColorsInTensor.GetPhyAddr(),
                (__ubuf__ float *)vColorsOutTensor.GetPhyAddr(),
                (__ubuf__ float *)vExpTensor_.GetPhyAddr(),
                (__ubuf__ float *)expTensor_.GetPhyAddr(),
                isTailBlock_ ? validTaskNum_ : taskNumPerCurLoop_,
                taskNumPerCurLoop_, clamp_, calcExpGrad_
            );
        }
        colorsQue_.FreeTensor(colorsTensor);
        vColorsInQue_.FreeTensor(vColorsInTensor);
        vColorsOutQue_.EnQue(vColorsOutTensor);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx, uint32_t cameraIdx, int32_t taskLoopIndex)
    {
        LocalTensor<float> vColorsOutTensor = vColorsOutQue_.DeQue<float>();
        uint64_t taskCopyinIndex = taskStartIndex_ + taskLoopIndex * taskNumPerLoop_;
        uint64_t baseOffset = batchIdx * cameraNum_ * gaussNum_ + cameraIdx * gaussNum_;
        const uint32_t colorDim = 3;
        uint64_t offset = 0;
        if (isTailBlock_) {
            DataCopyExtParams copyParams {
                1,
                static_cast<uint32_t>(validTaskNum_ * FLOAT_SIZE),
                0, 0, 0
            };
            for (uint32_t i = 0; i < colorDim; i++) {
                offset = baseOffset * colorDim + i * gaussNum_ + taskCopyinIndex;
                DataCopyPad(vColorsOutGM_[offset], vColorsOutTensor[i * taskNumPerCurLoop_], copyParams);
            }
        } else {
            for (uint32_t i = 0; i < colorDim; i++) {
                offset = baseOffset * colorDim + i * gaussNum_ + taskCopyinIndex;
                DataCopy(vColorsOutGM_[offset], vColorsOutTensor[i * taskNumPerCurLoop_], taskNumPerCurLoop_);
            }
        }
        vColorsOutQue_.FreeTensor(vColorsOutTensor);

        // need reduce from UB to GM
        if (calcExpGrad_) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            SetAtomicAdd<float>();
            uint32_t expSize = gridWidth_ * gridHeight_ * gridDepth_;
            DataCopyExtParams copyParams{1, expSize * FLOAT_SIZE * 12, 0, 0, 0};
            DataCopyPad(vExposureGM_, vExpTensor_, copyParams);
            SetAtomicNone();
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }

private:
    TPipe *pipe_;
    TBuf<TPosition::VECCALC> expBuf_, vExpBuf_, cntBuf_;
    TQue<QuePosition::VECIN, 1> colorsQue_, vColorsInQue_, gridIndexQue_, gridWeightQue_;
    TQue<QuePosition::VECOUT, 1> vColorsOutQue_;
    LocalTensor<float> expTensor_, vExpTensor_;
    LocalTensor<int32_t> cntTensor_;

    GlobalTensor<float> colorsGM_;
    GlobalTensor<float> vColorsInGM_;
    GlobalTensor<float> exposureGM_;
    GlobalTensor<int32_t> cntGM_;
    GlobalTensor<int32_t> gridIndexGM_;
    GlobalTensor<float> gridWeightGM_;
    GlobalTensor<float> vColorsOutGM_;
    GlobalTensor<float> vExposureGM_;

    uint32_t batchSize_, gaussNum_, cameraNum_;
    uint32_t gridWidth_, gridHeight_, gridDepth_;
    bool useGrid_, calcExpGrad_, clamp_, isTailBlock_;
    uint32_t totalTaskNum_, tailNum_, taskNumPerScore_, taskNumPerLcore_, cnt_, validTaskNum_;
    uint32_t numScore_, numLcore_, blockDim_, ubTotalSize_;
    uint64_t blockIndex_;
    uint32_t taskNumPerLoop_, taskNumPerCurLoop_, taskNumPerCore_, taskStartIndex_, taskLoop_;
};

extern "C" __global__ __aicore__ void color_exposure_bwd(GM_ADDR colors, GM_ADDR vColorsIn, GM_ADDR exposure,
    GM_ADDR cnt, GM_ADDR gridIndex, GM_ADDR gridWeight, GM_ADDR vColorsOut, GM_ADDR vExposure, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        ColorExposureBwd op;
        op.Init(colors, vColorsIn, exposure, cnt, gridIndex, gridWeight, vColorsOut, vExposure,
            &tiling_data, &pipe);
        op.Process();
    }
}
