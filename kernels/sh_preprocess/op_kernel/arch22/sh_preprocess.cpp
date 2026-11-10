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

using namespace AscendC;

constexpr uint32_t AXIS_NUM = 3;
constexpr uint32_t FLOAT_BYTES = 4;

__aicore__ inline uint32_t Align8(uint32_t value)
{
    return (value + 7) / 8 * 8;
}

class ShPreprocessKernel {
public:
    __aicore__ inline void Init(GM_ADDR camtoworlds, GM_ADDR means, GM_ADDR raysD,
                                const ShPreprocessTilingData *td, TPipe *pipe)
    {
        gaussianNum_ = td->gaussianNum;
        taskNumPerLoop_ = td->taskNumPerLoop;
        taskNumPerCore_ = td->taskNumPerCore;
        taskStartIdx_ = GetBlockIdx() * taskNumPerCore_;
        if (taskStartIdx_ >= gaussianNum_) return;
        if (taskStartIdx_ + taskNumPerCore_ > gaussianNum_) {
            taskNumPerCore_ = gaussianNum_ - taskStartIdx_;
        }

        uint32_t aligned = Align8(taskNumPerLoop_);
        pipe->InitBuffer(meansBuf_, aligned * AXIS_NUM * FLOAT_BYTES);
        pipe->InitBuffer(dirsBuf_, aligned * AXIS_NUM * FLOAT_BYTES);
        pipe->InitBuffer(normBuf_, aligned * FLOAT_BYTES);
        meansTensor_ = meansBuf_.Get<float>();
        dirsTensor_ = dirsBuf_.Get<float>();
        normTensor_ = normBuf_.Get<float>();
        meansGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(means), AXIS_NUM * gaussianNum_);
        camtoworldsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(camtoworlds), 16);
        raysDGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(raysD), AXIS_NUM * gaussianNum_);
    }

    __aicore__ inline void Process()
    {
        if (taskStartIdx_ >= gaussianNum_) return;
        float rayOx = camtoworldsGM_.GetValue(3);
        float rayOy = camtoworldsGM_.GetValue(7);
        float rayOz = camtoworldsGM_.GetValue(11);
        uint32_t taskLoop = (taskNumPerCore_ + taskNumPerLoop_ - 1) / taskNumPerLoop_;

        for (uint32_t loop = 0; loop < taskLoop; ++loop) {
            uint32_t colOff = loop * taskNumPerLoop_;
            uint32_t realNum = taskNumPerCore_ - colOff;
            if (realNum > taskNumPerLoop_) realNum = taskNumPerLoop_;
            uint32_t aligned = Align8(realNum);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            for (uint32_t c = 0; c < AXIS_NUM; ++c) {
                DataCopyExtParams params{1, realNum * FLOAT_BYTES, 0, 0, 0};
                DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
                DataCopyPad(meansTensor_[c * aligned], meansGM_[taskStartIdx_ + c * gaussianNum_ + colOff],
                            params, pad);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            Adds(dirsTensor_, meansTensor_, -rayOx, aligned);
            Adds(dirsTensor_[aligned], meansTensor_[aligned], -rayOy, aligned);
            Adds(dirsTensor_[2 * aligned], meansTensor_[2 * aligned], -rayOz, aligned);
            Mul(meansTensor_, dirsTensor_, dirsTensor_, aligned);
            Mul(meansTensor_[aligned], dirsTensor_[aligned], dirsTensor_[aligned], aligned);
            Mul(meansTensor_[2 * aligned], dirsTensor_[2 * aligned], dirsTensor_[2 * aligned], aligned);
            Add(normTensor_, meansTensor_, meansTensor_[aligned], aligned);
            Add(normTensor_, normTensor_, meansTensor_[2 * aligned], aligned);
            Sqrt(normTensor_, normTensor_, aligned);
            Div(dirsTensor_, dirsTensor_, normTensor_, aligned);
            Div(dirsTensor_[aligned], dirsTensor_[aligned], normTensor_, aligned);
            Div(dirsTensor_[2 * aligned], dirsTensor_[2 * aligned], normTensor_, aligned);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            for (uint32_t c = 0; c < AXIS_NUM; ++c) {
                DataCopyExtParams params{1, realNum * FLOAT_BYTES, 0, 0, 0};
                DataCopyPad(raysDGM_[taskStartIdx_ + c * gaussianNum_ + colOff],
                            dirsTensor_[c * aligned], params);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }
    }

private:
    uint32_t gaussianNum_, taskNumPerLoop_, taskNumPerCore_, taskStartIdx_;
    TBuf<TPosition::VECCALC> meansBuf_, dirsBuf_, normBuf_;
    LocalTensor<float> meansTensor_, dirsTensor_, normTensor_;
    GlobalTensor<float> meansGM_, camtoworldsGM_, raysDGM_;
};

extern "C" __global__ __aicore__ void sh_preprocess(GM_ADDR camtoworlds, GM_ADDR means,
                                                      GM_ADDR raysD, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        ShPreprocessKernel op;
        op.Init(camtoworlds, means, raysD, &tilingData, &pipe);
        op.Process();
    }
}
