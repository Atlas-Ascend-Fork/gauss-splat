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

using namespace AscendC;
using namespace std;

constexpr uint32_t AXIS_NUM = 3;
constexpr uint32_t ALIGN_VAL = 8;
constexpr int32_t FLOAT_SIZE_B = 4;

__aicore__ inline uint32_t Align8(uint32_t v)
{
    return ((v + ALIGN_VAL - 1) / ALIGN_VAL) * ALIGN_VAL;
}

class ShPreprocessBwdKernel {
public:
    __aicore__ inline ShPreprocessBwdKernel() {}

    __aicore__ inline void GetTilingData(const ShPreprocessBwdTilingData *td)
    {
        gaussianNum_    = td->gaussianNum;
        taskNumPerLoop_ = td->taskNumPerLoop;
        taskNumPerCore_ = td->taskNumPerCore;
    }

    __aicore__ inline void Init(GM_ADDR means, GM_ADDR camtoworlds, GM_ADDR raysD,
                                 GM_ADDR vRaysD, GM_ADDR vMeans,
                                 const ShPreprocessBwdTilingData *td, TPipe *pipe)
    {
        GetTilingData(td);
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();

        taskStartIdx_ = blockIdx_ * taskNumPerCore_;
        if (taskStartIdx_ >= gaussianNum_) {
            taskNumPerCore_ = 0;
            return;
        }
        if (taskStartIdx_ + taskNumPerCore_ > gaussianNum_) {
            taskNumPerCore_ = gaussianNum_ - taskStartIdx_;
        }
        if (taskNumPerCore_ <= 0) return;
        taskLoop_ = (taskNumPerCore_ + taskNumPerLoop_ - 1) / taskNumPerLoop_;

        uint32_t loopAli = Align8(taskNumPerLoop_);
        uint32_t axisBuf = loopAli * AXIS_NUM * FLOAT_SIZE_B;

        pipe_->InitBuffer(dirsBuf_, axisBuf);
        pipe_->InitBuffer(vRaysDBuf_, axisBuf);
        pipe_->InitBuffer(raysDBuf_, axisBuf);
        pipe_->InitBuffer(sqBuf_, axisBuf);                 // 3 channels for dot/norm accumulation
        pipe_->InitBuffer(normBuf_, loopAli * FLOAT_SIZE_B);
        pipe_->InitBuffer(dotBuf_, loopAli * FLOAT_SIZE_B);

        dirsTensor_    = dirsBuf_.Get<float>();
        vRaysDTensor_  = vRaysDBuf_.Get<float>();
        raysDTensor_   = raysDBuf_.Get<float>();
        sqTensor_      = sqBuf_.Get<float>();
        normTensor_    = normBuf_.Get<float>();
        dotTensor_     = dotBuf_.Get<float>();

        meansGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(means),
                                 AXIS_NUM * gaussianNum_);
        camtoworldsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(camtoworlds), 16);
        raysDGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(raysD),
                                 AXIS_NUM * gaussianNum_);
        vRaysDGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vRaysD),
                                  AXIS_NUM * gaussianNum_);
        vMeansGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(vMeans),
                                  AXIS_NUM * gaussianNum_);
    }

    __aicore__ inline void Process()
    {
        if (taskStartIdx_ >= gaussianNum_) return;
        rayOx_ = camtoworldsGM_.GetValue(0 * 4 + 3);
        rayOy_ = camtoworldsGM_.GetValue(1 * 4 + 3);
        rayOz_ = camtoworldsGM_.GetValue(2 * 4 + 3);

        for (uint32_t loop = 0; loop < taskLoop_; loop++) {
            uint32_t realNum = taskNumPerLoop_;
            if (loop == taskLoop_ - 1) {
                realNum = taskNumPerCore_ - loop * taskNumPerLoop_;
            }
            uint32_t aliNum = Align8(realNum);
            uint32_t gOff = loop * taskNumPerLoop_;

            // load means, rays_d, v_rays_d (aligned, pad-safe)
            {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                CopyInChannels3(dirsTensor_, meansGM_, gOff, realNum, aliNum);
                CopyInChannels3(raysDTensor_, raysDGM_, gOff, realNum, aliNum);
                CopyInChannels3(vRaysDTensor_, vRaysDGM_, gOff, realNum, aliNum);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            }

            // d = means - rays_o
            Adds(dirsTensor_, dirsTensor_, -rayOx_, aliNum);
            Adds(dirsTensor_[aliNum], dirsTensor_[aliNum], -rayOy_, aliNum);
            Adds(dirsTensor_[aliNum * 2], dirsTensor_[aliNum * 2], -rayOz_, aliNum);

            // dot = sum(rays_d * v_rays_d) = rx*vrx + ry*vry + rz*vrz
            Mul(sqTensor_, raysDTensor_, vRaysDTensor_, aliNum);
            Mul(sqTensor_[aliNum], raysDTensor_[aliNum], vRaysDTensor_[aliNum], aliNum);
            Mul(sqTensor_[aliNum * 2], raysDTensor_[aliNum * 2], vRaysDTensor_[aliNum * 2], aliNum);
            Add(dotTensor_, sqTensor_, sqTensor_[aliNum], aliNum);
            Add(dotTensor_, dotTensor_, sqTensor_[aliNum * 2], aliNum);

            // ||d|| = sqrt(dx*dx + dy*dy + dz*dz)
            Mul(sqTensor_, dirsTensor_, dirsTensor_, aliNum);
            Mul(sqTensor_[aliNum], dirsTensor_[aliNum], dirsTensor_[aliNum], aliNum);
            Mul(sqTensor_[aliNum * 2], dirsTensor_[aliNum * 2], dirsTensor_[aliNum * 2], aliNum);
            Add(normTensor_, sqTensor_, sqTensor_[aliNum], aliNum);
            Add(normTensor_, normTensor_, sqTensor_[aliNum * 2], aliNum);
            Sqrt(normTensor_, normTensor_, aliNum);

            // v_means = (v_rays_d - dot * rays_d) / ||d||
            Mul(sqTensor_, dotTensor_, raysDTensor_, aliNum);
            Sub(vRaysDTensor_, vRaysDTensor_, sqTensor_, aliNum);
            Mul(sqTensor_, dotTensor_, raysDTensor_[aliNum], aliNum);
            Sub(vRaysDTensor_[aliNum], vRaysDTensor_[aliNum], sqTensor_, aliNum);
            Mul(sqTensor_, dotTensor_, raysDTensor_[aliNum * 2], aliNum);
            Sub(vRaysDTensor_[aliNum * 2], vRaysDTensor_[aliNum * 2], sqTensor_, aliNum);

            Div(vRaysDTensor_, vRaysDTensor_, normTensor_, aliNum);
            Div(vRaysDTensor_[aliNum], vRaysDTensor_[aliNum], normTensor_, aliNum);
            Div(vRaysDTensor_[aliNum * 2], vRaysDTensor_[aliNum * 2], normTensor_, aliNum);

            // store v_means (real length, pad-safe)
            {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                CopyOutChannels3(vMeansGM_, gOff, vRaysDTensor_, realNum);
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            }
        }
    }

    __aicore__ inline void CopyInChannels3(LocalTensor<float> dst, GlobalTensor<float> src,
                                            uint32_t colOff, uint32_t realNum, uint32_t aliNum)
    {
        for (uint32_t c = 0; c < AXIS_NUM; c++) {
            DataCopyExtParams params{1, realNum * FLOAT_SIZE_B, 0, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
            DataCopyPad(dst[c * aliNum], src[taskStartIdx_ + c * gaussianNum_ + colOff], params, pad);
        }
    }

    __aicore__ inline void CopyOutChannels3(GlobalTensor<float> dst, uint32_t colOff,
                                             LocalTensor<float> src, uint32_t realNum)
    {
        uint32_t aliNum = Align8(realNum);
        for (uint32_t c = 0; c < AXIS_NUM; c++) {
            DataCopyExtParams params{1, realNum * FLOAT_SIZE_B, 0, 0, 0};
            DataCopyPad(dst[taskStartIdx_ + c * gaussianNum_ + colOff], src[c * aliNum], params);
        }
    }

private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t gaussianNum_, taskNumPerLoop_, taskNumPerCore_, taskLoop_;
    uint32_t taskStartIdx_;
    float rayOx_, rayOy_, rayOz_;

    TBuf<TPosition::VECCALC> dirsBuf_, vRaysDBuf_, raysDBuf_, sqBuf_, normBuf_, dotBuf_;
    LocalTensor<float> dirsTensor_, vRaysDTensor_, raysDTensor_, sqTensor_, normTensor_, dotTensor_;

    GlobalTensor<float> meansGM_, camtoworldsGM_, raysDGM_, vRaysDGM_, vMeansGM_;
};

extern "C" __global__ __aicore__ void sh_preprocess_bwd(GM_ADDR means, GM_ADDR camtoworlds,
                                                         GM_ADDR raysD, GM_ADDR vRaysD,
                                                         GM_ADDR vMeans, GM_ADDR workspace,
                                                         GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        ShPreprocessBwdKernel op;
        op.Init(means, camtoworlds, raysD, vRaysD, vMeans, &tilingData, &pipe);
        op.Process();
    }
}
