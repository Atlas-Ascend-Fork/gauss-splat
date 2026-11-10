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
constexpr uint32_t REG_LEN = 64;   // 1 vector register = 64 floats (256 bytes)
constexpr int32_t FLOAT_SIZE_B = 4;

__aicore__ inline uint32_t Align8(uint32_t v)
{
    return ((v + ALIGN_VAL - 1) / ALIGN_VAL) * ALIGN_VAL;
}

__aicore__ inline uint32_t Align64(uint32_t v)
{
    return ((v + REG_LEN - 1) / REG_LEN) * REG_LEN;
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

        // channel stride is 64-float aligned so VF slices cover the whole buffer
        uint32_t chMax = Align64(Align8(taskNumPerLoop_));
        uint32_t axisBuf = chMax * AXIS_NUM * FLOAT_SIZE_B;

        pipe_->InitBuffer(dirsBuf_, axisBuf);
        pipe_->InitBuffer(vRaysDBuf_, axisBuf);
        pipe_->InitBuffer(raysDBuf_, axisBuf);

        dirsTensor_    = dirsBuf_.Get<float>();
        vRaysDTensor_  = vRaysDBuf_.Get<float>();
        raysDTensor_   = raysDBuf_.Get<float>();

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
            uint32_t ch = Align64(aliNum);
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

            // v_means = (v_rays_d - (rays_d . v_rays_d) * rays_d) / ||d||
            // computed per 64-float slice in registers; dot / norm never touch UB.
            ComputeVMeans(ch);

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
        uint32_t ch = Align64(aliNum);
        for (uint32_t c = 0; c < AXIS_NUM; c++) {
            DataCopyExtParams params{1, realNum * FLOAT_SIZE_B, 0, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, 0, 0.0f};
            DataCopyPad(dst[c * ch], src[taskStartIdx_ + c * gaussianNum_ + colOff], params, pad);
        }
    }

    __aicore__ inline void CopyOutChannels3(GlobalTensor<float> dst, uint32_t colOff,
                                             LocalTensor<float> src, uint32_t realNum)
    {
        uint32_t ch = Align64(Align8(realNum));
        for (uint32_t c = 0; c < AXIS_NUM; c++) {
            DataCopyExtParams params{1, realNum * FLOAT_SIZE_B, 0, 0, 0};
            DataCopyPad(dst[taskStartIdx_ + c * gaussianNum_ + colOff], src[c * ch], params);
        }
    }

    // v_means = (v - (r . v) * r) / ||d||, per 64-float slice:
    //   d = means - rays_o; invn = rsqrt(d.d); dot = r.v; vmeans = (v - dot*r) * invn.
    // All intermediates stay in vector registers; only the 3 output channels are
    // written back to UB (in-place over vRaysDTensor_, which CopyOut then drains).
    // Pad lanes (beyond realNum) see dirs=0 so norm_sq = rox^2+roy^2+roz^2: finite
    // invn unless the camera sits at the origin, and either way pad lanes never
    // leave UB (CopyOut writes exactly realNum).
    __aicore__ inline void ComputeVMeans(uint32_t ch)
    {
        __ubuf__ float *dirsP = reinterpret_cast<__ubuf__ float *>(dirsTensor_.GetPhyAddr());
        __ubuf__ float *raysP = reinterpret_cast<__ubuf__ float *>(raysDTensor_.GetPhyAddr());
        __ubuf__ float *vP = reinterpret_cast<__ubuf__ float *>(vRaysDTensor_.GetPhyAddr());
        float rox = rayOx_;
        float roy = rayOy_;
        float roz = rayOz_;

        for (uint32_t off = 0; off < ch; off += REG_LEN) {
            __ubuf__ float *mxP = dirsP + off;
            __ubuf__ float *myP = dirsP + ch + off;
            __ubuf__ float *mzP = dirsP + 2 * ch + off;
            __ubuf__ float *rxP = raysP + off;
            __ubuf__ float *ryP = raysP + ch + off;
            __ubuf__ float *rzP = raysP + 2 * ch + off;
            __ubuf__ float *vxP = vP + off;
            __ubuf__ float *vyP = vP + ch + off;
            __ubuf__ float *vzP = vP + 2 * ch + off;

            __VEC_SCOPE__
            {
                MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
                MicroAPI::RegTensor<float> mx, my, mz, rx, ry, rz, vx, vy, vz;
                MicroAPI::RegTensor<float> s, dot, invn;

                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(mx, mxP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(my, myP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(mz, mzP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(rx, rxP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(ry, ryP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(rz, rzP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vx, vxP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vy, vyP, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vz, vzP, 1, mask);

                // d = means - rays_o; norm_sq = d.d
                MicroAPI::Adds<float>(mx, mx, -rox, mask);
                MicroAPI::Adds<float>(my, my, -roy, mask);
                MicroAPI::Adds<float>(mz, mz, -roz, mask);
                MicroAPI::Mul<float>(invn, mx, mx, mask);
                MicroAPI::Mul<float>(s, my, my, mask);
                MicroAPI::Add<float>(invn, invn, s, mask);
                MicroAPI::Mul<float>(s, mz, mz, mask);
                MicroAPI::Add<float>(invn, invn, s, mask);
                MicroAPI::Sqrt<float>(invn, invn, mask);   // ||d||

                // dot = rays_d . v_rays_d
                MicroAPI::Mul<float>(dot, rx, vx, mask);
                MicroAPI::Mul<float>(s, ry, vy, mask);
                MicroAPI::Add<float>(dot, dot, s, mask);
                MicroAPI::Mul<float>(s, rz, vz, mask);
                MicroAPI::Add<float>(dot, dot, s, mask);

                // vmeans = (v - dot * rays_d) / ||d||
                MicroAPI::Mul<float>(s, dot, rx, mask);
                MicroAPI::Sub<float>(vx, vx, s, mask);
                MicroAPI::Div<float>(vx, vx, invn, mask);
                MicroAPI::Mul<float>(s, dot, ry, mask);
                MicroAPI::Sub<float>(vy, vy, s, mask);
                MicroAPI::Div<float>(vy, vy, invn, mask);
                MicroAPI::Mul<float>(s, dot, rz, mask);
                MicroAPI::Sub<float>(vz, vz, s, mask);
                MicroAPI::Div<float>(vz, vz, invn, mask);

                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vxP, vx, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vyP, vy, 1, mask);
                MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vzP, vz, 1, mask);
            }
        }
    }

private:
    TPipe *pipe_;
    uint32_t blockIdx_;
    uint32_t gaussianNum_, taskNumPerLoop_, taskNumPerCore_, taskLoop_;
    uint32_t taskStartIdx_;
    float rayOx_, rayOy_, rayOz_;

    TBuf<TPosition::VECCALC> dirsBuf_, vRaysDBuf_, raysDBuf_;
    LocalTensor<float> dirsTensor_, vRaysDTensor_, raysDTensor_;

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
