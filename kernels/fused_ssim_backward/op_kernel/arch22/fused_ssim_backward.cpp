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

namespace {
constexpr int32_t FLOAT_SIZE = 4;
constexpr int32_t MAX_WS = 16;
constexpr int32_t HALO_LEFT = 8;
constexpr int32_t HALO_ALIGN = 8;
constexpr float SSIM_C1 = 0.0001f;
constexpr float SSIM_C2 = 0.0009f;
constexpr float TWO_F = 2.0f;
constexpr float NEG_ONE_F = -1.0f;
constexpr float ZERO_F = 0.0f;

__aicore__ inline int64_t AlignUp8(int64_t x)
{
    return (x + 7) / 8 * 8;
}
} // namespace

// K_a   = (2g/M) * { mu2*(B-A) - ssim*mu1*(Q-P) }
// K_b   = (2g/M) * A
// K_g   = -(2g/M) * ssim * P
// K_a2  = (2g/M) * { mu1*(B-A) - ssim*mu2*(Q-P) }   (2-sided only)
// grad_img1 = conv(K_a, W)  + img2*conv(K_b, W) + img1*conv(K_g, W)
// grad_img2 = conv(K_a2, W) + img1*conv(K_b, W) + img2*conv(K_g, W)
// where M = P*Q, A = 2*m12+C1, B = 2*sigma12+C2, P = m11+m22+C1, Q = sigma1+sigma2+C2.
class FusedSsimBackward {
public:
    __aicore__ inline FusedSsimBackward() {}

    __aicore__ inline void Init(GM_ADDR gradSsim, GM_ADDR img1, GM_ADDR img2, GM_ADDR weight,
                                GM_ADDR mu1In, GM_ADDR mu2In,
                                GM_ADDR s11In, GM_ADDR s22In, GM_ADDR s12In,
                                GM_ADDR gradImg1Out, GM_ADDR gradImg2Out,
                                GM_ADDR workspace,
                                const FusedSsimBackwardTilingData *tiling, TPipe *pipe)
    {
        blockIdx_ = GetBlockIdx();
        bcNum_ = tiling->bcNum;
        height_ = tiling->height;
        width_ = tiling->width;
        windowSize_ = tiling->windowSize;
        radius_ = tiling->radius;
        coresPerImage_ = tiling->coresPerImage;
        rowsPerCore_ = tiling->rowsPerCore;
        needCoreNum_ = tiling->needCoreNum;
        sysWsBytes_ = tiling->sysWsBytes;
        perCoreWsFloats_ = tiling->perCoreWsFloats;
        needGradImg2_ = tiling->needGradImg2;
        chunkW_ = tiling->chunkW;
        numWChunks_ = tiling->numWChunks;
        numKMaps_ = (needGradImg2_ == 1) ? 4 : 3;

        blockRows_ = 2 * radius_ + 1;
        maxEffW_ = chunkW_ + 2 * HALO_ALIGN;
        wAlign_ = AlignUp8(maxEffW_);
        padLen_ = AlignUp8(HALO_LEFT + maxEffW_ + radius_);

        gradSsimGM_.SetGlobalBuffer((__gm__ float *)gradSsim);
        img1GM_.SetGlobalBuffer((__gm__ float *)img1);
        img2GM_.SetGlobalBuffer((__gm__ float *)img2);
        weightGM_.SetGlobalBuffer((__gm__ float *)weight);
        mu1GM_.SetGlobalBuffer((__gm__ float *)mu1In);
        mu2GM_.SetGlobalBuffer((__gm__ float *)mu2In);
        s11GM_.SetGlobalBuffer((__gm__ float *)s11In);
        s22GM_.SetGlobalBuffer((__gm__ float *)s22In);
        s12GM_.SetGlobalBuffer((__gm__ float *)s12In);
        gradImg1GM_.SetGlobalBuffer((__gm__ float *)gradImg1Out);
        gradImg2GM_.SetGlobalBuffer((__gm__ float *)gradImg2Out);

        __gm__ uint8_t* wsBase = workspace + sysWsBytes_
            + blockIdx_ * perCoreWsFloats_ * static_cast<int64_t>(sizeof(float));
        wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsBase));

        pipe_ = pipe;
        // hL_ aliases scratchL_; imgL_ aliases imputL_ (disjoint lifetimes).
        pipe_->InitBuffer(weightBuf_, MAX_WS * FLOAT_SIZE);
        pipe_->InitBuffer(inputBuf_, 6 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(scratchBuf_, 5 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(tmpBuf_, wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(kBuf_, numKMaps_ * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(vBuf_, numKMaps_ * padLen_ * FLOAT_SIZE);

        weightL_ = weightBuf_.Get<float>();
        inputL_ = inputBuf_.Get<float>();
        scratchL_ = scratchBuf_.Get<float>();
        tmpL_ = tmpBuf_.Get<float>();
        kL_ = kBuf_.Get<float>();
        vL_ = vBuf_.Get<float>();
        hL_ = scratchL_;
        imgL_ = inputL_;

        rowPadParams_ = DataCopyPadExtParams<float>{false, 0, 0, 0.0f};

        DataCopy(weightL_, weightGM_, MAX_WS);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int32_t k = 0; k < windowSize_; k++) {
            wv_[k] = weightL_.GetValue(k);
        }
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= needCoreNum_) {
            return;
        }
        int64_t bc = blockIdx_ / coresPerImage_;
        int64_t band = blockIdx_ % coresPerImage_;
        int64_t h0 = band * rowsPerCore_;
        int64_t h1 = h0 + rowsPerCore_;
        if (h1 > height_) {
            h1 = height_;
        }
        int64_t imgBase = bc * height_ * width_;

        for (int64_t j = 0; j < numWChunks_; j++) {
            c0_ = j * chunkW_;
            c1_ = c0_ + chunkW_;
            if (c1_ > width_) {
                c1_ = width_;
            }
            outW_ = c1_ - c0_;
            lc0_ = c0_ - HALO_ALIGN;
            if (lc0_ < 0) {
                lc0_ = 0;
            }
            int64_t lc1 = c1_ + HALO_ALIGN;
            if (lc1 > width_) {
                lc1 = width_;
            }
            loadW_ = lc1 - lc0_;
            dataOff_ = c0_ - lc0_;
            rowLoadParams_ = DataCopyExtParams{1, static_cast<uint32_t>(loadW_ * FLOAT_SIZE), 0, 0, 0};
            rowStoreParams_ = DataCopyExtParams{1, static_cast<uint32_t>(outW_ * FLOAT_SIZE), 0, 0, 0};

            for (int64_t h = h0; h < h1; h++) {
                VerticalConvK(imgBase, h);
                HorizontalConvK();
                ComputeGradImg(imgBase, h);
            }
        }
    }

    __aicore__ inline void VerticalConvK(int64_t imgBase, int64_t h)
    {
        for (int32_t slot = 0; slot < numKMaps_; slot++) {
            Duplicate(vL_[slot * padLen_], ZERO_F, padLen_);
        }
        for (int64_t k = 0; k < blockRows_; k++) {
            int64_t gmRow = h - radius_ + k;
            if (gmRow < 0 || gmRow >= height_) {
                continue;
            }
            LoadInputRow(imgBase, gmRow);
            ComputeKVec(kL_);
            float w = wv_[k];
            Axpy(vL_[0 * padLen_ + HALO_LEFT], kL_[0 * wAlign_], w, loadW_);
            Axpy(vL_[1 * padLen_ + HALO_LEFT], kL_[1 * wAlign_], w, loadW_);
            Axpy(vL_[2 * padLen_ + HALO_LEFT], kL_[2 * wAlign_], w, loadW_);
            if (needGradImg2_ == 1) {
                Axpy(vL_[3 * padLen_ + HALO_LEFT], kL_[3 * wAlign_], w, loadW_);
            }
        }
    }

    __aicore__ inline void HorizontalConvK()
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopy(wsGM_, vL_, static_cast<uint32_t>(perCoreWsFloats_));
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        LocalTensor<float> tmp = tmpL_[0 * wAlign_];
        for (int32_t slot = 0; slot < numKMaps_; slot++) {
            Duplicate(hL_[slot * wAlign_], ZERO_F, wAlign_);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            int64_t baseAligned = slot * padLen_ + HALO_LEFT + dataOff_ - radius_;
            for (int32_t dx = 0; dx < windowSize_; dx++) {
                int64_t srcElemOffset = baseAligned + dx;
                DataCopy(tmp, wsGM_[srcElemOffset], static_cast<uint32_t>(wAlign_));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                Muls(tmp, tmp, wv_[dx], outW_);
                Add(hL_[slot * wAlign_], hL_[slot * wAlign_], tmp, outW_);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }
        }
    }

    __aicore__ inline void ComputeGradImg(int64_t imgBase, int64_t h)
    {
        int64_t row_off = imgBase + h * width_ + c0_;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        DataCopyPad(imgL_[0 * wAlign_], img1GM_[row_off], rowStoreParams_, rowPadParams_);
        DataCopyPad(imgL_[1 * wAlign_], img2GM_[row_off], rowStoreParams_, rowPadParams_);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // hL_ aliases scratchL_; use inputL_[2..4] as accumulators
        LocalTensor<float> gi1 = inputL_[2 * wAlign_];
        LocalTensor<float> acc = inputL_[3 * wAlign_];
        Mul(gi1, imgL_[1 * wAlign_], hL_[1 * wAlign_], outW_);
        Add(gi1, gi1, hL_[0 * wAlign_], outW_);
        Mul(acc, imgL_[0 * wAlign_], hL_[2 * wAlign_], outW_);
        Add(gi1, gi1, acc, outW_);

        if (needGradImg2_ == 1) {
            LocalTensor<float> gi2 = inputL_[4 * wAlign_];
            Mul(gi2, imgL_[0 * wAlign_], hL_[1 * wAlign_], outW_);
            Add(gi2, gi2, hL_[3 * wAlign_], outW_);
            Mul(acc, imgL_[1 * wAlign_], hL_[2 * wAlign_], outW_);
            Add(gi2, gi2, acc, outW_);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            DataCopyPad(gradImg1GM_[row_off], gi1, rowStoreParams_);
            DataCopyPad(gradImg2GM_[row_off], gi2, rowStoreParams_);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            return;
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(gradImg1GM_[row_off], gi1, rowStoreParams_);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void LoadInputRow(int64_t imgBase, int64_t gmRow)
    {
        int64_t row_off = imgBase + gmRow * width_ + lc0_;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        for (int32_t m = 0; m < 6; m++) {
            Duplicate(inputL_[m * wAlign_], ZERO_F, wAlign_);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        DataCopyPad(inputL_[0 * wAlign_], mu1GM_[row_off], rowLoadParams_, rowPadParams_);
        DataCopyPad(inputL_[1 * wAlign_], mu2GM_[row_off], rowLoadParams_, rowPadParams_);
        DataCopyPad(inputL_[2 * wAlign_], s11GM_[row_off], rowLoadParams_, rowPadParams_);
        DataCopyPad(inputL_[3 * wAlign_], s22GM_[row_off], rowLoadParams_, rowPadParams_);
        DataCopyPad(inputL_[4 * wAlign_], s12GM_[row_off], rowLoadParams_, rowPadParams_);
        DataCopyPad(inputL_[5 * wAlign_], gradSsimGM_[row_off], rowLoadParams_, rowPadParams_);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void ComputeKVec(LocalTensor<float> kOut)
    {
        LocalTensor<float> mu1 = inputL_[0 * wAlign_];
        LocalTensor<float> mu2 = inputL_[1 * wAlign_];
        LocalTensor<float> S11 = inputL_[2 * wAlign_];
        LocalTensor<float> S22 = inputL_[3 * wAlign_];
        LocalTensor<float> S12 = inputL_[4 * wAlign_];
        LocalTensor<float> grad = inputL_[5 * wAlign_];

        LocalTensor<float> s0 = scratchL_[0 * wAlign_];
        LocalTensor<float> s1 = scratchL_[1 * wAlign_];
        LocalTensor<float> s2 = scratchL_[2 * wAlign_];
        LocalTensor<float> s3 = scratchL_[3 * wAlign_];
        LocalTensor<float> s4 = scratchL_[4 * wAlign_];

        LocalTensor<float> Ka = kOut[0 * wAlign_];
        LocalTensor<float> Kb = kOut[1 * wAlign_];
        LocalTensor<float> Kg = kOut[2 * wAlign_];

        Mul(s0, mu1, mu1, loadW_);
        Mul(s1, mu2, mu2, loadW_);
        Mul(s2, mu1, mu2, loadW_);

        Add(s3, s0, s1, loadW_);
        Adds(s3, s3, SSIM_C1, loadW_);

        Muls(Ka, s2, TWO_F, loadW_);
        Adds(Ka, Ka, SSIM_C1, loadW_);

        Sub(s0, S11, s0, loadW_);
        Maxs(s0, s0, ZERO_F, loadW_);
        Sub(s1, S22, s1, loadW_);
        Maxs(s1, s1, ZERO_F, loadW_);

        Add(s0, s0, s1, loadW_);
        Adds(s0, s0, SSIM_C2, loadW_);

        Sub(s1, S12, s2, loadW_);

        Muls(Kb, s1, TWO_F, loadW_);
        Adds(Kb, Kb, SSIM_C2, loadW_);

        Mul(s2, s3, s0, loadW_);

        Mul(s1, Ka, Kb, loadW_);
        Div(s1, s1, s2, loadW_);

        Muls(s4, grad, TWO_F, loadW_);
        Div(s4, s4, s2, loadW_);

        Sub(s0, s0, s3, loadW_);
        Sub(s2, Kb, Ka, loadW_);

        Mul(Kb, s4, Ka, loadW_);

        Mul(Kg, s4, s1, loadW_);
        Mul(Kg, Kg, s3, loadW_);
        Muls(Kg, Kg, NEG_ONE_F, loadW_);

        if (needGradImg2_ == 1) {
            LocalTensor<float> Ka2 = kOut[3 * wAlign_];
            Mul(Ka2, mu1, s2, loadW_);
            Mul(s3, mu2, s0, loadW_);
            Mul(s3, s3, s1, loadW_);
            Sub(Ka2, Ka2, s3, loadW_);
            Mul(Ka2, Ka2, s4, loadW_);
        }

        Mul(Ka, mu2, s2, loadW_);
        Mul(s2, mu1, s0, loadW_);
        Mul(s2, s2, s1, loadW_);
        Sub(Ka, Ka, s2, loadW_);
        Mul(Ka, Ka, s4, loadW_);
    }

private:
    TPipe *pipe_;
    GlobalTensor<float> gradSsimGM_;
    GlobalTensor<float> img1GM_;
    GlobalTensor<float> img2GM_;
    GlobalTensor<float> weightGM_;
    GlobalTensor<float> mu1GM_;
    GlobalTensor<float> mu2GM_;
    GlobalTensor<float> s11GM_;
    GlobalTensor<float> s22GM_;
    GlobalTensor<float> s12GM_;
    GlobalTensor<float> gradImg1GM_;
    GlobalTensor<float> gradImg2GM_;
    GlobalTensor<float> wsGM_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> scratchBuf_;
    TBuf<TPosition::VECCALC> kBuf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    LocalTensor<float> weightL_;
    LocalTensor<float> inputL_;
    LocalTensor<float> scratchL_;
    LocalTensor<float> kL_;
    LocalTensor<float> vL_;
    LocalTensor<float> hL_;
    LocalTensor<float> imgL_;
    LocalTensor<float> tmpL_;

    int64_t blockIdx_;
    int64_t needCoreNum_;
    int64_t bcNum_;
    int64_t height_;
    int64_t width_;
    int64_t windowSize_;
    int64_t radius_;
    int64_t coresPerImage_;
    int64_t rowsPerCore_;
    int64_t blockRows_;
    int64_t wAlign_;
    int64_t padLen_;
    int64_t sysWsBytes_;
    int64_t perCoreWsFloats_;
    int64_t needGradImg2_;
    int64_t numKMaps_;
    int64_t chunkW_;
    int64_t numWChunks_;
    int64_t maxEffW_;
    int64_t c0_;
    int64_t c1_;
    int64_t outW_;
    int64_t lc0_;
    int64_t loadW_;
    int64_t dataOff_;
    DataCopyExtParams rowLoadParams_;
    DataCopyExtParams rowStoreParams_;
    DataCopyPadExtParams<float> rowPadParams_;
    float wv_[MAX_WS];
};

extern "C" __global__ __aicore__ void fused_ssim_backward(GM_ADDR grad_ssim_map,
                                                           GM_ADDR img1, GM_ADDR img2, GM_ADDR weight,
                                                           GM_ADDR mu1, GM_ADDR mu2,
                                                           GM_ADDR s11, GM_ADDR s22, GM_ADDR s12,
                                                           GM_ADDR grad_img1, GM_ADDR grad_img2,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1) || TILING_KEY_IS(2)) {
        FusedSsimBackward op;
        op.Init(grad_ssim_map, img1, img2, weight, mu1, mu2, s11, s22, s12,
                grad_img1, grad_img2, workspace, &tilingData, &pipe);
        op.Process();
    }
}
