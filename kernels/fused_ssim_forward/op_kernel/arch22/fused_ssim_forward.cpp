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
constexpr float SSIM_C2 = 0.0009f;      // 0.03^2
constexpr float TWO_F = 2.0f;
constexpr float ZERO_F = 0.0f;

__aicore__ inline int64_t AlignUp8(int64_t x)
{
    return (x + 7) / 8 * 8;
}
} // namespace

class FusedSsimForward {
public:
    __aicore__ inline FusedSsimForward() {}

    __aicore__ inline void Init(GM_ADDR img1, GM_ADDR img2, GM_ADDR weight,
                                GM_ADDR ssimMap, GM_ADDR mu1Out, GM_ADDR mu2Out,
                                GM_ADDR s11Out, GM_ADDR s22Out, GM_ADDR s12Out,
                                GM_ADDR workspace,
                                const FusedSsimForwardTilingData *tiling, TPipe *pipe)
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
        chunkW_ = tiling->chunkW;
        numWChunks_ = tiling->numWChunks;

        blockRows_ = 2 * radius_ + 1;
        maxEffW_ = chunkW_ + 2 * HALO_ALIGN;
        wAlign_ = AlignUp8(maxEffW_);
        padLen_ = AlignUp8(HALO_LEFT + maxEffW_ + radius_);

        img1GM_.SetGlobalBuffer((__gm__ float *)img1);
        img2GM_.SetGlobalBuffer((__gm__ float *)img2);
        weightGM_.SetGlobalBuffer((__gm__ float *)weight);
        ssimGM_.SetGlobalBuffer((__gm__ float *)ssimMap);
        mu1GM_.SetGlobalBuffer((__gm__ float *)mu1Out);
        mu2GM_.SetGlobalBuffer((__gm__ float *)mu2Out);
        s11GM_.SetGlobalBuffer((__gm__ float *)s11Out);
        s22GM_.SetGlobalBuffer((__gm__ float *)s22Out);
        s12GM_.SetGlobalBuffer((__gm__ float *)s12Out);

        __gm__ uint8_t *wsBase = workspace + sysWsBytes_
            + blockIdx_ * perCoreWsFloats_ * static_cast<int64_t>(sizeof(float));
        wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(wsBase));

        pipe_ = pipe;
        pipe_->InitBuffer(weightBuf_, MAX_WS * FLOAT_SIZE);
        pipe_->InitBuffer(block1Buf_, blockRows_ * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(block2Buf_, blockRows_ * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(vBuf_, 5 * padLen_ * FLOAT_SIZE);
        pipe_->InitBuffer(hBuf_, 5 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(tmpBuf_, 5 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(offsetBuf_, wAlign_ * static_cast<int64_t>(sizeof(uint32_t)));

        weightL_ = weightBuf_.Get<float>();
        block1L_ = block1Buf_.Get<float>();
        block2L_ = block2Buf_.Get<float>();
        vL_ = vBuf_.Get<float>();
        hL_ = hBuf_.Get<float>();
        tmpL_ = tmpBuf_.Get<float>();
        offsetL_ = offsetBuf_.Get<uint32_t>();
        rowPadParams_ = DataCopyPadExtParams<float>{false, 0, 0, 0.0f};
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= needCoreNum_) {
            return;
        }

        DataCopy(weightL_, weightGM_, MAX_WS);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int32_t k = 0; k < windowSize_; k++) {
            wv_[k] = weightL_.GetValue(k);
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
                LoadBlock(imgBase, h);
                VerticalConv();
                HorizontalConvViaGm();
                CombineAndStore(imgBase, h);
            }
        }
    }

private:
    __aicore__ inline void LoadBlock(int64_t imgBase, int64_t h)
    {
        int64_t rowLo = h - radius_;
        int64_t rowHi = h + radius_;
        if (rowLo < 0) {
            rowLo = 0;
        }
        if (rowHi > height_ - 1) {
            rowHi = height_ - 1;
        }
        int64_t validRows = rowHi - rowLo + 1;
        int64_t dstRow = rowLo - (h - radius_);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        Duplicate(block1L_, ZERO_F, blockRows_ * wAlign_);
        Duplicate(block2L_, ZERO_F, blockRows_ * wAlign_);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        for (int64_t k = 0; k < blockRows_; k++) {
            int64_t gmRow = rowLo + k;
            int64_t ubRow = dstRow + k;
            int64_t row_off = imgBase + gmRow * width_ + lc0_;
            DataCopyPad(block1L_[ubRow * wAlign_], img1GM_[row_off], rowLoadParams_, rowPadParams_);
            DataCopyPad(block2L_[ubRow * wAlign_], img2GM_[row_off], rowLoadParams_, rowPadParams_);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void VerticalConv()
    {
        for (int32_t k = 0; k < 5; k++) {
            Duplicate(vL_[k * padLen_], ZERO_F, padLen_);
        }
        LocalTensor<float> sq1 = tmpL_[0 * wAlign_];
        LocalTensor<float> sq2 = tmpL_[1 * wAlign_];
        LocalTensor<float> sq12 = tmpL_[2 * wAlign_];
        LocalTensor<float> scratch = tmpL_[3 * wAlign_];
        for (int32_t k = 0; k < blockRows_; k++) {
            LocalTensor<float> r1 = block1L_[k * wAlign_];
            LocalTensor<float> r2 = block2L_[k * wAlign_];
            Mul(sq1, r1, r1, loadW_);
            Mul(sq2, r2, r2, loadW_);
            Mul(sq12, r1, r2, loadW_);
            float w = wv_[k];
            Muls(scratch, r1, w, loadW_);
            Add(vL_[0 * padLen_ + HALO_LEFT], vL_[0 * padLen_ + HALO_LEFT], scratch, loadW_);
            Muls(scratch, r2, w, loadW_);
            Add(vL_[1 * padLen_ + HALO_LEFT], vL_[1 * padLen_ + HALO_LEFT], scratch, loadW_);
            Muls(scratch, sq1, w, loadW_);
            Add(vL_[2 * padLen_ + HALO_LEFT], vL_[2 * padLen_ + HALO_LEFT], scratch, loadW_);
            Muls(scratch, sq2, w, loadW_);
            Add(vL_[3 * padLen_ + HALO_LEFT], vL_[3 * padLen_ + HALO_LEFT], scratch, loadW_);
            Muls(scratch, sq12, w, loadW_);
            Add(vL_[4 * padLen_ + HALO_LEFT], vL_[4 * padLen_ + HALO_LEFT], scratch, loadW_);
        }
    }

    __aicore__ inline void HorizontalConvViaGm()
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopy(wsGM_, vL_, static_cast<uint32_t>(perCoreWsFloats_));
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        LocalTensor<float> tmp = tmpL_[0 * wAlign_];
        for (int32_t k = 0; k < 5; k++) {
            Duplicate(hL_[k * wAlign_], ZERO_F, wAlign_);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            int64_t baseAligned = k * padLen_ + HALO_LEFT + dataOff_;
            for (int32_t dx = 0; dx < windowSize_; dx++) {
                int64_t srcElemOffset = baseAligned - radius_ + dx;
                DataCopy(tmp, wsGM_[srcElemOffset], static_cast<uint32_t>(wAlign_));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                Muls(tmp, tmp, wv_[dx], outW_);
                Add(hL_[k * wAlign_], hL_[k * wAlign_], tmp, outW_);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }
        }
    }

    __aicore__ inline void CombineAndStore(int64_t imgBase, int64_t h)
    {
        LocalTensor<float> mu1 = hL_[0 * wAlign_];
        LocalTensor<float> mu2 = hL_[1 * wAlign_];
        LocalTensor<float> s11 = hL_[2 * wAlign_];
        LocalTensor<float> s22 = hL_[3 * wAlign_];
        LocalTensor<float> s12 = hL_[4 * wAlign_];

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        int64_t row_off = imgBase + h * width_ + c0_;
        DataCopyPad(mu1GM_[row_off], mu1, rowStoreParams_);
        DataCopyPad(mu2GM_[row_off], mu2, rowStoreParams_);
        DataCopyPad(s11GM_[row_off], s11, rowStoreParams_);
        DataCopyPad(s22GM_[row_off], s22, rowStoreParams_);
        DataCopyPad(s12GM_[row_off], s12, rowStoreParams_);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        LocalTensor<float> m11 = tmpL_[0 * wAlign_];
        LocalTensor<float> m22 = tmpL_[1 * wAlign_];
        LocalTensor<float> m12 = tmpL_[2 * wAlign_];
        LocalTensor<float> numer = tmpL_[3 * wAlign_];
        LocalTensor<float> denom = tmpL_[4 * wAlign_];

        Mul(m11, mu1, mu1, outW_);
        Mul(m22, mu2, mu2, outW_);
        Mul(m12, mu1, mu2, outW_);
        Sub(s11, s11, m11, outW_);
        Sub(s22, s22, m22, outW_);
        Sub(s12, s12, m12, outW_);
        Maxs(s11, s11, ZERO_F, outW_);
        Maxs(s22, s22, ZERO_F, outW_);

        Muls(numer, m12, TWO_F, outW_);
        Adds(numer, numer, SSIM_C1, outW_);
        Muls(s12, s12, TWO_F, outW_);
        Adds(s12, s12, SSIM_C2, outW_);
        Mul(numer, numer, s12, outW_);

        Add(denom, m11, m22, outW_);
        Adds(denom, denom, SSIM_C1, outW_);
        Add(s11, s11, s22, outW_);
        Adds(s11, s11, SSIM_C2, outW_);
        Mul(denom, denom, s11, outW_);

        Div(numer, numer, denom, outW_);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(ssimGM_[imgBase + h * width_ + c0_], numer, rowStoreParams_);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    TPipe *pipe_;
    GlobalTensor<float> img1GM_;
    GlobalTensor<float> img2GM_;
    GlobalTensor<float> weightGM_;
    GlobalTensor<float> ssimGM_;
    GlobalTensor<float> mu1GM_;
    GlobalTensor<float> mu2GM_;
    GlobalTensor<float> s11GM_;
    GlobalTensor<float> s22GM_;
    GlobalTensor<float> s12GM_;
    GlobalTensor<float> wsGM_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> block1Buf_;
    TBuf<TPosition::VECCALC> block2Buf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> hBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    LocalTensor<float> weightL_;
    LocalTensor<float> block1L_;
    LocalTensor<float> block2L_;
    LocalTensor<float> vL_;
    LocalTensor<float> hL_;
    LocalTensor<float> tmpL_;
    LocalTensor<uint32_t> offsetL_;

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

extern "C" __global__ __aicore__ void fused_ssim_forward(GM_ADDR img1, GM_ADDR img2, GM_ADDR weight,
                                                         GM_ADDR ssim_map,
                                                         GM_ADDR mu1, GM_ADDR mu2,
                                                         GM_ADDR s11, GM_ADDR s22, GM_ADDR s12,
                                                         GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        FusedSsimForward op;
        op.Init(img1, img2, weight, ssim_map, mu1, mu2, s11, s22, s12,
                workspace, &tilingData, &pipe);
        op.Process();
    }
}
