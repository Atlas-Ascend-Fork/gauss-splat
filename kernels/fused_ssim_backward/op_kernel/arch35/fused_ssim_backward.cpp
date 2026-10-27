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
constexpr uint32_t VL = 64;

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

        pipe_ = pipe;
        // inputL_ reused for prefetch (EID2); imgL_ separate.
        pipe_->InitBuffer(weightBuf_, MAX_WS * FLOAT_SIZE);
        pipe_->InitBuffer(inputBuf_, 6 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(imgBuf_, 2 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(gradScratchBuf_, 2 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(ringBuf_, blockRows_ * 3 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(vBuf_, 3 * padLen_ * FLOAT_SIZE);
        pipe_->InitBuffer(hBuf_, 3 * wAlign_ * FLOAT_SIZE);

        weightL_ = weightBuf_.Get<float>();
        inputL_ = inputBuf_.Get<float>();
        imgL_ = imgBuf_.Get<float>();
        gradScratchL_ = gradScratchBuf_.Get<float>();
        ringL_ = ringBuf_.Get<float>();
        vL_ = vBuf_.Get<float>();
        hL_ = hBuf_.Get<float>();
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

        curMode_ = 0;
        ProcessChunkLoop(imgBase, h0, h1);
        if (needGradImg2_ == 1) {
            curMode_ = 1;
            ProcessChunkLoop(imgBase, h0, h1);
        }
    }

private:
    __aicore__ inline void SetChunk(int64_t j)
    {
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
    }

    __aicore__ inline void ProcessChunkLoop(int64_t imgBase, int64_t h0, int64_t h1)
    {
        for (int64_t j = 0; j < numWChunks_; j++) {
            SetChunk(j);
            ProcessRowReuse(imgBase, h0, h1);
        }
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

    // Async prefetch on MTE2 (EID2); phantom rows skip load.
    __aicore__ inline void PrefetchInputRow(int64_t imgBase, int64_t gmRow)
    {
        if (gmRow < 0 || gmRow >= height_) {
            return;
        }
        int64_t row_off = imgBase + gmRow * width_ + lc0_;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
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
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    }

    // Commit prefetched input into ring slot; phantom row zeros slot (no wait).
    __aicore__ inline void CommitRingFromPrefetch(int64_t gmRow, int64_t physSlot)
    {
        LocalTensor<float> rb = ringL_[physSlot * 3 * wAlign_];
        if (gmRow < 0 || gmRow >= height_) {
            Duplicate(rb[0 * wAlign_], ZERO_F, wAlign_);
            Duplicate(rb[1 * wAlign_], ZERO_F, wAlign_);
            Duplicate(rb[2 * wAlign_], ZERO_F, wAlign_);
            return;
        }
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        ComputeKVecReg(rb);
    }

    __aicore__ inline void FillRingSlot(int64_t imgBase, int64_t gmRow, int64_t physSlot)
    {
        LocalTensor<float> rb = ringL_[physSlot * 3 * wAlign_];
        if (gmRow < 0 || gmRow >= height_) {
            Duplicate(rb[0 * wAlign_], ZERO_F, wAlign_);
            Duplicate(rb[1 * wAlign_], ZERO_F, wAlign_);
            Duplicate(rb[2 * wAlign_], ZERO_F, wAlign_);
            return;
        }
        LoadInputRow(imgBase, gmRow);
        ComputeKVecReg(rb);
    }

    // Sliding-window ring with software pipelining.
    __aicore__ inline void ProcessRowReuse(int64_t imgBase, int64_t h0, int64_t h1)
    {
        rotateBase_ = 0;
        for (int64_t k = 0; k < blockRows_; k++) {
            FillRingSlot(imgBase, h0 - radius_ + k, k);
        }
        for (int64_t h = h0; h < h1; h++) {
            if (h > h0) {
                CommitRingFromPrefetch(h + radius_, rotateBase_);
                rotateBase_ = (rotateBase_ + 1) % blockRows_;
            }
            VerticalConvFromRing();
            if (h + 1 < h1) {
                PrefetchInputRow(imgBase, (h + 1) + radius_);
            }
            PrefetchImg(imgBase, h);
            HorizontalConvReg();
            ComputeGradImgReg(imgBase, h);
        }
    }

    __aicore__ inline void VerticalConvFromRing()
    {
        for (int32_t slot = 0; slot < 3; slot++) {
            Duplicate(vL_[slot * padLen_], ZERO_F, padLen_);
        }
        __ubuf__ float *ringBase = reinterpret_cast<__ubuf__ float *>(ringL_.GetPhyAddr());
        __ubuf__ float *vBase = reinterpret_cast<__ubuf__ float *>(vL_.GetPhyAddr());

        const int32_t slot0 = static_cast<int32_t>((rotateBase_ + 0) % blockRows_);
        const int32_t slot1 = static_cast<int32_t>((rotateBase_ + 1) % blockRows_);
        const int32_t slot2 = static_cast<int32_t>((rotateBase_ + 2) % blockRows_);
        const int32_t slot3 = static_cast<int32_t>((rotateBase_ + 3) % blockRows_);
        const int32_t slot4 = static_cast<int32_t>((rotateBase_ + 4) % blockRows_);
        const int32_t slot5 = static_cast<int32_t>((rotateBase_ + 5) % blockRows_);
        const int32_t slot6 = static_cast<int32_t>((rotateBase_ + 6) % blockRows_);
        const int32_t slot7 = static_cast<int32_t>((rotateBase_ + 7) % blockRows_);
        const int32_t slot8 = static_cast<int32_t>((rotateBase_ + 8) % blockRows_);
        const int32_t slot9 = static_cast<int32_t>((rotateBase_ + 9) % blockRows_);
        const int32_t slot10 = static_cast<int32_t>((rotateBase_ + 10) % blockRows_);
        const int32_t slot11 = static_cast<int32_t>((rotateBase_ + 11) % blockRows_);
        const int32_t slot12 = static_cast<int32_t>((rotateBase_ + 12) % blockRows_);
        const int32_t slot13 = static_cast<int32_t>((rotateBase_ + 13) % blockRows_);
        const int32_t slot14 = static_cast<int32_t>((rotateBase_ + 14) % blockRows_);

        using DCB = AscendC::MicroAPI::DataCopyMode;
        #define SSIM_BVROW(JJ)                                                                       \
            do {                                                                                      \
                __ubuf__ float *pa = ringBase + slot##JJ * 3 * wAlign_ + 0 * wAlign_ + off;          \
                __ubuf__ float *pb = ringBase + slot##JJ * 3 * wAlign_ + 1 * wAlign_ + off;          \
                __ubuf__ float *pg = ringBase + slot##JJ * 3 * wAlign_ + 2 * wAlign_ + off;          \
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(ra, pa, 1, maskCnt);         \
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(rb, pb, 1, maskCnt);         \
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(rg, pg, 1, maskCnt);         \
                AscendC::MicroAPI::Axpy(acc_a, ra, wv_[(JJ)], maskCnt);                               \
                AscendC::MicroAPI::Axpy(acc_b, rb, wv_[(JJ)], maskCnt);                               \
                AscendC::MicroAPI::Axpy(acc_g, rg, wv_[(JJ)], maskCnt);                               \
            } while (0)

        const uint16_t numChunks = static_cast<uint16_t>((loadW_ + VL - 1) / VL);
        uint32_t remaining = static_cast<uint32_t>(loadW_);

        if (blockRows_ == 7) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg maskCnt;
                AscendC::MicroAPI::RegTensor<float> ra, rb, rg;
                AscendC::MicroAPI::RegTensor<float> acc_a, acc_b, acc_g;
                for (uint16_t c = 0; c < numChunks; ++c) {
                    const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                    maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                    AscendC::MicroAPI::Duplicate(acc_a, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_b, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_g, ZERO_F, maskCnt);
                    SSIM_BVROW(0); SSIM_BVROW(1); SSIM_BVROW(2); SSIM_BVROW(3);
                    SSIM_BVROW(4); SSIM_BVROW(5); SSIM_BVROW(6);
                    AscendC::MicroAPI::DataCopy(vBase + 0 * padLen_ + HALO_LEFT + off, acc_a, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 1 * padLen_ + HALO_LEFT + off, acc_b, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 2 * padLen_ + HALO_LEFT + off, acc_g, maskCnt);
                }
            }
        } else if (blockRows_ == 11) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg maskCnt;
                AscendC::MicroAPI::RegTensor<float> ra, rb, rg;
                AscendC::MicroAPI::RegTensor<float> acc_a, acc_b, acc_g;
                for (uint16_t c = 0; c < numChunks; ++c) {
                    const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                    maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                    AscendC::MicroAPI::Duplicate(acc_a, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_b, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_g, ZERO_F, maskCnt);
                    SSIM_BVROW(0); SSIM_BVROW(1); SSIM_BVROW(2); SSIM_BVROW(3);
                    SSIM_BVROW(4); SSIM_BVROW(5); SSIM_BVROW(6); SSIM_BVROW(7);
                    SSIM_BVROW(8); SSIM_BVROW(9); SSIM_BVROW(10);
                    AscendC::MicroAPI::DataCopy(vBase + 0 * padLen_ + HALO_LEFT + off, acc_a, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 1 * padLen_ + HALO_LEFT + off, acc_b, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 2 * padLen_ + HALO_LEFT + off, acc_g, maskCnt);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg maskCnt;
                AscendC::MicroAPI::RegTensor<float> ra, rb, rg;
                AscendC::MicroAPI::RegTensor<float> acc_a, acc_b, acc_g;
                for (uint16_t c = 0; c < numChunks; ++c) {
                    const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                    maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                    AscendC::MicroAPI::Duplicate(acc_a, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_b, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_g, ZERO_F, maskCnt);
                    SSIM_BVROW(0); SSIM_BVROW(1); SSIM_BVROW(2); SSIM_BVROW(3);
                    SSIM_BVROW(4); SSIM_BVROW(5); SSIM_BVROW(6); SSIM_BVROW(7);
                    SSIM_BVROW(8); SSIM_BVROW(9); SSIM_BVROW(10); SSIM_BVROW(11);
                    SSIM_BVROW(12); SSIM_BVROW(13); SSIM_BVROW(14);
                    AscendC::MicroAPI::DataCopy(vBase + 0 * padLen_ + HALO_LEFT + off, acc_a, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 1 * padLen_ + HALO_LEFT + off, acc_b, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 2 * padLen_ + HALO_LEFT + off, acc_g, maskCnt);
                }
            }
        }
        #undef SSIM_BVROW
    }

    __aicore__ inline void ComputeKVecReg(LocalTensor<float> kOut)
    {
        __ubuf__ float *inBase = reinterpret_cast<__ubuf__ float *>(inputL_.GetPhyAddr());
        __ubuf__ float *kBase = reinterpret_cast<__ubuf__ float *>(kOut.GetPhyAddr());
        const int64_t mode = curMode_;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg maskCnt;
            AscendC::MicroAPI::RegTensor<float> mu1, mu2, ra, rb, rc, rd, w0, w1, w2;

            const uint16_t numChunks = static_cast<uint16_t>((loadW_ + VL - 1) / VL);
            uint32_t remaining = static_cast<uint32_t>(loadW_);
            using DCB = AscendC::MicroAPI::DataCopyMode;

            for (uint16_t c = 0; c < numChunks; ++c) {
                const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);

                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(mu1, inBase + 0 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(mu2, inBase + 1 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(ra, inBase + 2 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(rb, inBase + 3 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(rc, inBase + 4 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(rd, inBase + 5 * wAlign_ + off, 1, maskCnt);

                AscendC::MicroAPI::Mul(w0, mu1, mu1, maskCnt);
                AscendC::MicroAPI::Mul(w1, mu2, mu2, maskCnt);
                AscendC::MicroAPI::Sub(ra, ra, w0, maskCnt);
                AscendC::MicroAPI::Maxs(ra, ra, ZERO_F, maskCnt);
                AscendC::MicroAPI::Sub(rb, rb, w1, maskCnt);
                AscendC::MicroAPI::Maxs(rb, rb, ZERO_F, maskCnt);
                AscendC::MicroAPI::Mul(w2, mu1, mu2, maskCnt);
                AscendC::MicroAPI::Sub(rc, rc, w2, maskCnt);
                AscendC::MicroAPI::Add(w0, w0, w1, maskCnt);
                AscendC::MicroAPI::Adds(w0, w0, SSIM_C1, maskCnt);
                AscendC::MicroAPI::Muls(w2, w2, TWO_F, maskCnt);
                AscendC::MicroAPI::Adds(w2, w2, SSIM_C1, maskCnt);
                AscendC::MicroAPI::Add(ra, ra, rb, maskCnt);
                AscendC::MicroAPI::Adds(ra, ra, SSIM_C2, maskCnt);
                AscendC::MicroAPI::Muls(rc, rc, TWO_F, maskCnt);
                AscendC::MicroAPI::Adds(rc, rc, SSIM_C2, maskCnt);
                AscendC::MicroAPI::Mul(w1, w0, ra, maskCnt);
                AscendC::MicroAPI::Mul(rb, w2, rc, maskCnt);
                AscendC::MicroAPI::Div(rb, rb, w1, maskCnt);
                AscendC::MicroAPI::Muls(rd, rd, TWO_F, maskCnt);
                AscendC::MicroAPI::Div(rd, rd, w1, maskCnt);
                AscendC::MicroAPI::Sub(ra, ra, w0, maskCnt);
                AscendC::MicroAPI::Sub(rc, rc, w2, maskCnt);
                AscendC::MicroAPI::Mul(w1, rd, w2, maskCnt);
                AscendC::MicroAPI::DataCopy(kBase + 1 * wAlign_ + off, w1, maskCnt);
                AscendC::MicroAPI::Mul(w2, rd, rb, maskCnt);
                AscendC::MicroAPI::Mul(w2, w2, w0, maskCnt);
                AscendC::MicroAPI::Muls(w2, w2, NEG_ONE_F, maskCnt);
                AscendC::MicroAPI::DataCopy(kBase + 2 * wAlign_ + off, w2, maskCnt);
                if (mode == 0) {
                    AscendC::MicroAPI::Mul(w0, mu2, rc, maskCnt);
                    AscendC::MicroAPI::Mul(w1, mu1, ra, maskCnt);
                } else {
                    AscendC::MicroAPI::Mul(w0, mu1, rc, maskCnt);
                    AscendC::MicroAPI::Mul(w1, mu2, ra, maskCnt);
                }
                AscendC::MicroAPI::Mul(w1, w1, rb, maskCnt);
                AscendC::MicroAPI::Sub(w0, w0, w1, maskCnt);
                AscendC::MicroAPI::Mul(w0, w0, rd, maskCnt);
                AscendC::MicroAPI::DataCopy(kBase + 0 * wAlign_ + off, w0, maskCnt);
            }
        }
    }

    __aicore__ inline void HorizontalConvReg()
    {
        __ubuf__ float *vBase = reinterpret_cast<__ubuf__ float *>(vL_.GetPhyAddr());
        __ubuf__ float *hBase = reinterpret_cast<__ubuf__ float *>(hL_.GetPhyAddr());

        #define SSIM_BTAP(DX)                                                 \
            do {                                                              \
                __ubuf__ float *p_##DX = vBase + srcBase + off + (DX);        \
                AscendC::MicroAPI::DataCopyUnAlignPre(ureg, p_##DX);          \
                AscendC::MicroAPI::DataCopyUnAlign(src, ureg, p_##DX);        \
                AscendC::MicroAPI::Axpy(acc, src, wv_[(DX)], maskCnt);        \
            } while (0)

        const uint16_t numChunks = static_cast<uint16_t>((outW_ + VL - 1) / VL);
        for (uint16_t k = 0; k < 3; ++k) {
            const int64_t srcBase = k * padLen_ + HALO_LEFT + dataOff_ - radius_;
            const int64_t dstBase = k * wAlign_;
            uint32_t remaining = static_cast<uint32_t>(outW_);

            if (blockRows_ == 7) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::MaskReg maskCnt;
                    AscendC::MicroAPI::UnalignReg ureg;
                    AscendC::MicroAPI::RegTensor<float> acc, src;
                    for (uint16_t c = 0; c < numChunks; ++c) {
                        const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                        maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                        AscendC::MicroAPI::Duplicate(acc, ZERO_F, maskCnt);
                        SSIM_BTAP(0); SSIM_BTAP(1); SSIM_BTAP(2); SSIM_BTAP(3);
                        SSIM_BTAP(4); SSIM_BTAP(5); SSIM_BTAP(6);
                        AscendC::MicroAPI::DataCopy(hBase + dstBase + off, acc, maskCnt);
                    }
                }
            } else if (blockRows_ == 11) {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::MaskReg maskCnt;
                    AscendC::MicroAPI::UnalignReg ureg;
                    AscendC::MicroAPI::RegTensor<float> acc, src;
                    for (uint16_t c = 0; c < numChunks; ++c) {
                        const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                        maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                        AscendC::MicroAPI::Duplicate(acc, ZERO_F, maskCnt);
                        SSIM_BTAP(0); SSIM_BTAP(1); SSIM_BTAP(2); SSIM_BTAP(3);
                        SSIM_BTAP(4); SSIM_BTAP(5); SSIM_BTAP(6); SSIM_BTAP(7);
                        SSIM_BTAP(8); SSIM_BTAP(9); SSIM_BTAP(10);
                        AscendC::MicroAPI::DataCopy(hBase + dstBase + off, acc, maskCnt);
                    }
                }
            } else {
                __VEC_SCOPE__
                {
                    AscendC::MicroAPI::MaskReg maskCnt;
                    AscendC::MicroAPI::UnalignReg ureg;
                    AscendC::MicroAPI::RegTensor<float> acc, src;
                    for (uint16_t c = 0; c < numChunks; ++c) {
                        const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                        maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                        AscendC::MicroAPI::Duplicate(acc, ZERO_F, maskCnt);
                        SSIM_BTAP(0); SSIM_BTAP(1); SSIM_BTAP(2); SSIM_BTAP(3);
                        SSIM_BTAP(4); SSIM_BTAP(5); SSIM_BTAP(6); SSIM_BTAP(7);
                        SSIM_BTAP(8); SSIM_BTAP(9); SSIM_BTAP(10); SSIM_BTAP(11);
                        SSIM_BTAP(12); SSIM_BTAP(13); SSIM_BTAP(14);
                        AscendC::MicroAPI::DataCopy(hBase + dstBase + off, acc, maskCnt);
                    }
                }
            }
        }
        #undef SSIM_BTAP
    }

    // Async img load (EID1), overlaps HorizontalConvReg.
    __aicore__ inline void PrefetchImg(int64_t imgBase, int64_t h)
    {
        int64_t row_off = imgBase + h * width_ + c0_;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        DataCopyPad(imgL_[0 * wAlign_], img1GM_[row_off], rowStoreParams_, rowPadParams_);
        DataCopyPad(imgL_[1 * wAlign_], img2GM_[row_off], rowStoreParams_, rowPadParams_);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    }

    __aicore__ inline void ComputeGradImgReg(int64_t imgBase, int64_t h)
    {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        __ubuf__ float *hBase = reinterpret_cast<__ubuf__ float *>(hL_.GetPhyAddr());
        __ubuf__ float *imgBase_ = reinterpret_cast<__ubuf__ float *>(imgL_.GetPhyAddr());
        __ubuf__ float *gBase = reinterpret_cast<__ubuf__ float *>(gradScratchL_.GetPhyAddr());
        const int64_t mode = curMode_;

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg maskCnt;
            AscendC::MicroAPI::RegTensor<float> img1, img2, cKa, cKb, cKg, gi, tmp;

            const uint16_t numChunks = static_cast<uint16_t>((outW_ + VL - 1) / VL);
            uint32_t remaining = static_cast<uint32_t>(outW_);
            using DCB = AscendC::MicroAPI::DataCopyMode;

            for (uint16_t c = 0; c < numChunks; ++c) {
                const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);

                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(img1, imgBase_ + 0 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(img2, imgBase_ + 1 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(cKa, hBase + 0 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(cKb, hBase + 1 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, DCB::DATA_BLOCK_COPY>(cKg, hBase + 2 * wAlign_ + off, 1, maskCnt);

                if (mode == 0) {
                    AscendC::MicroAPI::Mul(gi, img2, cKb, maskCnt);
                    AscendC::MicroAPI::Add(gi, gi, cKa, maskCnt);
                    AscendC::MicroAPI::Mul(tmp, img1, cKg, maskCnt);
                } else {
                    AscendC::MicroAPI::Mul(gi, img1, cKb, maskCnt);
                    AscendC::MicroAPI::Add(gi, gi, cKa, maskCnt);
                    AscendC::MicroAPI::Mul(tmp, img2, cKg, maskCnt);
                }
                AscendC::MicroAPI::Add(gi, gi, tmp, maskCnt);
                AscendC::MicroAPI::DataCopy(gBase + 0 * wAlign_ + off, gi, maskCnt);
            }
        }

        int64_t row_off = imgBase + h * width_ + c0_;
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        if (mode == 0) {
            DataCopyPad(gradImg1GM_[row_off], gradScratchL_[0 * wAlign_], rowStoreParams_);
        } else {
            DataCopyPad(gradImg2GM_[row_off], gradScratchL_[0 * wAlign_], rowStoreParams_);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

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
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> imgBuf_;
    TBuf<TPosition::VECCALC> gradScratchBuf_;
    TBuf<TPosition::VECCALC> ringBuf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> hBuf_;
    LocalTensor<float> weightL_;
    LocalTensor<float> inputL_;
    LocalTensor<float> imgL_;
    LocalTensor<float> gradScratchL_;
    LocalTensor<float> ringL_;
    LocalTensor<float> vL_;
    LocalTensor<float> hL_;

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
    int64_t chunkW_;
    int64_t numWChunks_;
    int64_t maxEffW_;
    int64_t rotateBase_;
    int64_t c0_;
    int64_t c1_;
    int64_t outW_;
    int64_t lc0_;
    int64_t loadW_;
    int64_t dataOff_;
    DataCopyExtParams rowLoadParams_;
    DataCopyExtParams rowStoreParams_;
    DataCopyPadExtParams<float> rowPadParams_;
    int64_t curMode_;
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
