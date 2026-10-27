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

        pipe_ = pipe;
        pipe_->InitBuffer(weightBuf_, MAX_WS * FLOAT_SIZE);
        pipe_->InitBuffer(block1Buf_, blockRows_ * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(block2Buf_, blockRows_ * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(vBuf_, 5 * padLen_ * FLOAT_SIZE);
        pipe_->InitBuffer(hBuf_, 5 * wAlign_ * FLOAT_SIZE);
        pipe_->InitBuffer(tmpBuf_, 5 * wAlign_ * FLOAT_SIZE);

        weightL_ = weightBuf_.Get<float>();
        block1L_ = block1Buf_.Get<float>();
        block2L_ = block2Buf_.Get<float>();
        vL_ = vBuf_.Get<float>();
        hL_ = hBuf_.Get<float>();
        tmpL_ = tmpBuf_.Get<float>();
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
                if (h == h0) {
                    LoadBlockFirst(imgBase, h);
                } else {
                    LoadBlockIncremental(imgBase, h);
                }
                VerticalConvReg();
                HorizontalConvReg();
                CombineAndStore(imgBase, h);
            }
        }
    }

private:
    __aicore__ inline void LoadBlockFirst(int64_t imgBase, int64_t h)
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        Duplicate(block1L_, ZERO_F, blockRows_ * wAlign_);
        Duplicate(block2L_, ZERO_F, blockRows_ * wAlign_);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        for (int64_t k = 0; k < blockRows_; k++) {
            int64_t gmRow = h - radius_ + k;
            if (gmRow < 0 || gmRow >= height_) continue;
            int64_t row_off = imgBase + gmRow * width_ + lc0_;
            DataCopyPad(block1L_[k * wAlign_], img1GM_[row_off], rowLoadParams_, rowPadParams_);
            DataCopyPad(block2L_[k * wAlign_], img2GM_[row_off], rowLoadParams_, rowPadParams_);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        rotateBase_ = 0;
    }

    __aicore__ inline void LoadBlockIncremental(int64_t imgBase, int64_t h)
    {
        int64_t dropSlot = rotateBase_;
        int64_t newGmRow = h + radius_;

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        if (newGmRow < 0 || newGmRow >= height_) {
            Duplicate(block1L_[dropSlot * wAlign_], ZERO_F, wAlign_);
            Duplicate(block2L_[dropSlot * wAlign_], ZERO_F, wAlign_);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        } else {
            int64_t row_off = imgBase + newGmRow * width_ + lc0_;
            DataCopyPad(block1L_[dropSlot * wAlign_], img1GM_[row_off], rowLoadParams_, rowPadParams_);
            DataCopyPad(block2L_[dropSlot * wAlign_], img2GM_[row_off], rowLoadParams_, rowPadParams_);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }
        rotateBase_ = (rotateBase_ + 1) % blockRows_;
    }

    __aicore__ inline void VerticalConvReg()
    {
        constexpr uint32_t VL = 64;
        for (int32_t k = 0; k < 5; k++) {
            Duplicate(vL_[k * padLen_], ZERO_F, padLen_);
        }
        __ubuf__ float *block1Base = reinterpret_cast<__ubuf__ float *>(block1L_.GetPhyAddr());
        __ubuf__ float *block2Base = reinterpret_cast<__ubuf__ float *>(block2L_.GetPhyAddr());
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

        #define SSIM_VROW(JJ)                                                                            \
            do {                                                                                          \
                __ubuf__ float *p_r1_##JJ = block1Base + slot##JJ * wAlign_ + off;                        \
                __ubuf__ float *p_r2_##JJ = block2Base + slot##JJ * wAlign_ + off;                        \
                AscendC::MicroAPI::DataCopy<float,                                                        \
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r1, p_r1_##JJ, 1, maskCnt);         \
                AscendC::MicroAPI::DataCopy<float,                                                        \
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r2, p_r2_##JJ, 1, maskCnt);         \
                AscendC::MicroAPI::Mul(sq1, r1, r1, maskCnt);                                             \
                AscendC::MicroAPI::Mul(sq2, r2, r2, maskCnt);                                             \
                AscendC::MicroAPI::Mul(sq12, r1, r2, maskCnt);                                            \
                AscendC::MicroAPI::Axpy(acc_mu1, r1, wv_[(JJ)], maskCnt);                                 \
                AscendC::MicroAPI::Axpy(acc_mu2, r2, wv_[(JJ)], maskCnt);                                 \
                AscendC::MicroAPI::Axpy(acc_s11, sq1, wv_[(JJ)], maskCnt);                                \
                AscendC::MicroAPI::Axpy(acc_s22, sq2, wv_[(JJ)], maskCnt);                                \
                AscendC::MicroAPI::Axpy(acc_s12, sq12, wv_[(JJ)], maskCnt);                               \
            } while (0)

        const uint16_t numChunks = static_cast<uint16_t>((loadW_ + VL - 1) / VL);
        uint32_t remaining = static_cast<uint32_t>(loadW_);

        if (blockRows_ == 7) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg maskCnt;
                AscendC::MicroAPI::RegTensor<float> r1, r2, sq1, sq2, sq12;
                AscendC::MicroAPI::RegTensor<float> acc_mu1, acc_mu2, acc_s11, acc_s22, acc_s12;
                for (uint16_t c = 0; c < numChunks; ++c) {
                    const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                    maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                    AscendC::MicroAPI::Duplicate(acc_mu1, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_mu2, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s11, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s22, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s12, ZERO_F, maskCnt);
                    SSIM_VROW(0); SSIM_VROW(1); SSIM_VROW(2); SSIM_VROW(3);
                    SSIM_VROW(4); SSIM_VROW(5); SSIM_VROW(6);
                    AscendC::MicroAPI::DataCopy(vBase + 0 * padLen_ + HALO_LEFT + off, acc_mu1, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 1 * padLen_ + HALO_LEFT + off, acc_mu2, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 2 * padLen_ + HALO_LEFT + off, acc_s11, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 3 * padLen_ + HALO_LEFT + off, acc_s22, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 4 * padLen_ + HALO_LEFT + off, acc_s12, maskCnt);
                }
            }
        } else if (blockRows_ == 11) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg maskCnt;
                AscendC::MicroAPI::RegTensor<float> r1, r2, sq1, sq2, sq12;
                AscendC::MicroAPI::RegTensor<float> acc_mu1, acc_mu2, acc_s11, acc_s22, acc_s12;
                for (uint16_t c = 0; c < numChunks; ++c) {
                    const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                    maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                    AscendC::MicroAPI::Duplicate(acc_mu1, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_mu2, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s11, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s22, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s12, ZERO_F, maskCnt);
                    SSIM_VROW(0); SSIM_VROW(1); SSIM_VROW(2); SSIM_VROW(3);
                    SSIM_VROW(4); SSIM_VROW(5); SSIM_VROW(6); SSIM_VROW(7);
                    SSIM_VROW(8); SSIM_VROW(9); SSIM_VROW(10);
                    AscendC::MicroAPI::DataCopy(vBase + 0 * padLen_ + HALO_LEFT + off, acc_mu1, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 1 * padLen_ + HALO_LEFT + off, acc_mu2, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 2 * padLen_ + HALO_LEFT + off, acc_s11, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 3 * padLen_ + HALO_LEFT + off, acc_s22, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 4 * padLen_ + HALO_LEFT + off, acc_s12, maskCnt);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg maskCnt;
                AscendC::MicroAPI::RegTensor<float> r1, r2, sq1, sq2, sq12;
                AscendC::MicroAPI::RegTensor<float> acc_mu1, acc_mu2, acc_s11, acc_s22, acc_s12;
                for (uint16_t c = 0; c < numChunks; ++c) {
                    const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                    maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);
                    AscendC::MicroAPI::Duplicate(acc_mu1, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_mu2, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s11, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s22, ZERO_F, maskCnt);
                    AscendC::MicroAPI::Duplicate(acc_s12, ZERO_F, maskCnt);
                    SSIM_VROW(0); SSIM_VROW(1); SSIM_VROW(2); SSIM_VROW(3);
                    SSIM_VROW(4); SSIM_VROW(5); SSIM_VROW(6); SSIM_VROW(7);
                    SSIM_VROW(8); SSIM_VROW(9); SSIM_VROW(10); SSIM_VROW(11);
                    SSIM_VROW(12); SSIM_VROW(13); SSIM_VROW(14);
                    AscendC::MicroAPI::DataCopy(vBase + 0 * padLen_ + HALO_LEFT + off, acc_mu1, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 1 * padLen_ + HALO_LEFT + off, acc_mu2, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 2 * padLen_ + HALO_LEFT + off, acc_s11, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 3 * padLen_ + HALO_LEFT + off, acc_s22, maskCnt);
                    AscendC::MicroAPI::DataCopy(vBase + 4 * padLen_ + HALO_LEFT + off, acc_s12, maskCnt);
                }
            }
        }
        #undef SSIM_VROW
    }

    __aicore__ inline void HorizontalConvReg()
    {
        constexpr uint32_t VL = 64;
        __ubuf__ float *vBase = reinterpret_cast<__ubuf__ float *>(vL_.GetPhyAddr());
        __ubuf__ float *hBase = reinterpret_cast<__ubuf__ float *>(hL_.GetPhyAddr());

        #define SSIM_TAP(DX)                                                  \
            do {                                                              \
                __ubuf__ float *p_##DX = vBase + srcBase + off + (DX);        \
                AscendC::MicroAPI::DataCopyUnAlignPre(ureg, p_##DX);          \
                AscendC::MicroAPI::DataCopyUnAlign(src, ureg, p_##DX);        \
                AscendC::MicroAPI::Axpy(acc, src, wv_[(DX)], maskCnt);        \
            } while (0)

        const uint16_t numChunks = static_cast<uint16_t>((outW_ + VL - 1) / VL);
        for (uint16_t k = 0; k < 5; ++k) {
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
                        SSIM_TAP(0); SSIM_TAP(1); SSIM_TAP(2); SSIM_TAP(3);
                        SSIM_TAP(4); SSIM_TAP(5); SSIM_TAP(6);
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
                        SSIM_TAP(0); SSIM_TAP(1); SSIM_TAP(2); SSIM_TAP(3);
                        SSIM_TAP(4); SSIM_TAP(5); SSIM_TAP(6); SSIM_TAP(7);
                        SSIM_TAP(8); SSIM_TAP(9); SSIM_TAP(10);
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
                        SSIM_TAP(0); SSIM_TAP(1); SSIM_TAP(2); SSIM_TAP(3);
                        SSIM_TAP(4); SSIM_TAP(5); SSIM_TAP(6); SSIM_TAP(7);
                        SSIM_TAP(8); SSIM_TAP(9); SSIM_TAP(10); SSIM_TAP(11);
                        SSIM_TAP(12); SSIM_TAP(13); SSIM_TAP(14);
                        AscendC::MicroAPI::DataCopy(hBase + dstBase + off, acc, maskCnt);
                    }
                }
            }
        }
        #undef SSIM_TAP
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

        constexpr uint32_t VL = 64;
        __ubuf__ float *hBase = reinterpret_cast<__ubuf__ float *>(hL_.GetPhyAddr());
        __ubuf__ float *ssimScratch = reinterpret_cast<__ubuf__ float *>(tmpL_.GetPhyAddr());

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg maskCnt;
            AscendC::MicroAPI::RegTensor<float> r_mu1, r_mu2, r_s11, r_s22, r_s12;
            AscendC::MicroAPI::RegTensor<float> r_m11, r_m22, r_m12;
            AscendC::MicroAPI::RegTensor<float> r_sig1, r_sig2, r_sig12;
            AscendC::MicroAPI::RegTensor<float> r_numer, r_denom;

            const uint16_t numChunks = static_cast<uint16_t>((outW_ + VL - 1) / VL);
            uint32_t remaining = static_cast<uint32_t>(outW_);

            for (uint16_t c = 0; c < numChunks; ++c) {
                const int32_t off = static_cast<int32_t>(c) * static_cast<int32_t>(VL);
                maskCnt = AscendC::MicroAPI::UpdateMask<uint32_t>(remaining);

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    r_mu1, hBase + 0 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    r_mu2, hBase + 1 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    r_s11, hBase + 2 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    r_s22, hBase + 3 * wAlign_ + off, 1, maskCnt);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    r_s12, hBase + 4 * wAlign_ + off, 1, maskCnt);

                AscendC::MicroAPI::Mul(r_m11, r_mu1, r_mu1, maskCnt);
                AscendC::MicroAPI::Mul(r_m22, r_mu2, r_mu2, maskCnt);
                AscendC::MicroAPI::Mul(r_m12, r_mu1, r_mu2, maskCnt);
                AscendC::MicroAPI::Sub(r_sig1, r_s11, r_m11, maskCnt);
                AscendC::MicroAPI::Sub(r_sig2, r_s22, r_m22, maskCnt);
                AscendC::MicroAPI::Sub(r_sig12, r_s12, r_m12, maskCnt);
                AscendC::MicroAPI::Maxs(r_sig1, r_sig1, ZERO_F, maskCnt);
                AscendC::MicroAPI::Maxs(r_sig2, r_sig2, ZERO_F, maskCnt);

                AscendC::MicroAPI::Muls(r_numer, r_m12, TWO_F, maskCnt);
                AscendC::MicroAPI::Adds(r_numer, r_numer, SSIM_C1, maskCnt);
                AscendC::MicroAPI::Muls(r_sig12, r_sig12, TWO_F, maskCnt);
                AscendC::MicroAPI::Adds(r_sig12, r_sig12, SSIM_C2, maskCnt);
                AscendC::MicroAPI::Mul(r_numer, r_numer, r_sig12, maskCnt);

                AscendC::MicroAPI::Add(r_denom, r_m11, r_m22, maskCnt);
                AscendC::MicroAPI::Adds(r_denom, r_denom, SSIM_C1, maskCnt);
                AscendC::MicroAPI::Add(r_sig1, r_sig1, r_sig2, maskCnt);
                AscendC::MicroAPI::Adds(r_sig1, r_sig1, SSIM_C2, maskCnt);
                AscendC::MicroAPI::Mul(r_denom, r_denom, r_sig1, maskCnt);

                AscendC::MicroAPI::Div(r_numer, r_numer, r_denom, maskCnt);
                AscendC::MicroAPI::DataCopy(ssimScratch + off, r_numer, maskCnt);
            }
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(ssimGM_[imgBase + h * width_ + c0_], tmpL_, rowStoreParams_);
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
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> block1Buf_;
    TBuf<TPosition::VECCALC> block2Buf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> hBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    LocalTensor<float> weightL_;
    LocalTensor<float> block1L_;
    LocalTensor<float> block2L_;
    LocalTensor<float> vL_;
    LocalTensor<float> hL_;
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
    int64_t rotateBase_;
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
