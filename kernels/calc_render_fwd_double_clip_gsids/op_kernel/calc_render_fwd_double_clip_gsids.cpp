/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cfloat>
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t UB_SIZE = 192 * 1024;
constexpr int64_t ELE_NUM = 4096;
constexpr int64_t NUM_FLOATS_PER_BLOCK = 32 / sizeof(float);
constexpr int64_t NUM_IN_FIRST_REPEAT = 64;
constexpr int64_t NUM_FOR_COMPARESCALAR = 64;
constexpr int MAX_TILE_SIZE = 32;
constexpr int NUM_GS_PER_LOOP = 4;
constexpr int NUM_GS_ATTRIBUTES = 10;
constexpr int MIN_DATACOPY_LEN = 16;
constexpr int NUM_STORE_CLIPINDEX = 2;
constexpr uint8_t GS_ID_0 = 0;
constexpr uint8_t GS_ID_1 = 1;
constexpr uint8_t GS_ID_2 = 2;
constexpr uint8_t GS_ID_3 = 3;
constexpr uint8_t ATTR_MEAN_X = 0;
constexpr uint8_t ATTR_MEAN_Y = 1;
constexpr uint8_t ATTR_CONIC_0 = 2;
constexpr uint8_t ATTR_CONIC_1 = 3;
constexpr uint8_t ATTR_CONIC_2 = 4;
constexpr uint8_t ATTR_OPACITY = 5;
constexpr uint8_t ATTR_COLOR_R = 6;
constexpr uint8_t ATTR_COLOR_G = 7;
constexpr uint8_t ATTR_COLOR_B = 8;
constexpr uint8_t ATTR_DEPTH = 9;
constexpr uint8_t BIT_64 = 64;
constexpr uint8_t CLIPIDX_OFFSET_PING = 4;
constexpr uint8_t CLIPIDX_OFFSET_PONG = 8;
constexpr int COMPARE_SCALAR_MIN_BIT = 256;
constexpr uint8_t ALPHACLIPIDX_LEN = 128;
constexpr uint8_t UB_OFFSET_X_X2Y2_TMP_1 = 0;
constexpr uint8_t UB_OFFSET_X_X2Y2_TMP_2 = 1;
constexpr uint8_t UB_OFFSET_X_X2Y2_TMP_3 = 2;
constexpr uint8_t UB_OFFSET_X_X2Y2_TMP_4 = 3;
constexpr uint8_t UB_OFFSET_Y_1 = 4;
constexpr uint8_t UB_OFFSET_Y_2 = 5;
constexpr uint8_t UB_OFFSET_Y_3 = 6;
constexpr uint8_t UB_OFFSET_Y_4 = 7;
constexpr uint8_t UB_OFFSET_GS_ALPHA_1 = 8;
constexpr uint8_t UB_OFFSET_GS_ALPHA_2 = 9;
constexpr uint8_t UB_OFFSET_GS_ALPHA_3 = 10;
constexpr uint8_t UB_OFFSET_GS_ALPHA_4 = 11;
constexpr uint8_t UB_OFFSET_LN_X2_1 = 12;
constexpr uint8_t UB_OFFSET_LN_X2_2 = 13;
constexpr uint8_t UB_OFFSET_LN_X2_3 = 14;
constexpr uint8_t UB_OFFSET_LN_X2_4 = 15;
constexpr uint8_t UB_OFFSET_Y2_1 = 16;
constexpr uint8_t UB_OFFSET_Y2_2 = 17;
constexpr uint8_t UB_OFFSET_Y2_3 = 18;
constexpr uint8_t UB_OFFSET_Y2_4 = 19;
constexpr uint8_t UB_OFFSET_T = 20;
constexpr uint8_t UB_OFFSET_TILECOORDX = 0;
constexpr uint8_t UB_OFFSET_TILECOORDY = 1;
constexpr uint8_t UB_OFFSET_COLOR_R = 2;
constexpr uint8_t UB_OFFSET_COLOR_G = 3;
constexpr uint8_t UB_OFFSET_COLOR_B = 4;
constexpr uint8_t UB_OFFSET_DEPTH = 5;
constexpr uint8_t UB_OFFSET_ERROR = 6;
constexpr uint8_t UB_OFFSET_LN1SUBALPHASUM = 7;

class CalcRenderFwdDoubleClipGsids {
public:
    __aicore__ inline CalcRenderFwdDoubleClipGsids()
    {
    }

    __aicore__ inline void Init(GM_ADDR gs, GM_ADDR tileCoords, GM_ADDR offsets,
                                GM_ADDR gsIds, int64_t nPixel, int64_t tileNum,
                                int64_t nGauss, GM_ADDR color, GM_ADDR depth,
                                GM_ADDR lastCumsum, GM_ADDR error, GM_ADDR gsClipIndex,
                                GM_ADDR alphaClipIndex)
    {
        vecIdx_ = GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * GetSubBlockNum();

        pingId_ = EVENT_ID6;
        pongId_ = EVENT_ID7;

        nPixel_ = nPixel;
        nPixel_1d_ = sqrt(nPixel_);
        if (nPixel_1d_ == 0) {
            nPixel_1d_ = 1;
        }
        if (nPixel_ > MAX_TILE_SIZE * MAX_TILE_SIZE) {
            calPixel_ = MAX_TILE_SIZE * MAX_TILE_SIZE;
        } else {
            calPixel_ = nPixel_;
        }
        tileNum_ = tileNum;

        gsGm_.SetGlobalBuffer((__gm__ float *)gs);
        gsIdsGm_.SetGlobalBuffer((__gm__ int64_t *)gsIds);
        gsClipIndexGm_.SetGlobalBuffer((__gm__ int64_t *)gsClipIndex);
        alphaClipIndexGm_.SetGlobalBuffer((__gm__ uint8_t *)alphaClipIndex);

        tileCoordsGm_.SetGlobalBuffer((__gm__ float *)tileCoords);

        colorRGm_.SetGlobalBuffer((__gm__ float *)color, tileNum_ * nPixel_);
        colorGGm_ = colorRGm_[tileNum_ * nPixel_];
        colorBGm_ = colorGGm_[tileNum_ * nPixel_];
        depthGm_.SetGlobalBuffer((__gm__ float *)depth, tileNum_ * nPixel_);
        lastCumsumGm_.SetGlobalBuffer((__gm__ float *)lastCumsum);
        errorGm_.SetGlobalBuffer((__gm__ float *)error);

        coreOffsetsGm_.SetGlobalBuffer((__gm__ int64_t *)offsets);
        scheduleGm_ = coreOffsetsGm_[vecNum_];
        tileOffsetsGm_= scheduleGm_[tileNum_];

        // shared ub space
        UbBuffInit();

        ping = true;
        reduceNum_ = calPixel_;
        repeatNum_ = reduceNum_ / NUM_IN_FIRST_REPEAT;
        oneRepeatNum_ = NUM_IN_FIRST_REPEAT;
        srcRepStride_ = oneRepeatNum_ / NUM_FLOATS_PER_BLOCK;
        lineStride_ = oneRepeatNum_ / nPixel_1d_;
    }

    __aicore__ inline void UbBuffInit()
    {
        // allocate ub space
        TPipe pipe;
        TBuf<QuePosition::VECCALC> ubBuf;
        pipe.InitBuffer(ubBuf, UB_SIZE);
        LocalTensor<float> ub = ubBuf.Get<float>();

        SharedBuffInit(ub, 0);

        uint32_t ping_offset = calPixel_ * UB_OFFSET_T + calPixel_;
        PingBuffInit(ub, ping_offset);

        uint32_t pong_offset = calPixel_ + calPixel_ * UB_OFFSET_LN1SUBALPHASUM + ping_offset;
        PongBuffInit(ub, pong_offset);

        uint32_t gsAttr_offset = pong_offset + calPixel_ + calPixel_ * UB_OFFSET_LN1SUBALPHASUM;

        uint32_t ub_offset = 0;
        gsAttr1Ping_ = ub[gsAttr_offset];

        ub_offset++;
        gsAttr2Ping_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        ub_offset++;
        gsAttr3Ping_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        ub_offset++;
        gsAttr4Ping_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        ub_offset++;
        gsAttr1Pong_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        ub_offset++;
        gsAttr2Pong_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        ub_offset++;
        gsAttr3Pong_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        ub_offset++;
        gsAttr4Pong_ = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset];

        nClip_ = calPixel_ / BIT_64;
        ub_offset++;
        LocalTensor<int64_t> ubInt64 = ub[gsAttr_offset + MIN_DATACOPY_LEN * ub_offset].ReinterpretCast<int64_t>();
        gsClipIndexPing_ = ubInt64[0];
        gsClipIndexPong_ = ubInt64[CLIPIDX_OFFSET_PING];
        LocalTensor<uint8_t> ubInt = ubInt64[CLIPIDX_OFFSET_PONG].ReinterpretCast<uint8_t>();
        alphaClipIndexUB_ = ubInt[0];
        alphaReduceMaxUB_u16 = ubInt[calPixel_ * NUM_STORE_CLIPINDEX].ReinterpretCast<uint16_t>();
        clipIs_ = ubInt[calPixel_ * NUM_STORE_CLIPINDEX + COMPARE_SCALAR_MIN_BIT + ALPHACLIPIDX_LEN];
    }

    __aicore__ inline void SharedBuffInit(LocalTensor<float>& ub, uint32_t buff_offset)
    {
        x_1 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_1 * calPixel_];
        x2y2_1 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_1 * calPixel_];
        tmpRes_1 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_1 * calPixel_];

        x_2 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_2 * calPixel_];
        x2y2_2 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_2 * calPixel_];
        tmpRes_2 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_2 * calPixel_];

        x_3 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_3 * calPixel_];
        x2y2_3 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_3 * calPixel_];
        tmpRes_3 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_3 * calPixel_];

        x_4 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_4 * calPixel_];
        x2y2_4 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_4 * calPixel_];
        tmpRes_4 = ub[buff_offset + UB_OFFSET_X_X2Y2_TMP_4 * calPixel_];

        y_1 = ub[buff_offset + UB_OFFSET_Y_1 * calPixel_];
        y_2 = ub[buff_offset + UB_OFFSET_Y_2 * calPixel_];
        y_3 = ub[buff_offset + UB_OFFSET_Y_3 * calPixel_];
        y_4 = ub[buff_offset + UB_OFFSET_Y_4 * calPixel_];

        gaussWeight_1 = ub[buff_offset + UB_OFFSET_GS_ALPHA_1 * calPixel_];
        alpha_1 = ub[buff_offset + UB_OFFSET_GS_ALPHA_1 * calPixel_];
        alphaT_ = ub[buff_offset + UB_OFFSET_GS_ALPHA_1 * calPixel_];

        gaussWeight_2 = ub[buff_offset + UB_OFFSET_GS_ALPHA_2 * calPixel_];
        alpha_2 = ub[buff_offset + UB_OFFSET_GS_ALPHA_2 * calPixel_];

        gaussWeight_3 = ub[buff_offset + UB_OFFSET_GS_ALPHA_3 * calPixel_];
        alpha_3 = ub[buff_offset + UB_OFFSET_GS_ALPHA_3 * calPixel_];

        gaussWeight_4 = ub[buff_offset + UB_OFFSET_GS_ALPHA_4 * calPixel_];
        alpha_4 = ub[buff_offset + UB_OFFSET_GS_ALPHA_4 * calPixel_];
        
        ln1SubAlpha_1 = ub[buff_offset + UB_OFFSET_LN_X2_1 * calPixel_];
        x2_1 = ub[buff_offset + UB_OFFSET_LN_X2_1 * calPixel_];

        ln1SubAlpha_2 = ub[buff_offset + UB_OFFSET_LN_X2_2 * calPixel_];
        x2_2 = ub[buff_offset + UB_OFFSET_LN_X2_2 * calPixel_];

        ln1SubAlpha_3 = ub[buff_offset + UB_OFFSET_LN_X2_3 * calPixel_];
        x2_3 = ub[buff_offset + UB_OFFSET_LN_X2_3 * calPixel_];

        ln1SubAlpha_4 = ub[buff_offset + UB_OFFSET_LN_X2_4 * calPixel_];
        x2_4 = ub[buff_offset + UB_OFFSET_LN_X2_4 * calPixel_];

        y2_1 = ub[buff_offset + UB_OFFSET_Y2_1 * calPixel_];
        y2_2 = ub[buff_offset + UB_OFFSET_Y2_2 * calPixel_];
        y2_3 = ub[buff_offset + UB_OFFSET_Y2_3 * calPixel_];
        y2_4 = ub[buff_offset + UB_OFFSET_Y2_4 * calPixel_];

        T_ = ub[buff_offset + UB_OFFSET_T * calPixel_];
    }

    __aicore__ inline void PingBuffInit(LocalTensor<float>& ub, uint32_t ping_offset)
    {
        tileCoordXPing_ = ub[ping_offset + UB_OFFSET_TILECOORDX * calPixel_];
        tileCoordYPing_ = ub[ping_offset + UB_OFFSET_TILECOORDY * calPixel_];
        colorRPing_ = ub[ping_offset + UB_OFFSET_COLOR_R * calPixel_];
        colorGPing_ = ub[ping_offset + UB_OFFSET_COLOR_G * calPixel_];
        colorBPing_ = ub[ping_offset + UB_OFFSET_COLOR_B * calPixel_];
        depthPing_ = ub[ping_offset + UB_OFFSET_DEPTH * calPixel_];
        errorPing_ = ub[ping_offset + UB_OFFSET_ERROR * calPixel_];
        ln1SubAlphaSumPing_ = ub[ping_offset + UB_OFFSET_LN1SUBALPHASUM * calPixel_];
    }

    __aicore__ inline void PongBuffInit(LocalTensor<float>& ub, uint32_t pong_offset)
    {
        tileCoordXPong_ = ub[pong_offset + UB_OFFSET_TILECOORDX * calPixel_];
        tileCoordYPong_ = ub[pong_offset + UB_OFFSET_TILECOORDY * calPixel_];
        colorRPong_ = ub[pong_offset + UB_OFFSET_COLOR_R * calPixel_];
        colorGPong_ = ub[pong_offset + UB_OFFSET_COLOR_G * calPixel_];
        colorBPong_ = ub[pong_offset + UB_OFFSET_COLOR_B * calPixel_];
        depthPong_ = ub[pong_offset + UB_OFFSET_DEPTH * calPixel_];
        errorPong_ = ub[pong_offset + UB_OFFSET_ERROR * calPixel_];
        ln1SubAlphaSumPong_ = ub[pong_offset + UB_OFFSET_LN1SUBALPHASUM * calPixel_];
    }

    __aicore__ inline void Process(int64_t tileIdx)
    {
        event_t flagId = ping ? pingId_ : pongId_;

        LocalTensor<float> ln1SubAlphaSum;
        LocalTensor<float> tileCoordX;
        LocalTensor<float> tileCoordY;

        LocalTensor<float> colorR;
        LocalTensor<float> colorG;
        LocalTensor<float> colorB;
        LocalTensor<float> depth;

        LocalTensor<float> error;
        LocalTensor<float> gsAttr1;
        LocalTensor<float> gsAttr2;
        LocalTensor<float> gsAttr3;
        LocalTensor<float> gsAttr4;
        LocalTensor<int64_t> gsClipIndexUb;

        bool gsPing = ping;
        event_t gs_flagId = flagId;

        int64_t prevOffset = 0;
        if (tileIdx > 0) {
            prevOffset = (int64_t)tileOffsetsGm_.GetValue(tileIdx - 1);
        }
        int64_t currOffset = (int64_t)tileOffsetsGm_.GetValue(tileIdx);

        GlobalTensor<float> tileCoordsGm = tileCoordsGm_[tileIdx * 2 * nPixel_];
        GlobalTensor<float> colorRGm = colorRGm_[tileIdx * nPixel_];
        GlobalTensor<float> colorGGm = colorGGm_[tileIdx * nPixel_];
        GlobalTensor<float> colorBGm = colorBGm_[tileIdx * nPixel_];
        GlobalTensor<float> depthGm = depthGm_[tileIdx * nPixel_];
        GlobalTensor<float> lastCumsumGm = lastCumsumGm_[tileIdx * nPixel_];
        GlobalTensor<float> errorGm = errorGm_[tileIdx * nPixel_];
        
        int64_t gsClipIndex = currOffset;
        SetFlag<HardEvent::MTE3_MTE2>(pingId_);
        SetFlag<HardEvent::MTE3_V>(pingId_);
        SetFlag<HardEvent::MTE3_MTE2>(pongId_);
        SetFlag<HardEvent::MTE3_V>(pongId_);
        for (int64_t j = 0; j < (nPixel_ + calPixel_ - 1) / calPixel_; j++) {
            if (ping) {
                ln1SubAlphaSum = ln1SubAlphaSumPing_;
                tileCoordX = tileCoordXPing_;
                tileCoordY = tileCoordYPing_;

                colorR = colorRPing_;
                colorG = colorGPing_;
                colorB = colorBPing_;
                depth = depthPing_;

                error = errorPing_;
                gsAttr1 = gsAttr1Ping_;
                gsAttr2 = gsAttr2Ping_;
                gsAttr3 = gsAttr3Ping_;
                gsAttr4 = gsAttr4Ping_;
                gsClipIndexUb = gsClipIndexPing_;
                flagId = pingId_;
            } else {
                ln1SubAlphaSum = ln1SubAlphaSumPong_;
                tileCoordX = tileCoordXPong_;
                tileCoordY = tileCoordYPong_;

                colorR = colorRPong_;
                colorG = colorGPong_;
                colorB = colorBPong_;
                depth = depthPong_;

                error = errorPong_;
                gsAttr1 = gsAttr1Pong_;
                gsAttr2 = gsAttr2Pong_;
                gsAttr3 = gsAttr3Pong_;
                gsAttr4 = gsAttr4Pong_;
                gsClipIndexUb = gsClipIndexPong_;
                flagId = pongId_;
            }

            WaitFlag<HardEvent::MTE3_MTE2>(flagId);

            DataCopy(tileCoordX, tileCoordsGm[j * calPixel_], calPixel_);
            DataCopy(tileCoordY, tileCoordsGm[nPixel_ + j * calPixel_], calPixel_);

            SetFlag<HardEvent::MTE2_V>(flagId);

            WaitFlag<HardEvent::MTE3_V>(flagId);
            // initialize to 0.0f
            Duplicate(ln1SubAlpha_1, 0.0f, calPixel_);
            Duplicate(ln1SubAlpha_2, 0.0f, calPixel_);
            Duplicate(ln1SubAlpha_3, 0.0f, calPixel_);
            Duplicate(ln1SubAlpha_4, 0.0f, calPixel_);
            Duplicate(ln1SubAlphaSum, 0.0f, calPixel_);
            Duplicate(colorR, 0.0f, calPixel_);
            Duplicate(colorG, 0.0f, calPixel_);
            Duplicate(colorB, 0.0f, calPixel_);
            Duplicate(depth, 0.0f, calPixel_);
            Duplicate(error, 0.0f, calPixel_);
            WaitFlag<HardEvent::MTE2_V>(flagId);
            // PipeBarrier<PIPE_V>();
            int16_t sta_pix1 = 0;
            int16_t end_pix1 = nPixel_1d_;
            int32_t calPixel_1 = calPixel_;
            int16_t sta_pix2 = 0;
            int16_t end_pix2 = nPixel_1d_;
            int32_t calPixel_2 = calPixel_;
            int16_t sta_pix3 = 0;
            int16_t end_pix3 = nPixel_1d_;
            int32_t calPixel_3 = calPixel_;
            int16_t sta_pix4 = 0;
            int16_t end_pix4 = nPixel_1d_;
            int32_t calPixel_4 = calPixel_;
            uint16_t alpha_wb_acc = 0;
            int64_t sta_idx = prevOffset;
            int64_t i = prevOffset;
            bool clip_break = false;
            for (; i <= currOffset - NUM_GS_PER_LOOP; i+= NUM_GS_PER_LOOP) {
                if (gsPing) {
                    DataCopy(gsAttr1Ping_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_0) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    DataCopy(gsAttr2Ping_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_1) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    DataCopy(gsAttr3Ping_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_2) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    DataCopy(gsAttr4Ping_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_3) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    gs_flagId = pingId_;
                } else {
                    DataCopy(gsAttr1Pong_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_0) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    DataCopy(gsAttr2Pong_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_1) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    DataCopy(gsAttr3Pong_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_2) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    DataCopy(gsAttr4Pong_, gsGm_[gsIdsGm_.GetValue(i + GS_ID_3) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    gs_flagId = pongId_;
                }
                // update ln1SubAlphaSum_
                if (alpha_wb_acc >= NUM_STORE_CLIPINDEX * calPixel_) {
                    SetFlag<HardEvent::S_MTE3>(gs_flagId);
                    WaitFlag<HardEvent::S_MTE3>(gs_flagId);

                    DataCopy(alphaClipIndexGm_[NUM_STORE_CLIPINDEX * sta_idx], alphaClipIndexUB_, alpha_wb_acc);

                    SetFlag<HardEvent::MTE3_MTE2>(gs_flagId);
                    SetFlag<HardEvent::MTE3_V>(gs_flagId);
                    WaitFlag<HardEvent::MTE3_MTE2>(gs_flagId);
                    WaitFlag<HardEvent::MTE3_V>(gs_flagId);
                    alpha_wb_acc = 0;
                    sta_idx = i;
                }

                SetFlag<HardEvent::MTE2_S>(gs_flagId);
                SetFlag<HardEvent::MTE2_V>(gs_flagId);
                // clip
                if (i != prevOffset) {
                    // clipIs_[T_ < 0.01] = 1
                    AscendC::CompareScalar(clipIs_, T_, 0.01f, AscendC::CMPMODE::LT, calPixel_);
                    SetFlag<HardEvent::V_S>(flagId);
                    WaitFlag<HardEvent::V_S>(flagId);
                    //  reinterpret into 64-bit integer for clip check
                    LocalTensor<uint64_t> clipIs64 = clipIs_.ReinterpretCast<uint64_t>();
                    bool clip = true;
                    for (int64_t i_clip = 0; i_clip < nClip_; ++i_clip) {
                        // if any bit is 0, then not all pixels are clipped
                        clip = (clipIs64.GetValue(i_clip) == UINT64_MAX);
                        if (clip == false)
                            break;
                    }
                    if (clip == true) {
                        gsClipIndex = i;
                        clip_break = true;
                        WaitFlag<HardEvent::MTE2_S>(gs_flagId);
                        WaitFlag<HardEvent::MTE2_V>(gs_flagId);
                        gsPing = !gsPing;
                        break;
                    }
                }
                if (i == prevOffset) {
                    Exp(T_, ln1SubAlphaSum, calPixel_);
                }

                // compute prob density of pixels
                WaitFlag<HardEvent::MTE2_S>(gs_flagId);
                WaitFlag<HardEvent::MTE2_V>(gs_flagId);
                if (gsPing) {
                    gsAttr1 = gsAttr1Ping_;
                    gsAttr2 = gsAttr2Ping_;
                    gsAttr3 = gsAttr3Ping_;
                    gsAttr4 = gsAttr4Ping_;
                } else {
                    gsAttr1 = gsAttr1Pong_;
                    gsAttr2 = gsAttr2Pong_;
                    gsAttr3 = gsAttr3Pong_;
                    gsAttr4 = gsAttr4Pong_;
                }
                Adds(x_1, tileCoordX, -gsAttr1.GetValue(ATTR_MEAN_X), calPixel_); // meanX
                Adds(x_2, tileCoordX, -gsAttr2.GetValue(ATTR_MEAN_X), calPixel_); // meanX
                Adds(x_3, tileCoordX, -gsAttr3.GetValue(ATTR_MEAN_X), calPixel_); // meanX
                Adds(x_4, tileCoordX, -gsAttr4.GetValue(ATTR_MEAN_X), calPixel_); // meanX

                Adds(y_1, tileCoordY, -gsAttr1.GetValue(ATTR_MEAN_Y), calPixel_); // meanY
                Adds(y_2, tileCoordY, -gsAttr2.GetValue(ATTR_MEAN_Y), calPixel_); // meanY
                Adds(y_3, tileCoordY, -gsAttr3.GetValue(ATTR_MEAN_Y), calPixel_); // meanY
                Adds(y_4, tileCoordY, -gsAttr4.GetValue(ATTR_MEAN_Y), calPixel_); // meanY

                // PipeBarrier<PIPE_V>();
                Mul(gaussWeight_1, x_1, y_1, calPixel_);
                Mul(gaussWeight_2, x_2, y_2, calPixel_);
                Mul(gaussWeight_3, x_3, y_3, calPixel_);
                Mul(gaussWeight_4, x_4, y_4, calPixel_);

                Mul(x2_1, x_1, x_1, calPixel_);
                Mul(x2_2, x_2, x_2, calPixel_);
                Mul(x2_3, x_3, x_3, calPixel_);
                Mul(x2_4, x_4, x_4, calPixel_);

                Mul(y2_1, y_1, y_1, calPixel_);
                Mul(y2_2, y_2, y_2, calPixel_);
                Mul(y2_3, y_3, y_3, calPixel_);
                Mul(y2_4, y_4, y_4, calPixel_);

                // PipeBarrier<PIPE_V>();
                Muls(gaussWeight_1, gaussWeight_1, -gsAttr1.GetValue(ATTR_CONIC_1), calPixel_); // conic1
                Muls(gaussWeight_2, gaussWeight_2, -gsAttr2.GetValue(ATTR_CONIC_1), calPixel_); // conic1
                Muls(gaussWeight_3, gaussWeight_3, -gsAttr3.GetValue(ATTR_CONIC_1), calPixel_); // conic1
                Muls(gaussWeight_4, gaussWeight_4, -gsAttr4.GetValue(ATTR_CONIC_1), calPixel_); // conic1

                Muls(x2_1, x2_1, -gsAttr1.GetValue(ATTR_CONIC_0), calPixel_); // conic0
                Muls(x2_2, x2_2, -gsAttr2.GetValue(ATTR_CONIC_0), calPixel_); // conic0
                Muls(x2_3, x2_3, -gsAttr3.GetValue(ATTR_CONIC_0), calPixel_); // conic0
                Muls(x2_4, x2_4, -gsAttr4.GetValue(ATTR_CONIC_0), calPixel_); // conic0

                Muls(y2_1, y2_1, -gsAttr1.GetValue(ATTR_CONIC_2), calPixel_); // conic2
                Muls(y2_2, y2_2, -gsAttr2.GetValue(ATTR_CONIC_2), calPixel_); // conic2
                Muls(y2_3, y2_3, -gsAttr3.GetValue(ATTR_CONIC_2), calPixel_); // conic2
                Muls(y2_4, y2_4, -gsAttr4.GetValue(ATTR_CONIC_2), calPixel_); // conic2

                // PipeBarrier<PIPE_V>();
                Add(x2y2_1, x2_1, y2_1, calPixel_);
                Add(x2y2_2, x2_2, y2_2, calPixel_);
                Add(x2y2_3, x2_3, y2_3, calPixel_);
                Add(x2y2_4, x2_4, y2_4, calPixel_);

                // PipeBarrier<PIPE_V>();
                Axpy(gaussWeight_1, x2y2_1, 0.5f, calPixel_);
                Axpy(gaussWeight_2, x2y2_2, 0.5f, calPixel_);
                Axpy(gaussWeight_3, x2y2_3, 0.5f, calPixel_);
                Axpy(gaussWeight_4, x2y2_4, 0.5f, calPixel_);

                // PipeBarrier<PIPE_V>();
                // gaussWeight_ = exp(-1/2 (c_0(x-miu_x)^2 + 2c_1(x-miu_x)(y-miu_y) + c_2(y-miu_y)^2))
                Exp(gaussWeight_1, gaussWeight_1, calPixel_);
                Exp(gaussWeight_2, gaussWeight_2, calPixel_);
                Exp(gaussWeight_3, gaussWeight_3, calPixel_);
                Exp(gaussWeight_4, gaussWeight_4, calPixel_);

                // PipeBarrier<PIPE_V>();
                // compute alpha, openness
                // opacities,  alpha = o e^(gaussWeight_)
                Muls(alpha_1, gaussWeight_1, gsAttr1.GetValue(ATTR_OPACITY), calPixel_);
                Muls(alpha_2, gaussWeight_2, gsAttr2.GetValue(ATTR_OPACITY), calPixel_);
                Muls(alpha_3, gaussWeight_3, gsAttr3.GetValue(ATTR_OPACITY), calPixel_);
                Muls(alpha_4, gaussWeight_4, gsAttr4.GetValue(ATTR_OPACITY), calPixel_);

                // PipeBarrier<PIPE_V>();
                // alpha_ = min(alpha_, 0.999)
                Mins(alpha_1, alpha_1, 0.999f, calPixel_);
                Mins(alpha_2, alpha_2, 0.999f, calPixel_);
                Mins(alpha_3, alpha_3, 0.999f, calPixel_);
                Mins(alpha_4, alpha_4, 0.999f, calPixel_);
                // PipeBarrier<PIPE_V>();

                WholeReduceMax(tmpRes_1, alpha_1, oneRepeatNum_, repeatNum_, 1, 1,
                    srcRepStride_, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                WholeReduceMax(tmpRes_2, alpha_2, oneRepeatNum_, repeatNum_, 1, 1,
                    srcRepStride_, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                WholeReduceMax(tmpRes_3, alpha_3, oneRepeatNum_, repeatNum_, 1, 1,
                    srcRepStride_, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                WholeReduceMax(tmpRes_4, alpha_4, oneRepeatNum_, repeatNum_, 1, 1,
                    srcRepStride_, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                // PipeBarrier<PIPE_V>();

                CompareScalar(alphaReduceMaxUB_u16,
                    tmpRes_1, 0.01f, AscendC::CMPMODE::GE, NUM_FOR_COMPARESCALAR);
                CompareScalar(alphaReduceMaxUB_u16[GS_ID_1 * MIN_DATACOPY_LEN],
                    tmpRes_2, 0.01f, AscendC::CMPMODE::GE, NUM_FOR_COMPARESCALAR);
                CompareScalar(alphaReduceMaxUB_u16[GS_ID_2 * MIN_DATACOPY_LEN],
                    tmpRes_3, 0.01f, AscendC::CMPMODE::GE, NUM_FOR_COMPARESCALAR);
                CompareScalar(alphaReduceMaxUB_u16[GS_ID_3 * MIN_DATACOPY_LEN],
                    tmpRes_4, 0.01f, AscendC::CMPMODE::GE, NUM_FOR_COMPARESCALAR);
                // PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_S>(gs_flagId);
                WaitFlag<HardEvent::V_S>(gs_flagId);
                auto alphaMax = alphaReduceMaxUB_u16.GetValue(0);
                end_pix1 = (BIT_64 - ScalarCountLeadingZero(alphaMax)) *  lineStride_;
                calPixel_1 = ScalarGetCountOfValue<1>(alphaMax) *  lineStride_;
                sta_pix1 = end_pix1 - calPixel_1;

                alpha_wb_acc+=NUM_STORE_CLIPINDEX;
                if (calPixel_1 == 0) {
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, nPixel_1d_);
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, 0);
                    sta_pix1 = 0;
                    // continue;
                }
                alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, (uint8_t)sta_pix1);
                alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, (uint8_t)end_pix1);
                
                calPixel_1 = calPixel_1 * nPixel_1d_;
                sta_pix1 = sta_pix1 * nPixel_1d_;

                SetFlag<HardEvent::S_V>(gs_flagId);
                WaitFlag<HardEvent::S_V>(gs_flagId);

                // compute transmittances
                // ln1SubAlpha_ = ln(1 - alpha_)
                Muls(ln1SubAlpha_1[sta_pix1], alpha_1[sta_pix1], -1.0f, calPixel_1);

                // do alpha blending
                Mul(alphaT_[sta_pix1], T_[sta_pix1], alpha_1[sta_pix1], calPixel_1);

                auto alphaMax2 = alphaReduceMaxUB_u16.GetValue(GS_ID_1 * MIN_DATACOPY_LEN);
                end_pix2 = (BIT_64 - ScalarCountLeadingZero(alphaMax2)) *  lineStride_;
                calPixel_2 = ScalarGetCountOfValue<1>(alphaMax2) *  lineStride_;
                sta_pix2 = end_pix2 - calPixel_2;

                alpha_wb_acc+=NUM_STORE_CLIPINDEX;
                if (calPixel_2 == 0) {
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, nPixel_1d_);
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, 0);
                    sta_pix2 = 0;
                    // continue;
                }
                alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, (uint8_t)sta_pix2);
                alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, (uint8_t)end_pix2);
                
                calPixel_2 = calPixel_2 * nPixel_1d_;
                sta_pix2 = sta_pix2 * nPixel_1d_;

                auto alphaMax3 = alphaReduceMaxUB_u16.GetValue(GS_ID_2 * MIN_DATACOPY_LEN);
                end_pix3 = (BIT_64 - ScalarCountLeadingZero(alphaMax3)) *  lineStride_;
                calPixel_3 = ScalarGetCountOfValue<1>(alphaMax3) *  lineStride_;
                sta_pix3 = end_pix3 - calPixel_3;

                alpha_wb_acc+=NUM_STORE_CLIPINDEX;
                if (calPixel_3 == 0) {
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, nPixel_1d_);
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, 0);
                    sta_pix3 = 0;
                    // continue;
                }
                alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, (uint8_t)sta_pix3);
                alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, (uint8_t)end_pix3);
                
                calPixel_3 = calPixel_3 * nPixel_1d_;
                sta_pix3 = sta_pix3 * nPixel_1d_;

                auto alphaMax4 = alphaReduceMaxUB_u16.GetValue(GS_ID_3 * MIN_DATACOPY_LEN);
                end_pix4 = (BIT_64 - ScalarCountLeadingZero(alphaMax4)) *  lineStride_;
                calPixel_4 = ScalarGetCountOfValue<1>(alphaMax4) *  lineStride_;
                sta_pix4 = end_pix4 - calPixel_4;

                alpha_wb_acc += NUM_STORE_CLIPINDEX;
                if (calPixel_4 == 0) {
                    sta_pix4 = 0;
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, nPixel_1d_);
                    alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, 0);
                    // continue;
                }
                alphaClipIndexUB_.SetValue(alpha_wb_acc - NUM_STORE_CLIPINDEX, (uint8_t)sta_pix4);
                alphaClipIndexUB_.SetValue(alpha_wb_acc - 1, (uint8_t)end_pix4);
                
                calPixel_4 = calPixel_4 * nPixel_1d_;
                sta_pix4 = sta_pix4 * nPixel_1d_;

                SetFlag<HardEvent::S_V>(gs_flagId);

                // PipeBarrier<PIPE_V>();
                Axpy(colorR[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_COLOR_R), calPixel_1); // colorR
                Axpy(colorG[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_COLOR_G), calPixel_1); // colorG
                Axpy(colorB[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_COLOR_B), calPixel_1); // colorB
                Axpy(depth[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_DEPTH), calPixel_1);  // depth

                Adds(ln1SubAlpha_1[sta_pix1], ln1SubAlpha_1[sta_pix1], 1.0f, calPixel_1);

                // PipeBarrier<PIPE_V>();
                Ln(ln1SubAlpha_1[sta_pix1], ln1SubAlpha_1[sta_pix1], calPixel_1);
                // do alpha blending
                // PipeBarrier<PIPE_V>();
                // int64_t gs_index = gsIdsGm_.GetValue(i) * 10; // index*gs属性长度
                // 误差计算
                // ln1SubAlpha_ = ln1SubAlpha_ - error
                Sub(ln1SubAlpha_1[sta_pix1], ln1SubAlpha_1[sta_pix1], error[sta_pix1], calPixel_1);
                // PipeBarrier<PIPE_V>();
                WaitFlag<HardEvent::S_V>(gs_flagId);
                // tmp = ln1SubAlphaSum + ln1SubAlpha_
                Add(tmpRes_1[sta_pix1], ln1SubAlphaSum[sta_pix1], ln1SubAlpha_1[sta_pix1], calPixel_1);
                Muls(ln1SubAlpha_2[sta_pix2], alpha_2[sta_pix2], -1.0f, calPixel_2);
                Muls(ln1SubAlpha_3[sta_pix3], alpha_3[sta_pix3], -1.0f, calPixel_3);
                Muls(ln1SubAlpha_4[sta_pix4], alpha_4[sta_pix4], -1.0f, calPixel_4);
                // PipeBarrier<PIPE_V>();
                // error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum
                Sub(error[sta_pix1], tmpRes_1[sta_pix1], ln1SubAlphaSum[sta_pix1], calPixel_1);
                Adds(ln1SubAlpha_2[sta_pix2], ln1SubAlpha_2[sta_pix2], 1.0f, calPixel_2);
                Adds(ln1SubAlpha_3[sta_pix3], ln1SubAlpha_3[sta_pix3], 1.0f, calPixel_3);
                Adds(ln1SubAlpha_4[sta_pix4], ln1SubAlpha_4[sta_pix4], 1.0f, calPixel_4);

                // PipeBarrier<PIPE_V>();
                // error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum - ln1SubAlpha_
                Sub(error[sta_pix1], error[sta_pix1], ln1SubAlpha_1[sta_pix1], calPixel_1);
                if (calPixel_1 != 0) {
                    // ln1SubAlphaSum += ln1SubAlpha_
                    DataCopy(ln1SubAlphaSum[sta_pix1], tmpRes_1[sta_pix1], calPixel_1);
                }
                Ln(ln1SubAlpha_2[sta_pix2], ln1SubAlpha_2[sta_pix2], calPixel_2);
                Ln(ln1SubAlpha_3[sta_pix3], ln1SubAlpha_3[sta_pix3], calPixel_3);
                Ln(ln1SubAlpha_4[sta_pix4], ln1SubAlpha_4[sta_pix4], calPixel_4);

                // PipeBarrier<PIPE_V>();
                // ln1SubAlpha_ = ln1SubAlpha_ - error
                Sub(ln1SubAlpha_2[sta_pix2], ln1SubAlpha_2[sta_pix2], error[sta_pix2], calPixel_2);
                // compute transmittances
                // T_ = exp(ln1SubAlphaSum) gs 2
                Exp(T_, ln1SubAlphaSum, calPixel_);
                // PipeBarrier<PIPE_V>();

                // tmp = ln1SubAlphaSum + ln1SubAlpha_
                Add(tmpRes_2[sta_pix2], ln1SubAlphaSum[sta_pix2], ln1SubAlpha_2[sta_pix2], calPixel_2);
                Mul(alphaT_[sta_pix2], T_[sta_pix2], alpha_2[sta_pix2], calPixel_2); // alphaT_ = T_ * alpha_

                // PipeBarrier<PIPE_V>();
                Sub(error[sta_pix2], tmpRes_2[sta_pix2], ln1SubAlphaSum[sta_pix2], calPixel_2);  // Gs_2
                Axpy(colorR[sta_pix2], alphaT_[sta_pix2], gsAttr2.GetValue(ATTR_COLOR_R), calPixel_2); // colorR
                Axpy(colorG[sta_pix2], alphaT_[sta_pix2], gsAttr2.GetValue(ATTR_COLOR_G), calPixel_2); // colorG
                Axpy(colorB[sta_pix2], alphaT_[sta_pix2], gsAttr2.GetValue(ATTR_COLOR_B), calPixel_2); // colorB
                Axpy(depth[sta_pix2], alphaT_[sta_pix2], gsAttr2.GetValue(ATTR_DEPTH), calPixel_2);  // depth
                
                // PipeBarrier<PIPE_V>();
                if (calPixel_2 != 0) {
                    //  ln1SubAlphaSum += ln1SubAlpha_
                    DataCopy(ln1SubAlphaSum[sta_pix2], tmpRes_2[sta_pix2], calPixel_2);
                }

                //  error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum - ln1SubAlpha_
                Sub(error[sta_pix2], error[sta_pix2], ln1SubAlpha_2[sta_pix2], calPixel_2);
                // PipeBarrier<PIPE_V>();

                 // ln1SubAlpha_ = ln1SubAlpha_ - error
                Sub(ln1SubAlpha_3[sta_pix3], ln1SubAlpha_3[sta_pix3], error[sta_pix3], calPixel_3);
                Exp(T_, ln1SubAlphaSum, calPixel_); // T_ = exp(ln1SubAlphaSum)
                // PipeBarrier<PIPE_V>();
                // tmp = ln1SubAlphaSum + ln1SubAlpha_
                Add(tmpRes_3[sta_pix3], ln1SubAlphaSum[sta_pix3], ln1SubAlpha_3[sta_pix3], calPixel_3);
                Mul(alphaT_[sta_pix3], T_[sta_pix3], alpha_3[sta_pix3], calPixel_3); // alphaT_ = T_ * alpha_

                // PipeBarrier<PIPE_V>();
                Sub(error[sta_pix3], tmpRes_3[sta_pix3], ln1SubAlphaSum[sta_pix3], calPixel_3);  // Gs_3
                Axpy(colorR[sta_pix3], alphaT_[sta_pix3], gsAttr3.GetValue(ATTR_COLOR_R), calPixel_3); // colorR
                Axpy(colorG[sta_pix3], alphaT_[sta_pix3], gsAttr3.GetValue(ATTR_COLOR_G), calPixel_3); // colorG
                Axpy(colorB[sta_pix3], alphaT_[sta_pix3], gsAttr3.GetValue(ATTR_COLOR_B), calPixel_3); // colorB
                Axpy(depth[sta_pix3], alphaT_[sta_pix3], gsAttr3.GetValue(ATTR_DEPTH), calPixel_3);  // depth
                // PipeBarrier<PIPE_V>();
                if (calPixel_3 != 0) {
                    //  ln1SubAlphaSum += ln1SubAlpha_
                    DataCopy(ln1SubAlphaSum[sta_pix3], tmpRes_3[sta_pix3], calPixel_3);
                }
                
                // error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum - ln1SubAlpha_
                Sub(error[sta_pix3], error[sta_pix3], ln1SubAlpha_3[sta_pix3], calPixel_3);
                // PipeBarrier<PIPE_V>();

                // ln1SubAlpha_ = ln1SubAlpha_ - error
                Sub(ln1SubAlpha_4[sta_pix4], ln1SubAlpha_4[sta_pix4], error[sta_pix4], calPixel_4);
                Exp(T_, ln1SubAlphaSum, calPixel_); // T_ = exp(ln1SubAlphaSum)
                // PipeBarrier<PIPE_V>();

                // tmp = ln1SubAlphaSum + ln1SubAlpha_
                Add(tmpRes_4[sta_pix4], ln1SubAlphaSum[sta_pix4], ln1SubAlpha_4[sta_pix4], calPixel_4);
                Mul(alphaT_[sta_pix4], T_[sta_pix4], alpha_4[sta_pix4], calPixel_4); // alphaT_ = T_ * alpha_

                // PipeBarrier<PIPE_V>();
                Sub(error[sta_pix4], tmpRes_4[sta_pix4], ln1SubAlphaSum[sta_pix4], calPixel_4);  // Gs_4
                Axpy(colorR[sta_pix4], alphaT_[sta_pix4], gsAttr4.GetValue(ATTR_COLOR_R), calPixel_4); // colorR
                Axpy(colorG[sta_pix4], alphaT_[sta_pix4], gsAttr4.GetValue(ATTR_COLOR_G), calPixel_4); // colorG
                Axpy(colorB[sta_pix4], alphaT_[sta_pix4], gsAttr4.GetValue(ATTR_COLOR_B), calPixel_4); // colorB
                Axpy(depth[sta_pix4], alphaT_[sta_pix4], gsAttr4.GetValue(ATTR_DEPTH), calPixel_4);  // depth
                // PipeBarrier<PIPE_V>();
                if (calPixel_4 != 0) {
                    //  ln1SubAlphaSum += ln1SubAlpha_
                    DataCopy(ln1SubAlphaSum[sta_pix4], tmpRes_4[sta_pix4], calPixel_4);
                }
                //  error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum - ln1SubAlpha_
                Sub(error[sta_pix4], error[sta_pix4], ln1SubAlpha_4[sta_pix4], calPixel_4);
                // PipeBarrier<PIPE_V>();
                Exp(T_, ln1SubAlphaSum, calPixel_);

                gsPing = !gsPing;
            }
            // rear tail process
            while (i < currOffset && !clip_break) {
                if (gsPing) {
                    DataCopy(gsAttr1Ping_, gsGm_[gsIdsGm_.GetValue(i) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    gs_flagId = pingId_;
                } else {
                    DataCopy(gsAttr1Pong_, gsGm_[gsIdsGm_.GetValue(i) * NUM_GS_ATTRIBUTES], MIN_DATACOPY_LEN);
                    gs_flagId = pongId_;
                }

                if (alpha_wb_acc >= NUM_STORE_CLIPINDEX * calPixel_) {
                    SetFlag<HardEvent::S_MTE3>(gs_flagId);
                    WaitFlag<HardEvent::S_MTE3>(gs_flagId);

                    DataCopy(alphaClipIndexGm_[NUM_STORE_CLIPINDEX*sta_idx], alphaClipIndexUB_, alpha_wb_acc);

                    SetFlag<HardEvent::MTE3_MTE2>(gs_flagId);
                    SetFlag<HardEvent::MTE3_V>(gs_flagId);
                    WaitFlag<HardEvent::MTE3_MTE2>(gs_flagId);
                    WaitFlag<HardEvent::MTE3_V>(gs_flagId);
                    alpha_wb_acc = 0;
                    sta_idx = i;
                }

                // update ln1SubAlphaSum_
                SetFlag<HardEvent::MTE2_S>(gs_flagId);
                // int64_t gs_index = gsIdsGm_.GetValue(i) * 10; // index*gs属性长度
                
                // compute prob density of pixels
                WaitFlag<HardEvent::MTE2_S>(gs_flagId);
                if (gsPing) {
                    gsAttr1 = gsAttr1Ping_;
                } else {
                    gsAttr1 = gsAttr1Pong_;
                }

                Adds(x_1, tileCoordX, -gsAttr1.GetValue(ATTR_MEAN_X), calPixel_); // meanX
                Adds(y_1, tileCoordY, -gsAttr1.GetValue(ATTR_MEAN_Y), calPixel_); // meanY
                // PipeBarrier<PIPE_V>();
                Mul(gaussWeight_1, x_1, y_1, calPixel_);
                Mul(x2_1, x_1, x_1, calPixel_);
                Mul(y2_1, y_1, y_1, calPixel_);
                // PipeBarrier<PIPE_V>();
                Muls(gaussWeight_1, gaussWeight_1, -gsAttr1.GetValue(ATTR_CONIC_1), calPixel_); // conic1
                Muls(x2_1, x2_1, -gsAttr1.GetValue(ATTR_CONIC_0), calPixel_); // conic0
                Muls(y2_1, y2_1, -gsAttr1.GetValue(ATTR_CONIC_2), calPixel_); // conic2
                // PipeBarrier<PIPE_V>();
                Add(x2y2_1, x2_1, y2_1, calPixel_);
                // PipeBarrier<PIPE_V>();
                Axpy(gaussWeight_1, x2y2_1, 0.5f, calPixel_);
                // PipeBarrier<PIPE_V>();
                // gaussWeight_ = exp(-1/2 (c_0(x-miu_x)^2 + 2c_1(x-miu_x)(y-miu_y) + c_2(y-miu_y)^2))
                Exp(gaussWeight_1, gaussWeight_1, calPixel_);

                // compute transmittances
                Exp(T_, ln1SubAlphaSum, calPixel_); // T_ = exp(ln1SubAlphaSum)
                // PipeBarrier<PIPE_V>();

                // compute alpha, openness
                // opacities,  alpha = o e^(gaussWeight_)
                Muls(alpha_1, gaussWeight_1, gsAttr1.GetValue(ATTR_OPACITY), calPixel_);
                // PipeBarrier<PIPE_V>();

                Mins(alpha_1, alpha_1, 0.999f, calPixel_); // alpha_ = min(alpha_, 0.999)
                // PipeBarrier<PIPE_V>();
                WholeReduceMax(tmpRes_1, alpha_1, oneRepeatNum_, repeatNum_, 1, 1,
                    srcRepStride_, AscendC::ReduceOrder::ORDER_ONLY_VALUE);
                
                PipeBarrier<PIPE_V>();
                CompareScalar(alphaReduceMaxUB_u16, tmpRes_1, 0.01f, AscendC::CMPMODE::GE, NUM_FOR_COMPARESCALAR);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(gs_flagId);
                WaitFlag<HardEvent::V_S>(gs_flagId);
                auto alphaMax = alphaReduceMaxUB_u16.GetValue(0);
                end_pix1 = (BIT_64 - ScalarCountLeadingZero(alphaMax)) *  lineStride_;
                calPixel_1 = ScalarGetCountOfValue<1>(alphaMax) *  lineStride_;
                sta_pix1 = end_pix1 - calPixel_1;
                alpha_wb_acc+=NUM_STORE_CLIPINDEX;
                if (calPixel_1 == 0) {
                    alphaClipIndexUB_.SetValue(alpha_wb_acc-NUM_STORE_CLIPINDEX, nPixel_1d_);
                    alphaClipIndexUB_.SetValue(alpha_wb_acc-1, 0);
                    i++;
                    continue;
                }
                alphaClipIndexUB_.SetValue(alpha_wb_acc-NUM_STORE_CLIPINDEX, (uint8_t)sta_pix1);
                alphaClipIndexUB_.SetValue(alpha_wb_acc-1, (uint8_t)end_pix1);
                
                calPixel_1 = calPixel_1 * nPixel_1d_;
                sta_pix1 = sta_pix1 * nPixel_1d_;

                SetFlag<HardEvent::S_V>(gs_flagId);
                WaitFlag<HardEvent::S_V>(gs_flagId);
                // compute transmittances
                // ln1SubAlpha_ = ln(1 - alpha_)
                // compute transmittances
                // 1 - alpha_
                Muls(ln1SubAlpha_1[sta_pix1], alpha_1[sta_pix1], -1.0f, calPixel_1);

                // PipeBarrier<PIPE_V>();
                Adds(ln1SubAlpha_1[sta_pix1], ln1SubAlpha_1[sta_pix1], 1.0f, calPixel_1);

                // PipeBarrier<PIPE_V>();
                Ln(ln1SubAlpha_1[sta_pix1], ln1SubAlpha_1[sta_pix1], calPixel_1);

                // do alpha blending
                Mul(alphaT_[sta_pix1], T_[sta_pix1], alpha_1[sta_pix1], calPixel_1);
                // PipeBarrier<PIPE_V>();

                Axpy(colorR[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_COLOR_R), calPixel_1); // colorR
                Axpy(colorG[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_COLOR_G), calPixel_1); // colorG
                Axpy(colorB[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_COLOR_B), calPixel_1); // colorB
                Axpy(depth[sta_pix1], alphaT_[sta_pix1], gsAttr1.GetValue(ATTR_DEPTH), calPixel_1);  // depth

                // 误差计算
                // ln1SubAlpha_ = ln1SubAlpha_ - error
                Sub(ln1SubAlpha_1[sta_pix1], ln1SubAlpha_1[sta_pix1], error[sta_pix1], calPixel_1);
                // PipeBarrier<PIPE_V>();
                // tmp = ln1SubAlphaSum + ln1SubAlpha_
                Add(tmpRes_1[sta_pix1], ln1SubAlphaSum[sta_pix1], ln1SubAlpha_1[sta_pix1], calPixel_1);
                // PipeBarrier<PIPE_V>();
                // error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum
                Sub(error[sta_pix1], tmpRes_1[sta_pix1], ln1SubAlphaSum[sta_pix1], calPixel_1);
                // PipeBarrier<PIPE_V>();
                //  error = ln1SubAlphaSum + ln1SubAlpha_ - ln1SubAlphaSum - ln1SubAlpha_
                Sub(error[sta_pix1], error[sta_pix1], ln1SubAlpha_1[sta_pix1], calPixel_1);
                //  ln1SubAlphaSum += ln1SubAlpha_
                DataCopy(ln1SubAlphaSum[sta_pix1], tmpRes_1[sta_pix1], calPixel_1);
                // PipeBarrier<PIPE_V>();

                gsPing = !gsPing;
                i++;
            }

            if (alpha_wb_acc != 0) {
                SetFlag<HardEvent::S_MTE3>(flagId);
                WaitFlag<HardEvent::S_MTE3>(flagId);

                DataCopyExtParams copyParams{1, alpha_wb_acc, 0, 0, 0};

                DataCopyPad(alphaClipIndexGm_[NUM_STORE_CLIPINDEX*sta_idx], alphaClipIndexUB_, copyParams);

                SetFlag<HardEvent::MTE3_MTE2>(flagId);
                SetFlag<HardEvent::MTE3_V>(flagId);
                WaitFlag<HardEvent::MTE3_MTE2>(flagId);
                WaitFlag<HardEvent::MTE3_V>(flagId);
            }
            SetFlag<HardEvent::V_MTE3>(flagId);
            WaitFlag<HardEvent::V_MTE3>(flagId);

            DataCopy(colorRGm[j * calPixel_], colorR, calPixel_);
            DataCopy(colorGGm[j * calPixel_], colorG, calPixel_);
            DataCopy(colorBGm[j * calPixel_], colorB, calPixel_);
            DataCopy(depthGm[j * calPixel_], depth, calPixel_);
            DataCopy(lastCumsumGm[j * calPixel_], ln1SubAlphaSum, calPixel_);
            DataCopy(errorGm[j * calPixel_], error, calPixel_);

            SetFlag<HardEvent::MTE3_MTE2>(flagId);
            SetFlag<HardEvent::MTE3_V>(flagId);
            ping = !ping;
        }

        gsClipIndexUb.SetValue(0, gsClipIndex);
        SetFlag<HardEvent::S_MTE3>(flagId);
        WaitFlag<HardEvent::S_MTE3>(flagId);
        {
            DataCopyExtParams copyParams{1, (uint32_t)(1 * sizeof(int64_t)), 0, 0, 0};

            DataCopyPad(gsClipIndexGm_[tileIdx], gsClipIndexUb, copyParams);
        }
        SetFlag<HardEvent::V_MTE3>(flagId);
        WaitFlag<HardEvent::V_MTE3>(flagId);
        WaitFlag<HardEvent::MTE3_MTE2>(pingId_);
        WaitFlag<HardEvent::MTE3_V>(pingId_);
        WaitFlag<HardEvent::MTE3_MTE2>(pongId_);
        WaitFlag<HardEvent::MTE3_V>(pongId_);
    }

    __aicore__ inline void loopProcess()
    {
        int64_t startScheduleIdx = 0;
        if (vecIdx_ > 0) {
            startScheduleIdx = (int64_t)coreOffsetsGm_.GetValue(vecIdx_ - 1);
        }
        int64_t endScheduleIdx = (int64_t)coreOffsetsGm_.GetValue(vecIdx_);
        for (int64_t scheduleIdx = startScheduleIdx; scheduleIdx < endScheduleIdx; scheduleIdx++) {
            int64_t tileIdx = scheduleGm_.GetValue(scheduleIdx);
            Process(tileIdx);
        }
    }

private:
    int64_t vecIdx_;
    int64_t vecNum_;

    int64_t nPixel_;
    int64_t tileNum_;

    event_t pingId_;
    event_t pongId_;
    bool ping;

    DataCopyParams colorOutParam_;

    GlobalTensor<float> gsGm_;
    GlobalTensor<float> tileCoordsGm_;
    
    GlobalTensor<int64_t> coreOffsetsGm_;
    GlobalTensor<int64_t> scheduleGm_;
    GlobalTensor<int64_t> tileOffsetsGm_;

    GlobalTensor<float> colorRGm_;
    GlobalTensor<float> colorGGm_;
    GlobalTensor<float> colorBGm_;
    GlobalTensor<float> depthGm_;

    GlobalTensor<float> lastCumsumGm_;
    GlobalTensor<float> errorGm_;
    
    // local
    LocalTensor<float> tileCoordXPing_;
    LocalTensor<float> tileCoordYPing_;

    LocalTensor<float> tmpRes_1;
    LocalTensor<float> tmpRes_2;
    LocalTensor<float> tmpRes_3;
    LocalTensor<float> tmpRes_4;

    LocalTensor<float> x_1;
    LocalTensor<float> y_1;
    LocalTensor<float> x2y2_1;
    LocalTensor<float> x2_1;
    LocalTensor<float> y2_1;

    LocalTensor<float> x_2;
    LocalTensor<float> y_2;
    LocalTensor<float> x2y2_2;
    LocalTensor<float> x2_2;
    LocalTensor<float> y2_2;

    LocalTensor<float> x_3;
    LocalTensor<float> y_3;
    LocalTensor<float> x2y2_3;
    LocalTensor<float> x2_3;
    LocalTensor<float> y2_3;

    LocalTensor<float> x_4;
    LocalTensor<float> y_4;
    LocalTensor<float> x2y2_4;
    LocalTensor<float> x2_4;
    LocalTensor<float> y2_4;

    LocalTensor<float> gaussWeight_1;
    LocalTensor<float> gaussWeight_2;
    LocalTensor<float> gaussWeight_3;
    LocalTensor<float> gaussWeight_4;

    LocalTensor<float> alpha_1;
    LocalTensor<float> alpha_2;
    LocalTensor<float> alpha_3;
    LocalTensor<float> alpha_4;
    LocalTensor<float> ln1SubAlpha_1;
    LocalTensor<float> ln1SubAlpha_2;
    LocalTensor<float> ln1SubAlpha_3;
    LocalTensor<float> ln1SubAlpha_4;
    LocalTensor<float> T_;
    LocalTensor<float> alphaT_; // 和alpha共享

    LocalTensor<float> lastLn1SubAlpha_;
    LocalTensor<float> ln1SubAlphaSumPing_;

    // cache_
    LocalTensor<float> cache_;

    // result
    LocalTensor<float> colorRPing_;
    LocalTensor<float> colorGPing_;
    LocalTensor<float> colorBPing_;
    LocalTensor<float> depthPing_;

    LocalTensor<float> errorPing_;

    int64_t pong_offset;
    LocalTensor<float> tileCoordXPong_;
    LocalTensor<float> tileCoordYPong_;

    LocalTensor<float> ln1SubAlphaSumPong_;

    LocalTensor<float> colorRPong_;
    LocalTensor<float> colorGPong_;
    LocalTensor<float> colorBPong_;
    LocalTensor<float> depthPong_;

    LocalTensor<float> errorPong_;
    
    LocalTensor<uint8_t> clipIs_;
    int64_t nClip_;

    int64_t calPixel_;
    GlobalTensor<int64_t> gsClipIndexGm_;
    GlobalTensor<uint8_t> alphaClipIndexGm_; // [n, 2] 0: 起始行数 1: 结束行数

    GlobalTensor<int64_t> gsIdsGm_;
    LocalTensor<int64_t> gsClipIndexPing_;
    LocalTensor<int64_t> gsClipIndexPong_;

    LocalTensor<float> gsAttr1Ping_;
    LocalTensor<float> gsAttr2Ping_;
    LocalTensor<float> gsAttr3Ping_;
    LocalTensor<float> gsAttr4Ping_;
    LocalTensor<float> gsAttr1Pong_;
    LocalTensor<float> gsAttr2Pong_;
    LocalTensor<float> gsAttr3Pong_;
    LocalTensor<float> gsAttr4Pong_;

    LocalTensor<uint8_t> alphaClipIndexUB_;
    LocalTensor<uint16_t> alphaReduceMaxUB_u16;

    int64_t nPixel_1d_;
    int32_t reduceNum_;
    int32_t repeatNum_;
    int32_t oneRepeatNum_;
    int32_t srcRepStride_;

    uint8_t  lineStride_;
};

extern "C" __global__ __aicore__ void calc_render_fwd_double_clip_gsids(
            GM_ADDR gs, GM_ADDR tileCoords, GM_ADDR offsets, GM_ADDR gsIds, GM_ADDR color,
            GM_ADDR depth, GM_ADDR lastCumsum, GM_ADDR error, GM_ADDR gsClipIndex,
            GM_ADDR alphaClipIndex, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    SetAtomicNone();

    auto tilingPtr = reinterpret_cast<__gm__ uint8_t *>(tiling);
    int64_t nPixel = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingPtr + 0 * sizeof(int64_t)));
    int64_t tileNum = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingPtr + 1 * sizeof(int64_t)));
    int64_t nGauss = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingPtr + 2 * sizeof(int64_t)));
    CalcRenderFwdDoubleClipGsids op;
    op.Init(gs,
            tileCoords,
            offsets,
            gsIds,
            nPixel,
            tileNum,
            nGauss,
            color,
            depth,
            lastCumsum,
            error,
            gsClipIndex,
            alphaClipIndex);
    op.loopProcess();
}