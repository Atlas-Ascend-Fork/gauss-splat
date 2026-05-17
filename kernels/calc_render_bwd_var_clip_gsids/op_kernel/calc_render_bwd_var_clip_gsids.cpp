/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t UB_SIZE = 192 * 1024;
constexpr int64_t NUM_FLOATS_PER_BLOCK = 32 / sizeof(float);
constexpr int64_t NUM_IN_FIRST_REPEAT = 64;
constexpr int64_t NUM_IN_SECOND_REPEAT = 32;
constexpr int MAX_TILE_SIZE = 32;
constexpr int64_t SUM_CACHE_SIZE = NUM_IN_FIRST_REPEAT;
constexpr int64_t V_CACHE_SIZE = 128;
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
constexpr int MIN_DATACOPY_LEN = 16;
constexpr int NUM_GS_ATTRIBUTES = 10;
constexpr int NUM_STORE_CLIPINDEX = 2;
constexpr uint8_t BIT_64 = 64;

constexpr uint8_t UB_OFFSET_X_T_X2Y2 = 0;
constexpr uint8_t UB_OFFSET_Y_MINUS = 1;
constexpr uint8_t UB_OFFSET_ALPHA_VGS = 2;
constexpr uint8_t UB_OFFSET_GS_OPAC = 3;
constexpr uint8_t UB_OFFSET_X2_CONIC0 = 4;
constexpr uint8_t UB_OFFSET_Y2_CONIC2 = 5;
constexpr uint8_t UB_OFFSET_LN_MINUS_MEANX = 6;
constexpr uint8_t UB_OFFSET_TMP_ALPHATVG = 7;
constexpr uint8_t UB_OFFSET_ALPHATVB_XY_CONIC1 = 8;
constexpr uint8_t UB_OFFSET_ALPHATVD_XY_HVGS = 9;
constexpr uint8_t UB_OFFSET_ALPHA_CLIP = 10;

constexpr uint8_t UB_OFFSET_TILECOORDX = 0;
constexpr uint8_t UB_OFFSET_TILECOORDY = 1;
constexpr uint8_t UB_OFFSET_COLOR_R = 2;
constexpr uint8_t UB_OFFSET_COLOR_G = 3;
constexpr uint8_t UB_OFFSET_COLOR_B = 4;
constexpr uint8_t UB_OFFSET_DEPTH = 5;
constexpr uint8_t UB_OFFSET_ALPHA = 6;
constexpr uint8_t UB_OFFSET_CUMSUM_R = 7;
constexpr uint8_t UB_OFFSET_CUMSUM_G = 8;
constexpr uint8_t UB_OFFSET_CUMSUN_B = 9;
constexpr uint8_t UB_OFFSET_CUMSUM_D = 10;
constexpr uint8_t UB_OFFSET_LASTCUMSUM = 11;
constexpr uint8_t UB_OFFSET_ERROR = 12;

inline int64_t min(int64_t x, int64_t y)
{
    int64_t res = y;

    bool flag = x <= y;
    if (flag) {
        res = x;
    }

    return res;
}

class CalcRenderBwdVarClipGsids {
public:
    __aicore__ inline CalcRenderBwdVarClipGsids() {}

    __aicore__ inline void Init(
        GM_ADDR vColor, GM_ADDR vDepth, GM_ADDR lastCumsum, GM_ADDR error,
        GM_ADDR gs,
        GM_ADDR tileCoords, GM_ADDR offsets, GM_ADDR gsClipIndex_gsIds, GM_ADDR alphaClipIndex,
        int64_t nPixel, int64_t tileNum, int64_t nGauss,
        GM_ADDR vGs)
    {
        vecIdx_ = GetBlockIdx() * GetSubBlockNum() + GetSubBlockIdx();
        vecNum_ = GetBlockNum() * GetSubBlockNum();

        pingId_ = EVENT_ID6;
        pongId_ = EVENT_ID7;

        nPixel_ = nPixel;
        nPixel_1d_ = sqrt(nPixel_);
        tileNum_ = tileNum;

        if (nPixel_ > MAX_TILE_SIZE * MAX_TILE_SIZE) {
            calPixel_ = MAX_TILE_SIZE * MAX_TILE_SIZE;
        } else {
            calPixel_ = nPixel_;
        }

        // inputs
        vColorRGm_.SetGlobalBuffer((__gm__ float *)vColor);
        vColorGGm_ = vColorRGm_[tileNum_ * nPixel_];
        vColorBGm_ = vColorGGm_[tileNum_ * nPixel_];
        vDepthGm_.SetGlobalBuffer((__gm__ float *)vDepth);

        lastCumsumGm_.SetGlobalBuffer((__gm__ float *)lastCumsum);
        errorGm_.SetGlobalBuffer((__gm__ float *)error);
        
        gsGm_.SetGlobalBuffer((__gm__ float *)gs);
        
        tileCoordsGm_.SetGlobalBuffer((__gm__ float *)tileCoords);

        coreOffsetsGm_.SetGlobalBuffer((__gm__ int64_t *)offsets);
        scheduleGm_ = coreOffsetsGm_[vecNum_];
        tileOffsetsGm_ = scheduleGm_[tileNum_];

        gsClipIndexGm_.SetGlobalBuffer((__gm__ int64_t *)gsClipIndex_gsIds);
        alphaClipIndexGm_.SetGlobalBuffer((__gm__ uint8_t *)alphaClipIndex);
        gsIdsGm_ = gsClipIndexGm_[tileNum];

        // outputs
        vGsGm_.SetGlobalBuffer((__gm__ float *)vGs);

        // shared ub space
        UbBuffInit();
    }

    __aicore__ inline void UbBuffInit()
    {
        // allocate ub space
        TPipe pipe;
        TBuf<QuePosition::VECCALC> ubBuf;
        pipe.InitBuffer(ubBuf, UB_SIZE);
        LocalTensor<uint8_t> ubInbytes = ubBuf.Get<uint8_t>();

        // ub for mask
        mask_ = ubInbytes[0];

        // ub for permanentvariables
        LocalTensor<float> ub = ubInbytes[calPixel_].ReinterpretCast<float>();
        int64_t permanentStart = 0;

        tileCoordX_ = ub[permanentStart + UB_OFFSET_TILECOORDX * calPixel_];
        tileCoordY_ = ub[permanentStart + UB_OFFSET_TILECOORDY * calPixel_];
        vColorR_ = ub[permanentStart + UB_OFFSET_COLOR_R * calPixel_];
        vColorG_ = ub[permanentStart + UB_OFFSET_COLOR_G * calPixel_];
        vColorB_ = ub[permanentStart + UB_OFFSET_COLOR_B * calPixel_];
        vDepth_ = ub[permanentStart + UB_OFFSET_DEPTH * calPixel_];
        vAlpha_ = ub[permanentStart + UB_OFFSET_ALPHA * calPixel_];
        cumsumBufR_ = ub[permanentStart + UB_OFFSET_CUMSUM_R * calPixel_];
        cumsumBufG_ = ub[permanentStart + UB_OFFSET_CUMSUM_G * calPixel_];
        cumsumBufB_ = ub[permanentStart + UB_OFFSET_CUMSUN_B * calPixel_];
        cumsumBufD_ = ub[permanentStart + UB_OFFSET_CUMSUM_D * calPixel_];
        lastCumsum_ = ub[permanentStart + UB_OFFSET_LASTCUMSUM * calPixel_];
        error_ = ub[permanentStart + UB_OFFSET_ERROR * calPixel_];

        // ub for temporary variables in recomputing
        int64_t computationStart = permanentStart + UB_OFFSET_ERROR * calPixel_ + calPixel_;

        x_ = ub[computationStart + UB_OFFSET_X_T_X2Y2 * calPixel_];
        lastT_ = ub[computationStart + UB_OFFSET_X_T_X2Y2 * calPixel_];
        x2y2_ = ub[computationStart + UB_OFFSET_X_T_X2Y2 * calPixel_];

        y_ = ub[computationStart + UB_OFFSET_Y_MINUS * calPixel_];
        _1MinusAlpha_ = ub[computationStart + UB_OFFSET_Y_MINUS * calPixel_];

        alpha_ = ub[computationStart + UB_OFFSET_ALPHA_VGS * calPixel_];
        alphaT_ = ub[computationStart + UB_OFFSET_ALPHA_VGS * calPixel_];
        vGaussWeight_ = ub[computationStart + UB_OFFSET_ALPHA_VGS * calPixel_];

        gaussWeight_ = ub[computationStart + UB_OFFSET_GS_OPAC * calPixel_];
        vOpacitiesSum_ = ub[computationStart + UB_OFFSET_GS_OPAC * calPixel_];

        x2_ = ub[computationStart + UB_OFFSET_X2_CONIC0 * calPixel_];
        vConic0sSum_ = ub[computationStart + UB_OFFSET_X2_CONIC0 * calPixel_];

        y2_ = ub[computationStart + UB_OFFSET_Y2_CONIC2 * calPixel_];
        vConic2sSum_ = ub[computationStart + UB_OFFSET_Y2_CONIC2 * calPixel_];

        // ub for temporary variables in kahan add
        ln1MinusAlpha_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        minusAdapted_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        alphaTVr_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        colorRT_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        colorBT_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        alphaTR_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        vX_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];
        vMeanXsSum_ = ub[computationStart + UB_OFFSET_LN_MINUS_MEANX * calPixel_];

        tempRes_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];
        alphaTVg_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];
        colorGT_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];
        depthT_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];
        alphaTG_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];
        vY_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];
        vMeanYsSum_ = ub[computationStart + UB_OFFSET_TMP_ALPHATVG * calPixel_];

        alphaTVb_ = ub[computationStart + UB_OFFSET_ALPHATVB_XY_CONIC1 * calPixel_];
        vAlphaR_ = ub[computationStart + UB_OFFSET_ALPHATVB_XY_CONIC1 * calPixel_];
        vAlphaB_ = ub[computationStart + UB_OFFSET_ALPHATVB_XY_CONIC1 * calPixel_];
        alphaTB_ = ub[computationStart + UB_OFFSET_ALPHATVB_XY_CONIC1 * calPixel_];
        xY_ = ub[computationStart + UB_OFFSET_ALPHATVB_XY_CONIC1 * calPixel_];
        vConic1sSum_ = ub[computationStart + UB_OFFSET_ALPHATVB_XY_CONIC1 * calPixel_];

        alphaTVd_ = ub[computationStart + UB_OFFSET_ALPHATVD_XY_HVGS * calPixel_];
        vAlphaG_ = ub[computationStart + UB_OFFSET_ALPHATVD_XY_HVGS * calPixel_];
        vAlphaD_ = ub[computationStart + UB_OFFSET_ALPHATVD_XY_HVGS * calPixel_];
        alphaTD_ = ub[computationStart + UB_OFFSET_ALPHATVD_XY_HVGS * calPixel_];
        halfVGaussWeight_ = ub[computationStart + UB_OFFSET_ALPHATVD_XY_HVGS * calPixel_];

        alphaClipIndexUB_ = ub[computationStart + UB_OFFSET_ALPHA_CLIP * calPixel_].ReinterpretCast<uint8_t>();

        int64_t cacheStart = computationStart + UB_OFFSET_ALPHA_CLIP * calPixel_ + calPixel_;
        uint8_t ub_offset = 0;
        vColorRsSum_ = ub[cacheStart + 0];

        ub_offset++;
        vColorGsSum_ = ub[cacheStart + SUM_CACHE_SIZE];

        ub_offset++;
        vColorBsSum_ = ub[cacheStart + ub_offset * SUM_CACHE_SIZE];

        ub_offset++;
        vDepthsSum_ = ub[cacheStart + ub_offset * SUM_CACHE_SIZE];
        cacheStart = cacheStart + SUM_CACHE_SIZE * ub_offset + SUM_CACHE_SIZE;
        vGsCache_ = ub[cacheStart + 0];
        gsAttr_ = ub[cacheStart + MIN_DATACOPY_LEN];
    }

    __aicore__ inline void Process(int64_t tileIdx)
    {
        event_t flagId = pingId_;

        int64_t prevOffset = 0;
        if (tileIdx > 0) {
            prevOffset = (int64_t)tileOffsetsGm_.GetValue(tileIdx - 1);
        }

        int64_t currOffset = (int64_t)gsClipIndexGm_.GetValue(tileIdx);

        GlobalTensor<float> tileCoordsGm = tileCoordsGm_[tileIdx * 2 * nPixel_];
        GlobalTensor<float> vColorRGm = vColorRGm_[tileIdx * nPixel_];
        GlobalTensor<float> vColorGGm = vColorGGm_[tileIdx * nPixel_];
        GlobalTensor<float> vColorBGm = vColorBGm_[tileIdx * nPixel_];
        GlobalTensor<float> vDepthGm = vDepthGm_[tileIdx * nPixel_];
        GlobalTensor<float> lastCumsumGm = lastCumsumGm_[tileIdx * nPixel_];
        GlobalTensor<float> errorGm = errorGm_[tileIdx * nPixel_];

        for (int64_t j = 0; j < (nPixel_ + calPixel_ - 1) / calPixel_; j++) {
            // copyIn
            DataCopy(tileCoordX_, tileCoordsGm[j * calPixel_], calPixel_);
            DataCopy(tileCoordY_, tileCoordsGm[nPixel_ + j * calPixel_], calPixel_);
            DataCopy(vColorR_, vColorRGm[j * calPixel_], calPixel_);
            DataCopy(vColorG_, vColorGGm[j * calPixel_], calPixel_);
            DataCopy(vColorB_, vColorBGm[j * calPixel_], calPixel_);
            DataCopy(vDepth_, vDepthGm[j * calPixel_], calPixel_);
            DataCopy(lastCumsum_, lastCumsumGm[j * calPixel_], calPixel_);
            DataCopy(error_, errorGm[j * calPixel_], calPixel_);

            // initialize
            Duplicate(cumsumBufR_, 0.0f, calPixel_);
            Duplicate(cumsumBufG_, 0.0f, calPixel_);
            Duplicate(cumsumBufB_, 0.0f, calPixel_);
            Duplicate(cumsumBufD_, 0.0f, calPixel_);

            Duplicate(vColorRsSum_, 0.0f, SUM_CACHE_SIZE);
            Duplicate(vColorGsSum_, 0.0f, SUM_CACHE_SIZE);
            Duplicate(vColorBsSum_, 0.0f, SUM_CACHE_SIZE);
            Duplicate(vDepthsSum_, 0.0f, SUM_CACHE_SIZE);

            SetFlag<HardEvent::MTE2_V>(flagId);
            WaitFlag<HardEvent::MTE2_V>(flagId);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::MTE3_V>(flagId);
            bool flag1 = false;
            int sta_pix1 = 0;
            int end_pix1 = nPixel_1d_;
            int32_t calPixel_1 = calPixel_;
            int64_t count = calPixel_;
            int sta_index = 0;
            for (int64_t i = currOffset - 1; i >= prevOffset; i--) {
                count++;
                int64_t gs_index = gsIdsGm_.GetValue(i) * NUM_GS_ATTRIBUTES; // index*gs属性长度
                DataCopy(gsAttr_, gsGm_[gs_index], MIN_DATACOPY_LEN);       // 搬运高斯球属性，搬运数量32B倍数，向下取整
                if (count >= calPixel_) {
                    count = 0;
                    sta_index = i-calPixel_+1;
                    sta_index = (sta_index >= 0) ?  sta_index : 0;
                    DataCopy(alphaClipIndexUB_, alphaClipIndexGm_[NUM_STORE_CLIPINDEX * sta_index],
                        NUM_STORE_CLIPINDEX * calPixel_);
                }
                SetFlag<HardEvent::MTE2_S>(flagId);
                WaitFlag<HardEvent::MTE2_S>(flagId);
                
                sta_pix1 = alphaClipIndexUB_.GetValue((i-sta_index)*NUM_STORE_CLIPINDEX) * nPixel_1d_;
                end_pix1 = alphaClipIndexUB_.GetValue((i-sta_index)*NUM_STORE_CLIPINDEX+1) * nPixel_1d_;
                if (sta_pix1 >= end_pix1) {
                    continue;
                }
                calPixel_1 = end_pix1 - sta_pix1;
                SetFlag<HardEvent::S_V>(flagId);
                WaitFlag<HardEvent::S_V>(flagId);
                WaitFlag<HardEvent::MTE3_V>(flagId);

               // compute alpha
               // compute prob density of pixels
                Adds(x_[sta_pix1], tileCoordX_[sta_pix1], -gsAttr_.GetValue(ATTR_MEAN_X), calPixel_1); // meanX
                Adds(y_[sta_pix1], tileCoordY_[sta_pix1], -gsAttr_.GetValue(ATTR_MEAN_Y), calPixel_1); // meanY
                PipeBarrier<PIPE_V>();
                Mul(gaussWeight_[sta_pix1], x_[sta_pix1], y_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Muls(gaussWeight_[sta_pix1], gaussWeight_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_1), calPixel_1);
                Mul(x2_[sta_pix1], x_[sta_pix1], x_[sta_pix1], calPixel_1);
                Mul(y2_[sta_pix1], y_[sta_pix1], y_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Muls(x_[sta_pix1], x2_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_0), calPixel_1); // conic0
                Muls(y_[sta_pix1], y2_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_2), calPixel_1); // conic2
                PipeBarrier<PIPE_V>();
                Add(x2y2_[sta_pix1], x_[sta_pix1], y_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Axpy(gaussWeight_[sta_pix1], x2y2_[sta_pix1], 0.5f, calPixel_1);
                PipeBarrier<PIPE_V>();
                Exp(gaussWeight_[sta_pix1], gaussWeight_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                // compute alpha, openness
                Muls(alpha_[sta_pix1], gaussWeight_[sta_pix1], gsAttr_.GetValue(ATTR_OPACITY), calPixel_1); // opacities
                PipeBarrier<PIPE_V>();
                Mins(alpha_[sta_pix1], alpha_[sta_pix1], 0.999f, calPixel_1);
                PipeBarrier<PIPE_V>();

                // compute 1 - alpha_
                Muls(_1MinusAlpha_[sta_pix1], alpha_[sta_pix1], -1.0f, calPixel_1);
                PipeBarrier<PIPE_V>();
                Adds(_1MinusAlpha_[sta_pix1], _1MinusAlpha_[sta_pix1], 1.0f, calPixel_1);
                PipeBarrier<PIPE_V>();

                if (i < currOffset - 1) {
                    // update lastCumsum_
                    Ln(ln1MinusAlpha_[sta_pix1], _1MinusAlpha_[sta_pix1], calPixel_1);
                    PipeBarrier<PIPE_V>();
                    Add(minusAdapted_[sta_pix1], ln1MinusAlpha_[sta_pix1], error_[sta_pix1], calPixel_1);
                    PipeBarrier<PIPE_V>();
                    Sub(tempRes_[sta_pix1], lastCumsum_[sta_pix1], minusAdapted_[sta_pix1], calPixel_1);
                    PipeBarrier<PIPE_V>();
                    Sub(error_[sta_pix1], tempRes_[sta_pix1], lastCumsum_[sta_pix1], calPixel_1);
                    PipeBarrier<PIPE_V>();
                    Add(error_[sta_pix1], error_[sta_pix1], minusAdapted_[sta_pix1], calPixel_1);
                    DataCopy(lastCumsum_[sta_pix1], tempRes_[sta_pix1], calPixel_1);
                    PipeBarrier<PIPE_V>();
                }
                // compute grad_colors and gard_depth
                Exp(lastT_[sta_pix1], lastCumsum_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Mul(alphaT_[sta_pix1], lastT_[sta_pix1], alpha_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();

                // compute grad
                Mul(alphaTVr_[sta_pix1], alphaT_[sta_pix1], vColorR_[sta_pix1], calPixel_1);
                Mul(alphaTVg_[sta_pix1], alphaT_[sta_pix1], vColorG_[sta_pix1], calPixel_1);
                Mul(alphaTVb_[sta_pix1], alphaT_[sta_pix1], vColorB_[sta_pix1], calPixel_1);
                Mul(alphaTVd_[sta_pix1], alphaT_[sta_pix1], vDepth_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();

                ReduceSum(vColorRsSum_, alphaTVr_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vColorGsSum_, alphaTVg_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vColorBsSum_, alphaTVb_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vDepthsSum_, alphaTVd_[sta_pix1], tempRes_, calPixel_1);

                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(flagId);

                // compute grad
                Muls(colorRT_[sta_pix1], lastT_[sta_pix1], gsAttr_.GetValue(ATTR_COLOR_R), calPixel_1); // colorR
                Div(vAlphaR_[sta_pix1], cumsumBufR_[sta_pix1], _1MinusAlpha_[sta_pix1], calPixel_1);
                Muls(colorGT_[sta_pix1], lastT_[sta_pix1], gsAttr_.GetValue(ATTR_COLOR_G), calPixel_1); // colorG
                Div(vAlphaG_[sta_pix1], cumsumBufG_[sta_pix1], _1MinusAlpha_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                
                Sub(vAlphaR_[sta_pix1], colorRT_[sta_pix1], vAlphaR_[sta_pix1], calPixel_1);
                Sub(vAlphaG_[sta_pix1], colorGT_[sta_pix1], vAlphaG_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Mul(vAlphaR_[sta_pix1], vAlphaR_[sta_pix1], vColorR_[sta_pix1], calPixel_1);
                Mul(vAlphaG_[sta_pix1], vAlphaG_[sta_pix1], vColorG_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                DataCopy(vAlpha_[sta_pix1], vAlphaR_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Add(vAlpha_[sta_pix1], vAlpha_[sta_pix1], vAlphaG_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Muls(colorBT_[sta_pix1], lastT_[sta_pix1], gsAttr_.GetValue(ATTR_COLOR_B), calPixel_1); // colorB
                Div(vAlphaB_[sta_pix1], cumsumBufB_[sta_pix1], _1MinusAlpha_[sta_pix1], calPixel_1);
                Muls(depthT_[sta_pix1], lastT_[sta_pix1], gsAttr_.GetValue(ATTR_DEPTH), calPixel_1); // depth
                Div(vAlphaD_[sta_pix1], cumsumBufD_[sta_pix1], _1MinusAlpha_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Sub(vAlphaB_[sta_pix1], colorBT_[sta_pix1], vAlphaB_[sta_pix1], calPixel_1);
                Sub(vAlphaD_[sta_pix1], depthT_[sta_pix1], vAlphaD_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Mul(vAlphaB_[sta_pix1], vAlphaB_[sta_pix1], vColorB_[sta_pix1], calPixel_1);
                Mul(vAlphaD_[sta_pix1], vAlphaD_[sta_pix1], vDepth_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Add(vAlpha_[sta_pix1], vAlpha_[sta_pix1], vAlphaB_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();
                Add(vAlpha_[sta_pix1], vAlpha_[sta_pix1], vAlphaD_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();

                WaitFlag<HardEvent::V_S>(flagId);
                vGsCache_.SetValue(ATTR_COLOR_R, vColorRsSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_COLOR_G, vColorGsSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_COLOR_B, vColorBsSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_DEPTH, vDepthsSum_.GetValue(0));
                
                // update cumsum_buf
                Axpy(cumsumBufR_[sta_pix1], alphaT_[sta_pix1], gsAttr_.GetValue(ATTR_COLOR_R), calPixel_1);
                Axpy(cumsumBufG_[sta_pix1], alphaT_[sta_pix1], gsAttr_.GetValue(ATTR_COLOR_G), calPixel_1);
                Axpy(cumsumBufB_[sta_pix1], alphaT_[sta_pix1], gsAttr_.GetValue(ATTR_COLOR_B), calPixel_1);
                Axpy(cumsumBufD_[sta_pix1], alphaT_[sta_pix1], gsAttr_.GetValue(ATTR_DEPTH), calPixel_1);
                PipeBarrier<PIPE_V>();
                
                Muls(alpha_[sta_pix1], gaussWeight_[sta_pix1], gsAttr_.GetValue(ATTR_OPACITY), calPixel_1); // opacities
                PipeBarrier<PIPE_V>();

                CompareScalar(mask_[sta_pix1], alpha_[sta_pix1], 0.999f, CMPMODE::LE, calPixel_1);
                Adds(x_[sta_pix1], tileCoordX_[sta_pix1], -gsAttr_.GetValue(ATTR_MEAN_X), calPixel_1); // meanX
                Adds(y_[sta_pix1], tileCoordY_[sta_pix1], -gsAttr_.GetValue(ATTR_MEAN_Y), calPixel_1);        // meanY
                PipeBarrier<PIPE_V>();

                Select(vAlpha_[sta_pix1], mask_[sta_pix1], vAlpha_[sta_pix1], 0.0f,
                                                    SELMODE::VSEL_TENSOR_SCALAR_MODE, calPixel_1);
                Muls(vX_[sta_pix1], x_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_0), calPixel_1); // conic0
                Muls(vY_[sta_pix1], y_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_2), calPixel_1); // conic2

                PipeBarrier<PIPE_V>();

                Mul(xY_[sta_pix1], x_[sta_pix1], y_[sta_pix1], calPixel_1);
                Mul(vOpacitiesSum_[sta_pix1], vAlpha_[sta_pix1], gaussWeight_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();

                Muls(vGaussWeight_[sta_pix1], vOpacitiesSum_[sta_pix1], -gsAttr_.GetValue(ATTR_OPACITY), calPixel_1);
                PipeBarrier<PIPE_V>();

                Axpy(vMeanXsSum_[sta_pix1], y_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_1), calPixel_1); // conic1
                Axpy(vMeanYsSum_[sta_pix1], x_[sta_pix1], -gsAttr_.GetValue(ATTR_CONIC_1), calPixel_1); // conic1

                Mul(vConic1sSum_[sta_pix1], vGaussWeight_[sta_pix1], xY_[sta_pix1], calPixel_1);
                Muls(halfVGaussWeight_[sta_pix1], vGaussWeight_[sta_pix1], 0.5f, calPixel_1);
                PipeBarrier<PIPE_V>();

                Mul(vConic0sSum_[sta_pix1], halfVGaussWeight_[sta_pix1], x2_[sta_pix1], calPixel_1);
                Mul(vConic2sSum_[sta_pix1], halfVGaussWeight_[sta_pix1], y2_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();

                Mul(vMeanXsSum_[sta_pix1], vMeanXsSum_[sta_pix1], vGaussWeight_[sta_pix1], calPixel_1);
                Mul(vMeanYsSum_[sta_pix1], vMeanYsSum_[sta_pix1], vGaussWeight_[sta_pix1], calPixel_1);
                PipeBarrier<PIPE_V>();

                ReduceSum(vOpacitiesSum_, vOpacitiesSum_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vConic1sSum_, vConic1sSum_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vConic0sSum_, vConic0sSum_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vConic2sSum_, vConic2sSum_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vMeanXsSum_, vMeanXsSum_[sta_pix1], tempRes_, calPixel_1);
                ReduceSum(vMeanYsSum_, vMeanYsSum_[sta_pix1], tempRes_, calPixel_1);
                PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_S>(flagId);
                WaitFlag<HardEvent::V_S>(flagId);
                vGsCache_.SetValue(ATTR_MEAN_X, vMeanXsSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_MEAN_Y, vMeanYsSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_CONIC_0, vConic0sSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_CONIC_1, vConic1sSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_CONIC_2, vConic2sSum_.GetValue(0));
                vGsCache_.SetValue(ATTR_OPACITY, vOpacitiesSum_.GetValue(0));
                {
                    SetFlag<HardEvent::S_MTE3>(flagId);
                    WaitFlag<HardEvent::S_MTE3>(flagId);
                    SetAtomicAdd<float>();

                    DataCopyExtParams copyParams{1, (uint32_t)(NUM_GS_ATTRIBUTES * sizeof(float)), 0, 0, 0};

                    DataCopyPad(vGsGm_[gs_index], vGsCache_, copyParams);
                    SetAtomicNone();
                }

                SetFlag<HardEvent::MTE3_V>(flagId);
            }

            WaitFlag<HardEvent::MTE3_V>(flagId);
        }
    }

    __aicore__ inline void loopProcess()
    {
        int64_t startScheduleIdx = 0;
        if (vecIdx_ > 0) {
            startScheduleIdx = (int64_t)coreOffsetsGm_.GetValue(vecIdx_ - 1);
        }
        int64_t endScheduleIdx = (int64_t)coreOffsetsGm_.GetValue(vecIdx_);

        for (int64_t scheduleIdx = startScheduleIdx; scheduleIdx <endScheduleIdx; scheduleIdx++) {
            int64_t tileIdx = scheduleGm_.GetValue(scheduleIdx);

            Process(tileIdx);
        }
    }

private:
    int64_t vecIdx_;
    int64_t vecNum_;

    int64_t tileNum_;
    int64_t nPixel_;
    int64_t nPixel_1d_;
    int64_t calPixel_;

    event_t pingId_;
    event_t pongId_;

    // gm
    GlobalTensor<float> vColorRGm_;
    GlobalTensor<float> vColorGGm_;
    GlobalTensor<float> vColorBGm_;
    GlobalTensor<float> vDepthGm_;

    GlobalTensor<float> lastCumsumGm_;
    GlobalTensor<float> errorGm_;

    GlobalTensor<float> gsGm_;

    GlobalTensor<float> tileCoordsGm_;

    GlobalTensor<int64_t> coreOffsetsGm_;
    GlobalTensor<int64_t> scheduleGm_;
    GlobalTensor<int64_t> tileOffsetsGm_;

    GlobalTensor<float> vGsGm_;

    GlobalTensor<int64_t> gsClipIndexGm_;
    GlobalTensor<int64_t> gsIdsGm_;
    GlobalTensor<uint8_t> alphaClipIndexGm_;

    // ub
    LocalTensor<uint8_t> alphaClipIndexUB_;
    // after alpha
    LocalTensor<float> vColorR_;
    LocalTensor<float> vColorG_;
    LocalTensor<float> vColorB_;
    LocalTensor<float> vDepth_;
    LocalTensor<float> vAlpha_;

    LocalTensor<float> cumsumBufR_;
    LocalTensor<float> cumsumBufG_;
    LocalTensor<float> cumsumBufB_;
    LocalTensor<float> cumsumBufD_;
    
    LocalTensor<float> lastCumsum_;
    LocalTensor<float> error_;

    LocalTensor<float> tileCoordX_;
    LocalTensor<float> tileCoordY_;

    LocalTensor<float> x_;
    LocalTensor<float> y_;
    LocalTensor<float> x2_;
    LocalTensor<float> y2_;
    LocalTensor<float> x2y2_;

    LocalTensor<float> gaussWeight_;

    LocalTensor<float> lastT_;
    LocalTensor<float> alphaT_;
    
    LocalTensor<float> alpha_;
    LocalTensor<float> _1MinusAlpha_;

    LocalTensor<float> alphaTVr_;
    LocalTensor<float> alphaTVg_;
    LocalTensor<float> alphaTVb_;
    LocalTensor<float> alphaTVd_;

    LocalTensor<float> colorRT_;
    LocalTensor<float> colorGT_;
    LocalTensor<float> colorBT_;
    LocalTensor<float> depthT_;

    LocalTensor<float> vAlphaR_;
    LocalTensor<float> vAlphaG_;
    LocalTensor<float> vAlphaB_;
    LocalTensor<float> vAlphaD_;

    LocalTensor<float> alphaTR_;
    LocalTensor<float> alphaTG_;
    LocalTensor<float> alphaTB_;
    LocalTensor<float> alphaTD_;
    
    LocalTensor<float> ln1MinusAlpha_;
    LocalTensor<float> minusAdapted_;
    LocalTensor<float> tempRes_;

    // before alpha
    LocalTensor<uint8_t> mask_;
    
    LocalTensor<float> vX_;
    LocalTensor<float> vY_;
    LocalTensor<float> xY_;

    LocalTensor<float> vGaussWeight_;
    LocalTensor<float> halfVGaussWeight_;

    LocalTensor<float> vMeanXsSum_;
    LocalTensor<float> vMeanYsSum_;
    LocalTensor<float> vOpacitiesSum_;

    LocalTensor<float> vConic0sSum_;
    LocalTensor<float> vConic1sSum_;
    LocalTensor<float> vConic2sSum_;

    LocalTensor<float> vColorRsSum_;
    LocalTensor<float> vColorGsSum_;
    LocalTensor<float> vColorBsSum_;
    LocalTensor<float> vDepthsSum_;

    LocalTensor<float> vGsCache_;
    LocalTensor<float> gsAttr_;
};

extern "C" __global__ __aicore__ void calc_render_bwd_var_clip_gsids(
    GM_ADDR vColor, GM_ADDR vDepth, GM_ADDR lastCumsum, GM_ADDR error,
    GM_ADDR gs,
    GM_ADDR tileCoords, GM_ADDR offsets, GM_ADDR gsClipIndex_gsIds, GM_ADDR alphaClipIndex,
    GM_ADDR vGs,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    SetAtomicNone();

    auto tilingPtr = reinterpret_cast<__gm__ uint8_t *>(tiling);
    int64_t nPixel = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingPtr + 0 * sizeof(int64_t)));
    int64_t tileNum = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingPtr + 1 * sizeof(int64_t)));
    int64_t nGauss = (*(__gm__ int64_t *)((__gm__ uint8_t *)tilingPtr + 2 * sizeof(int64_t)));

    CalcRenderBwdVarClipGsids op;
    op.Init(
        vColor, vDepth, lastCumsum, error,
        gs,
        tileCoords, offsets, gsClipIndex_gsIds, alphaClipIndex,
        nPixel, tileNum, nGauss,
        vGs);
    
    op.loopProcess();

    SetAtomicNone();
}