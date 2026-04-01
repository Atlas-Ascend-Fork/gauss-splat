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

#ifndef SPHERICAL_HARMONICS_BWD_H
#define SPHERICAL_HARMONICS_BWD_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "spherical_harmonics_bwd_common.h"

namespace SphericalHarmonicsBwdNs {
using namespace AscendC;

class SphericalHarmonicsBwd {
public:
    __aicore__ inline SphericalHarmonicsBwd(){};
    __aicore__ inline void Init(GM_ADDR dirs, GM_ADDR coeffs, GM_ADDR vColors, GM_ADDR vDirs, GM_ADDR vCoeffs,
                                TPipe *Ppipe, const SphericalHarmonicsBwdTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void procD0(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX, LocalTensor<float> &vY,
                                  LocalTensor<float> &vZ, int32_t element);
    __aicore__ inline void procD1(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX, LocalTensor<float> &vY,
                                  LocalTensor<float> &vZ, int32_t element);
    __aicore__ inline void procD2(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX, LocalTensor<float> &vY,
                                  LocalTensor<float> &vZ, int32_t element);
    __aicore__ inline void procD3(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX, LocalTensor<float> &vY,
                                  LocalTensor<float> &vZ, int32_t element);
    __aicore__ inline void procD4(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX, LocalTensor<float> &vY,
                                  LocalTensor<float> &vZ, int32_t element);
    __aicore__ inline void procRet(LocalTensor<float> &vDirs, LocalTensor<float> &vX, LocalTensor<float> &vY,
                                   LocalTensor<float> &vZ, int32_t element);

    __aicore__ inline void CopyIn(int64_t i, int64_t k, int32_t element);
    __aicore__ inline void CopyOut(int64_t i, int64_t k, int32_t element);

    __aicore__ inline void SubProcess(int64_t i, int64_t k, int64_t element);

    int64_t B_ = 0;
    int64_t N_ = 0;
    int64_t perCoreN_ = 0;
    int64_t curCoreN_ = 0;
    int64_t perLoopN_ = 0;
    int64_t lastLoopN_ = 0;
    int64_t perLoopMaxN_ = 0;
    int64_t loopN_ = 0;
    int64_t AlignN = 0;
    int64_t blockIdx = 0;
    int64_t needCoreNum = 0;
    int64_t degree = 0;
    int64_t K_ = 0;
    int64_t bufferLen_ = 0;
    GlobalTensor<float> dirsGm_;
    GlobalTensor<float> coeffsGm_;
    GlobalTensor<float> vColorsGm_;
    GlobalTensor<float> vDirsGm_;
    GlobalTensor<float> vCoeffsGm_;

    LocalTensor<float> copyInTensor;

    LocalTensor<float> x;
    LocalTensor<float> y;
    LocalTensor<float> z;
    LocalTensor<float> z2;
    LocalTensor<float> inorm;

    LocalTensor<float> fC1;
    LocalTensor<float> fS1;
    LocalTensor<float> fC1X;
    LocalTensor<float> fS1X;
    LocalTensor<float> fC1Y;
    LocalTensor<float> fS1Y;

    LocalTensor<float> fC2;
    LocalTensor<float> fS2;
    LocalTensor<float> fC2X;
    LocalTensor<float> fS2X;
    LocalTensor<float> fC2Y;
    LocalTensor<float> fS2Y;

    LocalTensor<float> fC3;
    LocalTensor<float> fS3;
    LocalTensor<float> fC3X;
    LocalTensor<float> fS3X;
    LocalTensor<float> fC3Y;
    LocalTensor<float> fS3Y;

    LocalTensor<float> pSH6;
    LocalTensor<float> pSH6Z;
    LocalTensor<float> pSH12;
    LocalTensor<float> pSH12Z;

    LocalTensor<float> dirsLocal;
    LocalTensor<float> coeffsLocal;
    LocalTensor<float> vColorsLocal;
    LocalTensor<float> vDirsLocal;
    LocalTensor<float> vCoeffsLocal;

    TQue<QuePosition::VECIN, 1> dirsQue_;
    TQue<QuePosition::VECIN, 1> coeffsQue_;
    TQue<QuePosition::VECIN, 1> vColorsQue_;

    TQue<QuePosition::VECOUT, 1> vDirsQue_;
    TQue<QuePosition::VECOUT, 1> vCoeffsQue_;

    TBuf<TPosition::VECCALC> intermediateBuf_;
    TBuf<TPosition::VECCALC> calBuf_;
};

__aicore__ inline void SphericalHarmonicsBwd::Init(GM_ADDR dirs, GM_ADDR coeffs, GM_ADDR vColors, GM_ADDR vDirs,
                                                   GM_ADDR vCoeffs, TPipe *Ppipe,
                                                   const SphericalHarmonicsBwdTilingData *tilingData)
{
    blockIdx = GetBlockIdx();
    B_ = tilingData->batchNum;
    N_ = tilingData->gaussNum;
    degree = tilingData->degree;
    perCoreN_ = tilingData->blockLength;
    perLoopMaxN_ = tilingData->perloopNum;
    needCoreNum = tilingData->needCoreNum;
    bufferLen_ = tilingData->bufferLen;
    // 本核需要处理多长的N
    curCoreN_ = perCoreN_;
    if (blockIdx == needCoreNum - 1) {
        // curCoreN_ = 尾核
        curCoreN_ = tilingData->lastcoreNum;
    }

    // 计算一次循环处理多长的N，防止超出curCoreN_
    perLoopN_ = Min(perLoopMaxN_, curCoreN_);
    // 计算需要多少个循环处理curCoreN_
    if (perLoopN_ != 0) {
        loopN_ = curCoreN_ / perLoopN_;
    } else {
        return;
    }
    // 计算最后一次循环需要处理多长的N
    lastLoopN_ = curCoreN_ - perLoopN_ * loopN_;

    K_ = tilingData->K;

    dirsGm_.SetGlobalBuffer((__gm__ float *)dirs);
    coeffsGm_.SetGlobalBuffer((__gm__ float *)coeffs);
    vColorsGm_.SetGlobalBuffer((__gm__ float *)vColors);
    vDirsGm_.SetGlobalBuffer((__gm__ float *)vDirs);
    vCoeffsGm_.SetGlobalBuffer((__gm__ float *)vCoeffs);

    Ppipe->InitBuffer(dirsQue_, 1, CONSTDIM_THREE * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(coeffsQue_, 1, CONSTDIM_THREE * K_ * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(vColorsQue_, 1, CONSTDIM_THREE * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(vDirsQue_, 1, CONSTDIM_THREE * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(vCoeffsQue_, 1, CONSTDIM_THREE * K_ * AlignBytes(perLoopN_, sizeof(float)));

    Ppipe->InitBuffer(intermediateBuf_, bufferLen_ * AlignBytes(perLoopN_, sizeof(float)));
    Ppipe->InitBuffer(calBuf_, CALBUF_SIZE * AlignBytes(perLoopN_, sizeof(float)));
}

__aicore__ inline void SphericalHarmonicsBwd::CopyIn(int64_t i, int64_t k, int32_t element)
{
    uint32_t len = element * sizeof(float);
    uint32_t stride = (N_ - element) * sizeof(float);
    int64_t offsetGm3 = i * CONSTDIM_THREE * N_ + perLoopN_ * k + perCoreN_ * blockIdx;
    int64_t offsetGm3k = i * CONSTDIM_THREE * K_ * N_ + perLoopN_ * k + perCoreN_ * blockIdx;

    DataCopyExtParams dirsDataCopyParams{static_cast<uint16_t>(CONSTDIM_THREE), static_cast<uint32_t>(len), stride, 0,
                                         0};
    DataCopyExtParams coeffsDataCopyParams{static_cast<uint16_t>(CONSTDIM_THREE * K_), static_cast<uint32_t>(len),
                                           stride, 0, 0};
    DataCopyPadExtParams<float> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(dirsLocal, dirsGm_[offsetGm3], dirsDataCopyParams, dataCopyPadParams);
    DataCopyPad(coeffsLocal, coeffsGm_[offsetGm3k], coeffsDataCopyParams, dataCopyPadParams);
    DataCopyPad(vColorsLocal, vColorsGm_[offsetGm3], dirsDataCopyParams, dataCopyPadParams);
    pipe_barrier(PIPE_ALL);

    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
}

__aicore__ inline void SphericalHarmonicsBwd::CopyOut(int64_t i, int64_t k, int32_t element)
{
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    uint32_t len = element * sizeof(float);
    uint32_t stride = (N_ - element) * sizeof(float);
    int64_t offsetGm3 = i * CONSTDIM_THREE * N_ + perLoopN_ * k + perCoreN_ * blockIdx;
    int64_t offsetGm3k = i * CONSTDIM_THREE * K_ * N_ + perLoopN_ * k + perCoreN_ * blockIdx;
    DataCopyExtParams vDirsDataCopyParams{static_cast<uint16_t>(CONSTDIM_THREE), static_cast<uint32_t>(len), 0, stride,
                                          0};
    DataCopyExtParams vCoeffsDataCopyParams{static_cast<uint16_t>(CONSTDIM_THREE * K_), static_cast<uint32_t>(len), 0,
                                            stride, 0};
    DataCopyPad(vDirsGm_[offsetGm3], vDirsLocal, vDirsDataCopyParams);
    DataCopyPad(vCoeffsGm_[offsetGm3k], vCoeffsLocal, vCoeffsDataCopyParams);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

__aicore__ inline void SphericalHarmonicsBwd::SubProcess(int64_t i, int64_t k, int64_t element)
{
    int64_t offsetGm = i * N_ + perCoreN_ * blockIdx + k * perLoopN_;
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> intermediateTensor = intermediateBuf_.Get<float>();

    x = dirsLocal[X_OFFSET];
    y = dirsLocal[offset * Y_OFFSET];
    z = dirsLocal[offset * Z_OFFSET];

    z2 = intermediateTensor[Z2_OFFSET];
    inorm = intermediateTensor[offset * INORM_OFFSET];

    LocalTensor<float> vX = intermediateTensor[offset * VX_OFFSET];
    LocalTensor<float> vY = intermediateTensor[offset * VY_OFFSET];
    LocalTensor<float> vZ = intermediateTensor[offset * VZ_OFFSET];

    CopyIn(i, k, element);

    Duplicate<float>(vDirsLocal, 0.0f, VDIRS_OFFSET * offset);

    if (degree == DEGREE_ZERO) {
        procD0(vCoeffsLocal, vX, vY, vZ, element);
    } else if (degree == 1) {
        procD0(vCoeffsLocal, vX, vY, vZ, element);
        procD1(vCoeffsLocal, vX, vY, vZ, element);
        procRet(vDirsLocal, vX, vY, vZ, element);
    } else if (degree == DEGREE_TWO) {
        fC1 = intermediateTensor[offset * FC1_OFFSET];
        fS1 = intermediateTensor[offset * FS1_OFFSET];
        fC1X = intermediateTensor[offset * FC1X_OFFSET];
        fS1X = intermediateTensor[offset * FS1X_OFFSET];
        fC1Y = intermediateTensor[offset * FC1Y_OFFSET];
        fS1Y = intermediateTensor[offset * FS1Y_OFFSET];

        pSH6 = intermediateTensor[offset * PSH6BUF_OFFSET];
        pSH6Z = intermediateTensor[offset * PSH6Z_OFFSET];
        procD0(vCoeffsLocal, vX, vY, vZ, element);
        procD1(vCoeffsLocal, vX, vY, vZ, element);
        procD2(vCoeffsLocal, vX, vY, vZ, element);
        procRet(vDirsLocal, vX, vY, vZ, element);
    } else if (degree == DEGREE_THREE) {
        fC1 = intermediateTensor[offset * FC1_OFFSET];
        fS1 = intermediateTensor[offset * FS1_OFFSET];
        fC1X = intermediateTensor[offset * FC1X_OFFSET];
        fS1X = intermediateTensor[offset * FS1X_OFFSET];
        fC1Y = intermediateTensor[offset * FC1Y_OFFSET];
        fS1Y = intermediateTensor[offset * FS1Y_OFFSET];
        pSH6 = intermediateTensor[offset * PSH6BUF_OFFSET];
        pSH6Z = intermediateTensor[offset * PSH6Z_OFFSET];

        fC2 = intermediateTensor[offset * FC2_OFFSET];
        fS2 = intermediateTensor[offset * FS2_OFFSET];
        fC2X = intermediateTensor[offset * FC2X_OFFSET];
        fS2X = intermediateTensor[offset * FS2X_OFFSET];
        fC2Y = intermediateTensor[offset * FC2Y_OFFSET];
        fS2Y = intermediateTensor[offset * FS2Y_OFFSET];

        pSH12 = intermediateTensor[offset * PSH12BUF_OFFSET];
        pSH12Z = intermediateTensor[offset * PSH12Z_OFFSET];
        procD0(vCoeffsLocal, vX, vY, vZ, element);
        procD1(vCoeffsLocal, vX, vY, vZ, element);
        procD2(vCoeffsLocal, vX, vY, vZ, element);
        procD3(vCoeffsLocal, vX, vY, vZ, element);
        procRet(vDirsLocal, vX, vY, vZ, element);
    } else if (degree == DEGREE_FOUR) {
        fC1 = intermediateTensor[offset * FC1_OFFSET];
        fS1 = intermediateTensor[offset * FS1_OFFSET];
        fC1X = intermediateTensor[offset * FC1X_OFFSET];
        fS1X = intermediateTensor[offset * FS1X_OFFSET];
        fC1Y = intermediateTensor[offset * FC1Y_OFFSET];
        fS1Y = intermediateTensor[offset * FS1Y_OFFSET];
        pSH6 = intermediateTensor[offset * PSH6BUF_OFFSET];
        pSH6Z = intermediateTensor[offset * PSH6Z_OFFSET];

        fC2 = intermediateTensor[offset * FC2_OFFSET];
        fS2 = intermediateTensor[offset * FS2_OFFSET];
        fC2X = intermediateTensor[offset * FC2X_OFFSET];
        fS2X = intermediateTensor[offset * FS2X_OFFSET];
        fC2Y = intermediateTensor[offset * FC2Y_OFFSET];
        fS2Y = intermediateTensor[offset * FS2Y_OFFSET];

        pSH12 = intermediateTensor[offset * PSH12BUF_OFFSET];
        pSH12Z = intermediateTensor[offset * PSH12Z_OFFSET];

        fC3 = intermediateTensor[offset * FC3_OFFSET];
        fS3 = intermediateTensor[offset * FS3_OFFSET];
        fC3X = intermediateTensor[offset * FC3X_OFFSET];
        fS3X = intermediateTensor[offset * FS3X_OFFSET];
        fC3Y = intermediateTensor[offset * FC3Y_OFFSET];
        fS3Y = intermediateTensor[offset * FS3Y_OFFSET];
        procD0(vCoeffsLocal, vX, vY, vZ, element);
        procD1(vCoeffsLocal, vX, vY, vZ, element);
        procD2(vCoeffsLocal, vX, vY, vZ, element);
        procD3(vCoeffsLocal, vX, vY, vZ, element);
        procD4(vCoeffsLocal, vX, vY, vZ, element);
        procRet(vDirsLocal, vX, vY, vZ, element);
    }

    CopyOut(i, k, element);
}

__aicore__ inline void SphericalHarmonicsBwd::Process()
{
    dirsLocal = dirsQue_.AllocTensor<float>();
    coeffsLocal = coeffsQue_.AllocTensor<float>();
    vColorsLocal = vColorsQue_.AllocTensor<float>();
    vDirsLocal = vDirsQue_.AllocTensor<float>();
    vCoeffsLocal = vCoeffsQue_.AllocTensor<float>();
    for (int64_t i = 0; i < B_; ++i) {
        int64_t k = 0;
        for (; k < loopN_; ++k) {
            SubProcess(i, k, perLoopN_);
        }
        if (lastLoopN_ > 0) {
            SubProcess(i, k, lastLoopN_);
        }
    }
    dirsQue_.FreeTensor(dirsLocal);
    coeffsQue_.FreeTensor(coeffsLocal);
    vColorsQue_.FreeTensor(vColorsLocal);
    vDirsQue_.FreeTensor(vDirsLocal);
    vCoeffsQue_.FreeTensor(vCoeffsLocal);
}

__aicore__ inline void SphericalHarmonicsBwd::procD0(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX,
                                                     LocalTensor<float> &vY, LocalTensor<float> &vZ,
                                                     int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    Muls(vCoeffs, vColorsLocal, L0_M0_SH_PARAM, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
};
__aicore__ inline void SphericalHarmonicsBwd::procD1(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX,
                                                     LocalTensor<float> &vY, LocalTensor<float> &vZ,
                                                     int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> calTensor = calBuf_.Get<float>();
    LocalTensor<float> tmpX = calTensor[offset * TMPX_OFFSET];
    LocalTensor<float> tmpY = calTensor[offset * TMPY_OFFSET];
    LocalTensor<float> tmpZ = calTensor[offset * TMPZ_OFFSET];
    LocalTensor<float> tmpVdir = calTensor[offset * TMPVDIR_OFFSET];

    LocalTensor<float> tmpX2 = calTensor[offset * TMPX2_OFFSET];
    LocalTensor<float> tmpY2 = calTensor[offset * TMPY2_OFFSET];
    LocalTensor<float> tmpZ2 = calTensor[offset * TMPZ2_OFFSET];

    Mul(tmpX2, x, x, element);
    Mul(tmpY2, y, y, element);
    Mul(tmpZ2, z, z, element);
    Add(inorm, tmpX2, tmpY2, element);
    pipe_barrier(PIPE_V);
    Add(inorm, inorm, tmpZ2, element);
    pipe_barrier(PIPE_V);
    Sqrt(inorm, inorm, element);
    Maxs(inorm, inorm, 1e-12f, element);
    Div(x, x, inorm, element);
    Div(y, y, inorm, element);
    Div(z, z, inorm, element);

    Muls(tmpX, x, -L1_M0_SH_PARAM, element);
    Muls(tmpY, y, -L1_M0_SH_PARAM, element);
    Muls(tmpZ, z, L1_M0_SH_PARAM, element);
    pipe_barrier(PIPE_V);

    Mul(vCoeffs[offset * (PSH1_OFFSET + X_OFFSET)], vColorsLocal[offset * X_OFFSET], tmpY, element);
    Mul(vCoeffs[offset * (PSH1_OFFSET + Y_OFFSET)], vColorsLocal[offset * Y_OFFSET], tmpY, element);
    Mul(vCoeffs[offset * (PSH1_OFFSET + Z_OFFSET)], vColorsLocal[offset * Z_OFFSET], tmpY, element);

    Mul(vCoeffs[offset * (PSH2_OFFSET + X_OFFSET)], vColorsLocal[offset * X_OFFSET], tmpZ, element);
    Mul(vCoeffs[offset * (PSH2_OFFSET + Y_OFFSET)], vColorsLocal[offset * Y_OFFSET], tmpZ, element);
    Mul(vCoeffs[offset * (PSH2_OFFSET + Z_OFFSET)], vColorsLocal[offset * Z_OFFSET], tmpZ, element);

    Mul(vCoeffs[offset * (PSH3_OFFSET + X_OFFSET)], vColorsLocal[offset * X_OFFSET], tmpX, element);
    Mul(vCoeffs[offset * (PSH3_OFFSET + Y_OFFSET)], vColorsLocal[offset * Y_OFFSET], tmpX, element);
    Mul(vCoeffs[offset * (PSH3_OFFSET + Z_OFFSET)], vColorsLocal[offset * Z_OFFSET], tmpX, element);

    pipe_barrier(PIPE_V);

    // v_x
    Mul(tmpVdir, coeffsLocal[offset * PSH3_OFFSET], vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    Add(vX, tmpVdir[offset * X_OFFSET], tmpVdir[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tmpVdir[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);
    Muls(vX, vX, -L1_M0_SH_PARAM, element);
    pipe_barrier(PIPE_V);

    // v_y
    Mul(tmpVdir, coeffsLocal[offset * PSH1_OFFSET], vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    Add(vY, tmpVdir[offset * X_OFFSET], tmpVdir[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tmpVdir[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);
    Muls(vY, vY, -L1_M0_SH_PARAM, element);
    pipe_barrier(PIPE_V);

    // v_x
    Mul(tmpVdir, coeffsLocal[offset * PSH2_OFFSET], vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    Add(vZ, tmpVdir[offset * X_OFFSET], tmpVdir[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tmpVdir[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);
    Muls(vZ, vZ, L1_M0_SH_PARAM, element);
    pipe_barrier(PIPE_V);
};
__aicore__ inline void SphericalHarmonicsBwd::procD2(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX,
                                                     LocalTensor<float> &vY, LocalTensor<float> &vZ,
                                                     int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> calTensor = calBuf_.Get<float>();
    LocalTensor<float> fTmp0B = calTensor;
    LocalTensor<float> temp = calTensor[offset];
    LocalTensor<float> tempSum = calTensor[offset * TMPSUMD2_OFFSET];
    LocalTensor<float> pSHXYZ = calTensor[offset * PSHXYZD2_OFFSET];
    float fTmp0BZ = L2_M1_SH_PARAM;

    Mul(z2, z, z, element);
    Muls(fTmp0B, z, L2_M1_SH_PARAM, element);
    Mul(temp, y, y, element);
    Mul(fC1, x, x, element);
    pipe_barrier(PIPE_V);
    Sub(fC1, fC1, temp, element);

    Mul(fS1, x, y, element);
    pipe_barrier(PIPE_V);
    Muls(fS1, fS1, 2.0f, element);

    // pSH6
    Muls(pSH6, z2, L2_M0_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(pSH6, pSH6, L2_M0_SH_PARAM_2, element);
    pipe_barrier(PIPE_V);
    Adds(vCoeffs[offset * PSH6_OFFSET], pSH6, 0.0f, element);
    // pSH7
    Mul(vCoeffs[offset * PSH7_OFFSET], x, fTmp0B, element);
    // pSH5
    Mul(vCoeffs[offset * PSH5_OFFSET], y, fTmp0B, element);
    // pSH8
    Muls(vCoeffs[offset * PSH8_OFFSET], fC1, L2_M2_SH_PARAM, element);
    // pSH4
    Muls(vCoeffs[offset * PSH4_OFFSET], fS1, L2_M2_SH_PARAM, element);
    pipe_barrier(PIPE_V);

    // vCoeffs[..., NUM_FOUR:NUM_NINE, NUM_THREE]
    Mul(vCoeffs[offset * (PSH4_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH4_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH4_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH4_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH5_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH5_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH5_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH5_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH6_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH6_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH6_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH6_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH7_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH7_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH7_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH7_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH8_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH8_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH8_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH8_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    pipe_barrier(PIPE_V);
    Mul(vCoeffs[offset * (PSH4_OFFSET + X_OFFSET)], vCoeffs[offset * PSH4_OFFSET], vColorsLocal[offset * X_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH5_OFFSET + X_OFFSET)], vCoeffs[offset * PSH5_OFFSET], vColorsLocal[offset * X_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH6_OFFSET + X_OFFSET)], vCoeffs[offset * PSH6_OFFSET], vColorsLocal[offset * X_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH7_OFFSET + X_OFFSET)], vCoeffs[offset * PSH7_OFFSET], vColorsLocal[offset * X_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH8_OFFSET + X_OFFSET)], vCoeffs[offset * PSH8_OFFSET], vColorsLocal[offset * X_OFFSET],
        element);

    Muls(fC1X, x, 2.0f, element);
    Muls(fC1Y, y, -2.0f, element);
    Muls(fS1X, y, 2.0f, element);
    Muls(fS1Y, x, 2.0f, element);

    // pSH4X * coeffsLocal
    Muls(pSHXYZ, fS1X, L2_M2_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH4_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH4_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH4_OFFSET + Z_OFFSET)], element);

    // pSH8X * coeffsLocal
    Muls(pSHXYZ, fC1X, L2_M2_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH8_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH8_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH8_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH7X * coeffsLocal
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], fTmp0B, coeffsLocal[offset * (PSH7_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], fTmp0B, coeffsLocal[offset * (PSH7_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], fTmp0B, coeffsLocal[offset * (PSH7_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vX + reduceSum
    Add(vX, vX, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    // pSH4Y * coeffsLocal
    Muls(pSHXYZ, fS1Y, L2_M2_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH4_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH4_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH4_OFFSET + Z_OFFSET)], element);

    // pSH8Y * coeffsLocal
    Muls(pSHXYZ, fC1Y, L2_M2_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH8_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH8_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH8_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH5Y * coeffsLocal
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], fTmp0B, coeffsLocal[offset * (PSH5_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], fTmp0B, coeffsLocal[offset * (PSH5_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], fTmp0B, coeffsLocal[offset * (PSH5_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vY + reduceSum
    Add(vY, vY, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    // pSH6Z * coeffsLocal
    Muls(pSH6Z, z, 2.0f * L2_M0_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSH6Z, coeffsLocal[offset * (PSH6_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSH6Z, coeffsLocal[offset * (PSH6_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSH6Z, coeffsLocal[offset * (PSH6_OFFSET + Z_OFFSET)], element);

    // pSH7Z * coeffsLocal
    Muls(pSHXYZ, x, fTmp0BZ, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH7_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH7_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH7_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH5Z * coeffsLocal
    Muls(pSHXYZ, y, fTmp0BZ, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH5_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH5_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH5_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vZ + reduceSum
    Add(vZ, vZ, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);
};

__aicore__ inline void SphericalHarmonicsBwd::procD3(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX,
                                                     LocalTensor<float> &vY, LocalTensor<float> &vZ,
                                                     int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> calTensor = calBuf_.Get<float>();
    LocalTensor<float> fTmp0C = calTensor[offset * FTMP0C_OFFSET];
    LocalTensor<float> fTmp1B = calTensor[offset * FTMP1B_OFFSET];
    LocalTensor<float> fTmp0CZ = calTensor[offset * FTMP0CZ_OFFSET];
    LocalTensor<float> temp = calTensor[offset * TEMPD3_OFFSET];
    LocalTensor<float> tempSum = calTensor[offset * TEMPSUMD3_OFFSET];
    LocalTensor<float> pSHXYZ = calTensor[offset * PSHXYZD3_OFFSET];
    float fTmp1BZ = L3_M2_SH_PARAM;

    Muls(fTmp0C, z2, L3_M1_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(fTmp0C, fTmp0C, L3_M1_SH_PARAM_2, element);
    Muls(fTmp1B, z, L3_M2_SH_PARAM, element);

    Mul(fC2, x, fC1, element);
    Mul(temp, y, fS1, element);
    pipe_barrier(PIPE_V);
    Sub(fC2, fC2, temp, element);
    pipe_barrier(PIPE_V);

    Mul(fS2, x, fS1, element);
    Mul(temp, y, fC1, element);
    pipe_barrier(PIPE_V);
    Add(fS2, fS2, temp, element);
    pipe_barrier(PIPE_V);

    // pSH9
    Muls(vCoeffs[offset * PSH9_OFFSET], fS2, L3_M3_SH_PARAM, element);
    // pSH10
    Mul(vCoeffs[offset * PSH10_OFFSET], fTmp1B, fS1, element);
    // pSH11
    Mul(vCoeffs[offset * PSH11_OFFSET], fTmp0C, y, element);
    // pSH12
    Muls(pSH12, z2, L3_M0_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(pSH12, pSH12, L3_M0_SH_PARAM_2, element);
    pipe_barrier(PIPE_V);
    Mul(pSH12, pSH12, z, element);
    pipe_barrier(PIPE_V);
    Adds(vCoeffs[offset * PSH12_OFFSET], pSH12, 0.0f, element);
    // pSH13
    Mul(vCoeffs[offset * PSH13_OFFSET], fTmp0C, x, element);
    // pSH14
    Mul(vCoeffs[offset * PSH14_OFFSET], fTmp1B, fC1, element);
    // pSH15
    Muls(vCoeffs[offset * PSH15_OFFSET], fC2, L3_M3_SH_PARAM, element);
    pipe_barrier(PIPE_V);

    // vCoeffs[..., NUM_NINE:NUM_SIXTEEN, NUM_THREE]
    Mul(vCoeffs[offset * (PSH9_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH9_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH9_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH9_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH10_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH10_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH10_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH10_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH11_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH11_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH11_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH11_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH12_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH12_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH12_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH12_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH13_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH13_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH13_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH13_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH14_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH14_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH14_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH14_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH15_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH15_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH15_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH15_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    pipe_barrier(PIPE_V);
    Mul(vCoeffs[offset * PSH9_OFFSET], vCoeffs[offset * PSH9_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH10_OFFSET], vCoeffs[offset * PSH10_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH11_OFFSET], vCoeffs[offset * PSH11_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH12_OFFSET], vCoeffs[offset * PSH12_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH13_OFFSET], vCoeffs[offset * PSH13_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH14_OFFSET], vCoeffs[offset * PSH14_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH15_OFFSET], vCoeffs[offset * PSH15_OFFSET], vColorsLocal[offset * X_OFFSET], element);

    // fTmp0CZ
    Muls(fTmp0CZ, z, 2.0f * L3_M1_SH_PARAM_1, element);
    // fC2X
    Mul(fC2X, x, fC1X, element);
    Mul(temp, y, fS1X, element);
    pipe_barrier(PIPE_V);
    Sub(fC2X, fC2X, temp, element);
    pipe_barrier(PIPE_V);
    Add(fC2X, fC2X, fC1, element);

    // fC2Y
    Mul(fC2Y, x, fC1Y, element);
    Mul(temp, y, fS1Y, element);
    pipe_barrier(PIPE_V);
    Sub(fC2Y, fC2Y, temp, element);
    pipe_barrier(PIPE_V);
    Sub(fC2Y, fC2Y, fS1, element);

    // fS2X
    Mul(fS2X, x, fS1X, element);
    Mul(temp, y, fC1X, element);
    pipe_barrier(PIPE_V);
    Add(fS2X, fS2X, temp, element);
    pipe_barrier(PIPE_V);
    Add(fS2X, fS2X, fS1, element);

    // fS2Y
    Mul(fS2Y, x, fS1Y, element);
    Mul(temp, y, fC1Y, element);
    pipe_barrier(PIPE_V);
    Add(fS2Y, fS2Y, temp, element);
    pipe_barrier(PIPE_V);
    Add(fS2Y, fS2Y, fC1, element);

    pipe_barrier(PIPE_V);

    // pSH9X * coeffsLocal
    Muls(pSHXYZ, fS2X, L3_M3_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH9_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH9_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH9_OFFSET + Z_OFFSET)], element);

    // pSH15X * coeffsLocal
    Muls(pSHXYZ, fC2X, L3_M3_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH15_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH15_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH15_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH10X * coeffsLocal
    Mul(pSHXYZ, fTmp1B, fS1X, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH14X * coeffsLocal
    Mul(pSHXYZ, fTmp1B, fC1X, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH13X * coeffsLocal
    Mul(temp[offset * X_OFFSET], fTmp0C, coeffsLocal[offset * (PSH13_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], fTmp0C, coeffsLocal[offset * (PSH13_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], fTmp0C, coeffsLocal[offset * (PSH13_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vX + reduceSum
    Add(vX, vX, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    // pSH9Y * coeffsLocal
    Muls(pSHXYZ, fS2Y, L3_M3_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH9_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH9_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH9_OFFSET + Z_OFFSET)], element);

    // pSH15Y * coeffsLocal
    Muls(pSHXYZ, fC2Y, L3_M3_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH15_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH15_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH15_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH10Y * coeffsLocal
    Mul(pSHXYZ, fTmp1B, fS1Y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH14Y * coeffsLocal
    Mul(pSHXYZ, fTmp1B, fC1Y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH11Y * coeffsLocal
    Mul(temp[offset * X_OFFSET], fTmp0C, coeffsLocal[offset * (PSH11_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], fTmp0C, coeffsLocal[offset * (PSH11_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], fTmp0C, coeffsLocal[offset * (PSH11_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vY + reduceSum
    Add(vY, vY, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    // pSH12Z * coeffsLocal
    Muls(pSH12Z, z2, 3.0f * L3_M0_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(pSH12Z, pSH12Z, L3_M0_SH_PARAM_2, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSH12Z, coeffsLocal[offset * (PSH12_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSH12Z, coeffsLocal[offset * (PSH12_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSH12Z, coeffsLocal[offset * (PSH12_OFFSET + Z_OFFSET)], element);

    // pSH13Z * coeffsLocal
    Mul(pSHXYZ, fTmp0CZ, x, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH13_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH13_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH13_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH11Z * coeffsLocal
    Mul(pSHXYZ, fTmp0CZ, y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH11_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH11_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH11_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH14Z * coeffsLocal
    Muls(pSHXYZ, fC1, fTmp1BZ, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH14_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH10Z * coeffsLocal
    Muls(pSHXYZ, fS1, fTmp1BZ, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH10_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vZ + reduceSum
    Add(vZ, vZ, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);
}

__aicore__ inline void SphericalHarmonicsBwd::procD4(LocalTensor<float> &vCoeffs, LocalTensor<float> &vX,
                                                     LocalTensor<float> &vY, LocalTensor<float> &vZ,
                                                     int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> calTensor = calBuf_.Get<float>();
    LocalTensor<float> fTmp0D = calTensor[offset * FTMP0D_OFFSET];
    LocalTensor<float> fTmp1C = calTensor[offset * FTMP1C_OFFSET];
    LocalTensor<float> fTmp2B = calTensor[offset * FTMP2B_OFFSET];
    LocalTensor<float> fTmp0DZ = calTensor[offset * FTMP0DZ_OFFSET];
    LocalTensor<float> fTmp1CZ = calTensor[offset * FTMP1CZ_OFFSET];
    LocalTensor<float> temp = calTensor[offset * TEMPD4_OFFSET];
    LocalTensor<float> tempSum = calTensor[offset * TEMPSUM_OFFSET];
    LocalTensor<float> pSHXYZ = calTensor[offset * PSHXYZ_OFFSET];
    float fTmp2BZ = L4_M3_SH_PARAM;

    // fTmp0D
    Muls(fTmp0D, z2, L4_M1_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(fTmp0D, fTmp0D, L4_M1_SH_PARAM_2, element);
    pipe_barrier(PIPE_V);
    Mul(fTmp0D, fTmp0D, z, element);

    // fTmp1C
    Muls(fTmp1C, z2, L4_M2_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(fTmp1C, fTmp1C, L4_M2_SH_PARAM_2, element);

    // fTmp2B
    Muls(fTmp2B, z, L4_M3_SH_PARAM, element);

    // fC3
    Mul(fC3, x, fC2, element);
    Mul(temp, y, fS2, element);
    pipe_barrier(PIPE_V);
    Sub(fC3, fC3, temp, element);
    pipe_barrier(PIPE_V);

    // fS3
    Mul(fS3, x, fS2, element);
    Mul(temp, y, fC2, element);
    pipe_barrier(PIPE_V);
    Add(fS3, fS3, temp, element);
    pipe_barrier(PIPE_V);

    // pSH16
    Muls(vCoeffs[offset * PSH16_OFFSET], fS3, L4_M4_SH_PARAM, element);

    // pSH17
    Mul(vCoeffs[offset * PSH17_OFFSET], fTmp2B, fS2, element);
    // pSH18
    Mul(vCoeffs[offset * PSH18_OFFSET], fTmp1C, fS1, element);
    // pSH19
    Mul(vCoeffs[offset * PSH19_OFFSET], fTmp0D, y, element);
    // pSH20
    Mul(vCoeffs[offset * PSH20_OFFSET], z, pSH12, element);
    Muls(temp, pSH6, L4_M0_SH_PARAM_2, element);
    pipe_barrier(PIPE_V);
    Muls(vCoeffs[offset * PSH20_OFFSET], vCoeffs[offset * PSH20_OFFSET], L4_M0_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Add(vCoeffs[offset * PSH20_OFFSET], vCoeffs[offset * PSH20_OFFSET], temp, element);
    // pSH21
    Mul(vCoeffs[offset * PSH21_OFFSET], fTmp0D, x, element);
    // pSH22
    Mul(vCoeffs[offset * PSH22_OFFSET], fTmp1C, fC1, element);
    // pSH23
    Mul(vCoeffs[offset * PSH23_OFFSET], fTmp2B, fC2, element);
    // pSH24
    Muls(vCoeffs[offset * PSH24_OFFSET], fC3, L4_M4_SH_PARAM, element);
    pipe_barrier(PIPE_V);

    Mul(vCoeffs[offset * (PSH16_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH16_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH16_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH16_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH17_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH17_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH17_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH17_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH18_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH18_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH18_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH18_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH19_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH19_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH19_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH19_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH20_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH20_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH20_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH20_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH21_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH21_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH21_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH21_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH22_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH22_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH22_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH22_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH23_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH23_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH23_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH23_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH24_OFFSET + Y_OFFSET)], vCoeffs[offset * PSH24_OFFSET], vColorsLocal[offset * Y_OFFSET],
        element);
    Mul(vCoeffs[offset * (PSH24_OFFSET + Z_OFFSET)], vCoeffs[offset * PSH24_OFFSET], vColorsLocal[offset * Z_OFFSET],
        element);

    pipe_barrier(PIPE_V);
    Mul(vCoeffs[offset * PSH16_OFFSET], vCoeffs[offset * PSH16_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH17_OFFSET], vCoeffs[offset * PSH17_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH18_OFFSET], vCoeffs[offset * PSH18_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH19_OFFSET], vCoeffs[offset * PSH19_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH20_OFFSET], vCoeffs[offset * PSH20_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH21_OFFSET], vCoeffs[offset * PSH21_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH22_OFFSET], vCoeffs[offset * PSH22_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH23_OFFSET], vCoeffs[offset * PSH23_OFFSET], vColorsLocal[offset * X_OFFSET], element);
    Mul(vCoeffs[offset * PSH24_OFFSET], vCoeffs[offset * PSH24_OFFSET], vColorsLocal[offset * X_OFFSET], element);

    // fTmp0DZ fTmp1CZ
    Muls(fTmp0DZ, z2, 3.0f * L4_M1_SH_PARAM_1, element);
    Muls(fTmp1CZ, z, 2.0f * L4_M2_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Adds(fTmp0DZ, fTmp0DZ, L4_M1_SH_PARAM_2, element);

    // fC3X
    Mul(fC3X, x, fC2X, element);
    Mul(temp, y, fS2X, element);
    pipe_barrier(PIPE_V);
    Sub(fC3X, fC3X, temp, element);
    pipe_barrier(PIPE_V);
    Add(fC3X, fC3X, fC2, element);

    // fC3Y
    Mul(fC3Y, x, fC2Y, element);
    Mul(temp, y, fS2Y, element);
    pipe_barrier(PIPE_V);
    Sub(fC3Y, fC3Y, temp, element);
    pipe_barrier(PIPE_V);
    Sub(fC3Y, fC3Y, fS2, element);

    // fS3X
    Mul(fS3X, x, fS2X, element);
    Mul(temp, y, fC2X, element);
    pipe_barrier(PIPE_V);
    Add(fS3X, fS3X, temp, element);
    pipe_barrier(PIPE_V);
    Add(fS3X, fS3X, fS2, element);

    // fS3Y
    Mul(fS3Y, x, fS2Y, element);
    Mul(temp, y, fC2Y, element);
    pipe_barrier(PIPE_V);
    Add(fS3Y, fS3Y, temp, element);
    pipe_barrier(PIPE_V);
    Add(fS3Y, fS3Y, fC2, element);

    pipe_barrier(PIPE_V);

    // pSH16X * coeffsLocal
    Muls(pSHXYZ, fS3X, L4_M4_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH16_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH16_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH16_OFFSET + Z_OFFSET)], element);

    // pSH24X * coeffsLocal
    Muls(pSHXYZ, fC3X, L4_M4_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH24_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH24_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH24_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH17X * coeffsLocal
    Mul(pSHXYZ, fTmp2B, fS2X, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH23X * coeffsLocal
    Mul(pSHXYZ, fTmp2B, fC2X, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH18X * coeffsLocal
    Mul(pSHXYZ, fTmp1C, fS1X, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH22X * coeffsLocal
    Mul(pSHXYZ, fTmp1C, fC1X, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH21X * coeffsLocal
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], fTmp0D, coeffsLocal[offset * (PSH21_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], fTmp0D, coeffsLocal[offset * (PSH21_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], fTmp0D, coeffsLocal[offset * (PSH21_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vX + reduceSum
    Add(vX, vX, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vX, vX, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    // pSH16Y * coeffsLocal
    Muls(pSHXYZ, fS3Y, L4_M4_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH16_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH16_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH16_OFFSET + Z_OFFSET)], element);

    // pSH24Y * coeffsLocal
    Muls(pSHXYZ, fC3Y, L4_M4_SH_PARAM, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH24_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH24_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH24_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH17Y * coeffsLocal
    Mul(pSHXYZ, fTmp2B, fS2Y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH23Y * coeffsLocal
    Mul(pSHXYZ, fTmp2B, fC2Y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH18Y * coeffsLocal
    Mul(pSHXYZ, fTmp1C, fS1Y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH22Y * coeffsLocal
    Mul(pSHXYZ, fTmp1C, fC1Y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH19Y * coeffsLocal
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], fTmp0D, coeffsLocal[offset * (PSH19_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], fTmp0D, coeffsLocal[offset * (PSH19_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], fTmp0D, coeffsLocal[offset * (PSH19_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vY + reduceSum
    Add(vY, vY, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vY, vY, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    // pSH20Z * coeffsLocal
    Muls(pSHXYZ, pSH6Z, L4_M0_SH_PARAM_2, element);
    Mul(temp, z, pSH12Z, element);
    pipe_barrier(PIPE_V);
    Add(temp, temp, pSH12, element);
    pipe_barrier(PIPE_V);
    Muls(temp, temp, L4_M0_SH_PARAM_1, element);
    pipe_barrier(PIPE_V);
    Add(pSHXYZ, pSHXYZ, temp, element);
    pipe_barrier(PIPE_V);
    Mul(tempSum[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH20_OFFSET + X_OFFSET)], element);
    Mul(tempSum[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH20_OFFSET + Y_OFFSET)], element);
    Mul(tempSum[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH20_OFFSET + Z_OFFSET)], element);

    // pSH21Z * coeffsLocal
    Mul(pSHXYZ, fTmp0DZ, x, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH21_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH21_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH21_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH19Z * coeffsLocal
    Mul(pSHXYZ, fTmp0DZ, y, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH19_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH19_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH19_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH22Z * coeffsLocal
    Mul(pSHXYZ, fTmp1CZ, fC1, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH22_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH18Z * coeffsLocal
    Mul(pSHXYZ, fTmp1CZ, fS1, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH18_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH23Z * coeffsLocal
    Muls(pSHXYZ, fC2, fTmp2BZ, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH23_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);

    // pSH17Z * coeffsLocal
    Muls(pSHXYZ, fS2, fTmp2BZ, element);
    pipe_barrier(PIPE_V);
    Mul(temp[offset * X_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + X_OFFSET)], element);
    Mul(temp[offset * Y_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + Y_OFFSET)], element);
    Mul(temp[offset * Z_OFFSET], pSHXYZ, coeffsLocal[offset * (PSH17_OFFSET + Z_OFFSET)], element);
    pipe_barrier(PIPE_V);
    Add(tempSum, tempSum, temp, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    // vColors * tempSum
    Mul(tempSum, tempSum, vColorsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
    // vZ + reduceSum
    Add(vZ, vZ, tempSum[offset * X_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tempSum[offset * Y_OFFSET], element);
    pipe_barrier(PIPE_V);
    Add(vZ, vZ, tempSum[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);
}

__aicore__ inline void SphericalHarmonicsBwd::procRet(LocalTensor<float> &vDirs, LocalTensor<float> &vX,
                                                      LocalTensor<float> &vY, LocalTensor<float> &vZ,
                                                      int32_t element)
{
    int64_t offset = Align(element, FLOAT_SIZE);
    LocalTensor<float> temp = calBuf_.Get<float>();
    LocalTensor<float> vDirN = vX;
    Mul(temp, vDirN, dirsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    /* ReduceSum */
    Add(temp, temp, temp[offset], element);
    pipe_barrier(PIPE_V);
    Add(temp, temp, temp[offset * Z_OFFSET], element);
    pipe_barrier(PIPE_V);

    Mul(dirsLocal, dirsLocal, temp, element);
    Mul(dirsLocal[offset], dirsLocal[offset], temp, element);
    Mul(dirsLocal[offset * Z_OFFSET], dirsLocal[offset * Z_OFFSET], temp, element);
    pipe_barrier(PIPE_V);

    Sub(vDirN, vDirN, dirsLocal, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);

    Div(vDirN, vDirN, inorm, element);
    Div(vDirN[offset], vDirN[offset], inorm, element);
    Div(vDirN[offset * Z_OFFSET], vDirN[offset * Z_OFFSET], inorm, element);
    pipe_barrier(PIPE_V);

    Add(vDirs, vDirs, vDirN, offset * CONSTDIM_THREE);
    pipe_barrier(PIPE_V);
}

} // namespace SphericalHarmonicsBwdNs
#endif // SPHERICAL_HARMONICS_BWD_H