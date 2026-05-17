/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPHERICAL_HARMONICS_BWD_COMMON_H
#define SPHERICAL_HARMONICS_BWD_COMMON_H

#include "kernel_operator.h"

namespace SphericalHarmonicsBwdNs {
using namespace AscendC;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;
constexpr int64_t INT64_ONE_BLOCK_NUM = 4;
constexpr int64_t FLOAT_SIZE = 4;

constexpr int64_t X_OFFSET = 0;
constexpr int64_t Y_OFFSET = 1;
constexpr int64_t Z_OFFSET = 2;
constexpr int64_t DEGREE_ZERO = 0;
constexpr int64_t DEGREE_ONE = 1;
constexpr int64_t DEGREE_TWO = 2;
constexpr int64_t DEGREE_THREE = 3;
constexpr int64_t DEGREE_FOUR = 4;
constexpr int64_t Z2_OFFSET = 0;
constexpr int64_t INORM_OFFSET = 1;
constexpr int64_t VX_OFFSET = 2;
constexpr int64_t VY_OFFSET = 3;
constexpr int64_t VZ_OFFSET = 4;
constexpr int64_t VDIRS_OFFSET = 3;
constexpr int64_t FC1_OFFSET = 5;
constexpr int64_t FS1_OFFSET = 6;
constexpr int64_t FC1X_OFFSET = 7;
constexpr int64_t FS1X_OFFSET = 8;
constexpr int64_t FC1Y_OFFSET = 9;
constexpr int64_t FS1Y_OFFSET = 10;
constexpr int64_t PSH6BUF_OFFSET = 11;
constexpr int64_t PSH6Z_OFFSET = 12;
constexpr int64_t FC2_OFFSET = 13;
constexpr int64_t FS2_OFFSET = 14;
constexpr int64_t FC2X_OFFSET = 15;
constexpr int64_t FS2X_OFFSET = 16;
constexpr int64_t FC2Y_OFFSET = 17;
constexpr int64_t FS2Y_OFFSET = 18;
constexpr int64_t PSH12BUF_OFFSET = 19;
constexpr int64_t PSH12Z_OFFSET = 20;
constexpr int64_t FC3_OFFSET = 21;
constexpr int64_t FS3_OFFSET = 22;
constexpr int64_t FC3X_OFFSET = 23;
constexpr int64_t FS3X_OFFSET = 24;
constexpr int64_t FC3Y_OFFSET = 25;
constexpr int64_t FS3Y_OFFSET = 26;
constexpr int64_t CONSTDIM_THREE = 3;
constexpr int64_t CALBUF_SIZE = 12;
constexpr int64_t TMPX_OFFSET = 0;
constexpr int64_t TMPY_OFFSET = 1;
constexpr int64_t TMPZ_OFFSET = 2;
constexpr int64_t TMPVDIR_OFFSET = 3;
constexpr int64_t TMPX2_OFFSET = 4;
constexpr int64_t TMPY2_OFFSET = 5;
constexpr int64_t TMPZ2_OFFSET = 6;
constexpr int64_t PSH1_OFFSET = 3;
constexpr int64_t PSH2_OFFSET = 6;
constexpr int64_t PSH3_OFFSET = 9;
constexpr int64_t PSH4_OFFSET = 12;
constexpr int64_t PSH5_OFFSET = 15;
constexpr int64_t PSH6_OFFSET = 18;
constexpr int64_t PSH7_OFFSET = 21;
constexpr int64_t PSH8_OFFSET = 24;
constexpr int64_t PSH9_OFFSET = 27;
constexpr int64_t PSH10_OFFSET = 30;
constexpr int64_t PSH11_OFFSET = 33;
constexpr int64_t PSH12_OFFSET = 36;
constexpr int64_t PSH13_OFFSET = 39;
constexpr int64_t PSH14_OFFSET = 42;
constexpr int64_t PSH15_OFFSET = 45;
constexpr int64_t PSH16_OFFSET = 48;
constexpr int64_t PSH17_OFFSET = 51;
constexpr int64_t PSH18_OFFSET = 54;
constexpr int64_t PSH19_OFFSET = 57;
constexpr int64_t PSH20_OFFSET = 60;
constexpr int64_t PSH21_OFFSET = 63;
constexpr int64_t PSH22_OFFSET = 66;
constexpr int64_t PSH23_OFFSET = 69;
constexpr int64_t PSH24_OFFSET = 72;
constexpr int64_t TMPSUMD2_OFFSET = 4;
constexpr int64_t PSHXYZD2_OFFSET = 7;
constexpr int64_t FTMP0C_OFFSET = 0;
constexpr int64_t FTMP1B_OFFSET = 1;
constexpr int64_t FTMP0CZ_OFFSET = 2;
constexpr int64_t TEMPD3_OFFSET = 3;
constexpr int64_t TEMPSUMD3_OFFSET = 6;
constexpr int64_t PSHXYZD3_OFFSET = 9;
constexpr int64_t FTMP0D_OFFSET = 0;
constexpr int64_t FTMP1C_OFFSET = 1;
constexpr int64_t FTMP2B_OFFSET = 2;
constexpr int64_t FTMP0DZ_OFFSET = 3;
constexpr int64_t FTMP1CZ_OFFSET = 4;
constexpr int64_t TEMPD4_OFFSET = 5;
constexpr int64_t TEMPSUM_OFFSET = 8;
constexpr int64_t PSHXYZ_OFFSET = 11;

constexpr float L0_M0_SH_PARAM = 0.2820947917738781f;
constexpr float L1_M0_SH_PARAM = 0.48860251190292f;
constexpr float L2_M0_SH_PARAM_1 = 0.9461746957575601f;
constexpr float L2_M0_SH_PARAM_2 = -0.3153915652525201f;
constexpr float L2_M1_SH_PARAM = -1.092548430592079f;
constexpr float L2_M2_SH_PARAM = 0.5462742152960395f;
constexpr float L3_M0_SH_PARAM_1 = 1.865881662950577f;
constexpr float L3_M0_SH_PARAM_2 = -1.119528997770346f;
constexpr float L3_M1_SH_PARAM_1 = -2.285228997322329f;
constexpr float L3_M1_SH_PARAM_2 = 0.4570457994644658f;
constexpr float L3_M2_SH_PARAM = 1.445305721320277f;
constexpr float L3_M3_SH_PARAM = -0.5900435899266435f;
constexpr float L4_M0_SH_PARAM_1 = 1.984313483298443f;
constexpr float L4_M0_SH_PARAM_2 = -1.006230589874905f;
constexpr float L4_M1_SH_PARAM_1 = -4.683325804901025f;
constexpr float L4_M1_SH_PARAM_2 = 2.007139630671868f;
constexpr float L4_M2_SH_PARAM_1 = 3.31161143515146f;
constexpr float L4_M2_SH_PARAM_2 = -0.47308734787878f;
constexpr float L4_M3_SH_PARAM = -1.770130769779931f;
constexpr float L4_M4_SH_PARAM = 0.6258357354491763f;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

template <typename T> __aicore__ inline T Min(T a, T b) { return a > b ? b : a; }

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <HardEvent event> __aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace SphericalHarmonicsBwdNs
#endif // SPHERICAL_HARMONICS_BWD_COMMON_H