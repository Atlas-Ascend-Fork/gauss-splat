/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPHERICAL_HARMONICS_BWD_H
#define SPHERICAL_HARMONICS_BWD_H

#include "kernel_operator.h"
#include "spherical_harmonics_bwd_common.h"

namespace SphericalHarmonicsBwdNs {
using namespace AscendC;

/*
 * Emit one SH coefficient gradient and accumulate its direction gradient.
 * All operands stay in RegTensor objects. Only the completed vCoeff row is
 * stored to UB; the accumulated vX/vY/vZ registers are stored once after all
 * active degrees for the channel have finished.
*/

__simd_callee__ inline void EmitShGradient(
    __ubuf__ float* vCoeffs, __ubuf__ float* coeffs, uint32_t ubOffset,
    Reg::RegTensor<float> &basis, Reg::RegTensor<float> &basisDx,
    Reg::RegTensor<float> &basisDy, Reg::RegTensor<float> &basisDz,
    Reg::RegTensor<float> &vColor, Reg::RegTensor<float> &vX,
    Reg::RegTensor<float> &vY, Reg::RegTensor<float> &vZ, Reg::MaskReg &mask) 
    {
        Reg::RegTensor<float> vCoeff;
        Reg::RegTensor<float> coeff;
        Reg::RegTensor<float> weight;
        Reg::RegTensor<float> contribution;

        Reg::Mul(vCoeff, basis, vColor, mask);
        Reg::StoreAlign(vCoeffs + ubOffset, vCoeff, mask);

        Reg::LoadAlign(coeff, coeffs + ubOffset);
        Reg::Mul(weight, coeff, vColor, mask);
        Reg::Mul(contribution, weight, basisDx, mask);
        Reg::Add(vX, vX, contribution, mask);
        Reg::Mul(contribution, weight, basisDy, mask);
        Reg::Add(vY, vY, contribution, mask);
        Reg::Mul(contribution, weight, basisDz, mask);
        Reg::Add(vZ, vZ, contribution, mask);
    }

/*
 * Each degree has its own VF and processes at most one vector (64 floats).
 * All arithmetic within a degree stays in registers. Across degrees and
 * channels, vDirsAccum keeps the accumulated normalized-direction gradient;
 * its fourth row retains the norm for the final Jacobian projection.
 * Repeat/channel/degree control flow remains in the aicore caller.
*/
__simd_vf__ inline void InitDirectionVF(
    __ubuf__ float* dirs, __ubuf__ float* vDirsAccum,
    uint32_t count, uint32_t rowStride, uint32_t pos)
    {
        uint32_t remain = count;
        Reg::MaskReg mask = Reg::UpdateMask<float>(remain);

        Reg::RegTensor<float> x;
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> z;
        Reg::RegTensor<float> vColor;
        Reg::RegTensor<float> vX;
        Reg::RegTensor<float> vY;
        Reg::RegTensor<float> vZ;
        Reg::RegTensor<float> basis;
        Reg::RegTensor<float> basisDx;
        Reg::RegTensor<float> basisDy;
        Reg::RegTensor<float> basisDz;
        Reg::RegTensor<float> t0;
        Reg::RegTensor<float> t1;
        Reg::RegTensor<float> t2;
        Reg::RegTensor<float> t3;
        Reg::RegTensor<float> x2;
        Reg::RegTensor<float> y2;
        Reg::RegTensor<float> z2;

        Reg::LoadAlign(x, dirs + pos);
        Reg::LoadAlign(y, dirs + rowStride + pos);
        Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
        // Normalize the common direction once and retain its norm for the 
        // final Jacobian projection after every channel has accumulated.
        Reg::Mul(x2, x, x, mask);
        Reg::Mul(y2, y, y, mask);
        Reg::Mul(z2, z, z, mask);
        Reg::Add(t0, x2, y2, mask);
        Reg::Add(t0, t0, z2, mask);
        Reg::Sqrt(t0, t0, mask);
        Reg::Maxs(t0, t0, 1e-12f, mask);
        Reg::Div(x, x, t0, mask);
        Reg::Div(y, y, t0, mask);
        Reg::Div(z, z, t0, mask);
        Reg::StoreAlign(dirs + pos, x, mask);
        Reg::StoreAlign(dirs + rowStride + pos, y, mask);
        Reg::StoreAlign(dirs + 2 * rowStride + pos, z, mask);
        Reg::StoreAlign(vDirsAccum + 3 * rowStride + pos, t0, mask);
        Reg::Duplicate(vX, 0.0f, mask);
        Reg::Duplicate(vY, 0.0f, mask);
        Reg::Duplicate(vZ, 0.0f, mask);

        Reg::StoreAlign(vDirsAccum + pos, vX, mask);
        Reg::StoreAlign(vDirsAccum + rowStride + pos, vY, mask);
        Reg::StoreAlign(vDirsAccum + 2 * rowStride + pos, vZ, mask);
    }   

// Degree 0: coefficient rows 0..0.
__simd_vf__ inline void ProcD0VF(
    __ubuf__ float* dirs, __ubuf__ float* coeffs, __ubuf__ float* vColors,
    __ubuf__ float* vDirsAccum, __ubuf__ float* vCoeffs,
    uint32_t count, uint32_t rowStride, uint32_t pos)
    {
        uint32_t remain = count;
        Reg::MaskReg mask = Reg::UpdateMask<float>(remain);

        Reg::RegTensor<float> x;
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> z;
        Reg::RegTensor<float> vColor;
        Reg::RegTensor<float> vX;
        Reg::RegTensor<float> vY;
        Reg::RegTensor<float> vZ;
        Reg::RegTensor<float> basis;
        Reg::RegTensor<float> basisDx;
        Reg::RegTensor<float> basisDy;
        Reg::RegTensor<float> basisDz;
        Reg::RegTensor<float> t0;
        Reg::RegTensor<float> t1;
        Reg::RegTensor<float> t2;
        Reg::RegTensor<float> t3;
        Reg::RegTensor<float> x2;
        Reg::RegTensor<float> y2;
        Reg::RegTensor<float> z2;

        Reg::LoadAlign(x, dirs + pos);
        Reg::LoadAlign(y, dirs + rowStride + pos);
        Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
        Reg::LoadAlign(vColor, vColors + pos);
        Reg::LoadAlign(vX, vDirsAccum + pos);
        Reg::LoadAlign(vY, vDirsAccum + rowStride + pos);
        Reg::LoadAlign(vZ, vDirsAccum + 2 * rowStride + pos);

        // l=0, m=0
        Reg::Duplicate(basis, L0_M0_SH_PARAM, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        Reg::StoreAlign(vDirsAccum + pos, vX, mask);
        Reg::StoreAlign(vDirsAccum + rowStride + pos, vY, mask);
        Reg::StoreAlign(vDirsAccum + 2 * rowStride + pos, vZ, mask);
    }

// Degree 1: coefficient rows 1..3.
__simd_vf__ inline void ProcD1VF(
    __ubuf__ float* dirs, __ubuf__ float* coeffs, __ubuf__ float* vColors,
    __ubuf__ float* vDirsAccum, __ubuf__ float* vCoeffs,
    uint32_t count, uint32_t rowStride, uint32_t pos) 
    {
        uint32_t remain = count;
        Reg::MaskReg mask = Reg::UpdateMask<float>(remain);

        Reg::RegTensor<float> x;
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> z;
        Reg::RegTensor<float> vColor;
        Reg::RegTensor<float> vX;
        Reg::RegTensor<float> vY;
        Reg::RegTensor<float> vZ;
        Reg::RegTensor<float> basis;
        Reg::RegTensor<float> basisDx;
        Reg::RegTensor<float> basisDy;
        Reg::RegTensor<float> basisDz;
        Reg::RegTensor<float> t0;
        Reg::RegTensor<float> t1;
        Reg::RegTensor<float> t2;
        Reg::RegTensor<float> t3;
        Reg::RegTensor<float> x2;
        Reg::RegTensor<float> y2;
        Reg::RegTensor<float> z2;

        Reg::LoadAlign(x, dirs + pos);
        Reg::LoadAlign(y, dirs + rowStride + pos);
        Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
        Reg::LoadAlign(vColor, vColors + pos);
        Reg::LoadAlign(vX, vDirsAccum + pos);
        Reg::LoadAlign(vY, vDirsAccum + rowStride + pos);
        Reg::LoadAlign(vZ, vDirsAccum + 2 * rowStride + pos);

        // k=1: -C1*y
        Reg::Muls(basis, y, -L1_M0_SH_PARAM, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Duplicate(basisDy, -L1_M0_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=2: C1*z
        Reg::Muls(basis, z, L1_M0_SH_PARAM, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Duplicate(basisDz, L1_M0_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 2 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=3: -C1*x
        Reg::Muls(basis, x, -L1_M0_SH_PARAM, mask);
        Reg::Duplicate(basisDx, -L1_M0_SH_PARAM, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 3 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        Reg::StoreAlign(vDirsAccum + pos, vX, mask);
        Reg::StoreAlign(vDirsAccum + rowStride + pos, vY, mask);
        Reg::StoreAlign(vDirsAccum + 2 * rowStride + pos, vZ, mask);
    }

// Degree 2: coefficient rows 4..8.
__simd_vf__ inline void ProcD2VF(
    __ubuf__ float* dirs, __ubuf__ float* coeffs, __ubuf__ float* vColors,
    __ubuf__ float* vDirsAccum, __ubuf__ float* vCoeffs,
    uint32_t count, uint32_t rowStride, uint32_t pos)
    {
        uint32_t remain = count;
        Reg::MaskReg mask = Reg::UpdateMask<float>(remain);

        Reg::RegTensor<float> x;
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> z;
        Reg::RegTensor<float> vColor;
        Reg::RegTensor<float> vX;
        Reg::RegTensor<float> vY;
        Reg::RegTensor<float> vZ;
        Reg::RegTensor<float> basis;
        Reg::RegTensor<float> basisDx;
        Reg::RegTensor<float> basisDy;
        Reg::RegTensor<float> basisDz;
        Reg::RegTensor<float> t0;
        Reg::RegTensor<float> t1;
        Reg::RegTensor<float> t2;
        Reg::RegTensor<float> t3;
        Reg::RegTensor<float> x2;
        Reg::RegTensor<float> y2;
        Reg::RegTensor<float> z2;

        Reg::LoadAlign(x, dirs + pos);
        Reg::LoadAlign(y, dirs + rowStride + pos);
        Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
        Reg::LoadAlign(vColor, vColors + pos);
        Reg::LoadAlign(vX, vDirsAccum + pos);
        Reg::LoadAlign(vY, vDirsAccum + rowStride + pos);
        Reg::LoadAlign(vZ, vDirsAccum + 2 * rowStride + pos);

        Reg::Mul(x2, x, x, mask);
        Reg::Mul(y2, y, y, mask);
        Reg::Mul(z2, z, z, mask);

        // k=4, 2*C22*x*y
        Reg::Mul(basis, x, y, mask);
        Reg::Muls(basis, basis, 2.0f * L2_M2_SH_PARAM, mask);
        Reg::Muls(basisDx, y, 2.0f * L2_M2_SH_PARAM, mask);
        Reg::Muls(basisDy, x, 2.0f * L2_M2_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 4 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=5, C21*y*z
        Reg::Mul(basis, y, z, mask);
        Reg::Muls(basis, basis, L2_M1_SH_PARAM, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Muls(basisDy, z, L2_M1_SH_PARAM, mask);
        Reg::Muls(basisDz, y, L2_M1_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 5 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=6, C20a*z^2 + C20b
        Reg::Muls(basis, z2, L2_M0_SH_PARAM_1, mask);
        Reg::Adds(basis, basis, L2_M0_SH_PARAM_2, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Muls(basisDz, z, 2.0f * L2_M0_SH_PARAM_1, mask);
        EmitShGradient(vCoeffs, coeffs, 6 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=7, C21*x*z
        Reg::Mul(basis, x, z, mask);
        Reg::Muls(basis, basis, L2_M1_SH_PARAM, mask);
        Reg::Muls(basisDx, z, L2_M1_SH_PARAM, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Muls(basisDz, x, L2_M1_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 7 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=8, C22*(x^2-y^2)
        Reg::Sub(t0, x2, y2, mask);
        Reg::Muls(basis, t0, L2_M2_SH_PARAM, mask);
        Reg::Muls(basisDx, x, 2.0f * L2_M2_SH_PARAM, mask);
        Reg::Muls(basisDy, y, -2.0f * L2_M2_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 8 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        Reg::StoreAlign(vDirsAccum + pos, vX, mask);
        Reg::StoreAlign(vDirsAccum + rowStride + pos, vY, mask);
        Reg::StoreAlign(vDirsAccum + 2 * rowStride + pos, vZ, mask);
    }

// Degree 3: coefficient rows 9..15.
__simd_vf__ inline void ProcD3VF(
    __ubuf__ float* dirs, __ubuf__ float* coeffs, __ubuf__ float* vColors,
    __ubuf__ float* vDirsAccum, __ubuf__ float* vCoeffs,
    uint32_t count, uint32_t rowStride, uint32_t pos)
    {
        uint32_t remain = count;
        Reg::MaskReg mask = Reg::UpdateMask<float>(remain);

        Reg::RegTensor<float> x;
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> z;
        Reg::RegTensor<float> vColor;
        Reg::RegTensor<float> vX;
        Reg::RegTensor<float> vY;
        Reg::RegTensor<float> vZ;
        Reg::RegTensor<float> basis;
        Reg::RegTensor<float> basisDx;
        Reg::RegTensor<float> basisDy;
        Reg::RegTensor<float> basisDz;
        Reg::RegTensor<float> t0;
        Reg::RegTensor<float> t1;
        Reg::RegTensor<float> t2;
        Reg::RegTensor<float> t3;
        Reg::RegTensor<float> x2;
        Reg::RegTensor<float> y2;
        Reg::RegTensor<float> z2;

        Reg::LoadAlign(x, dirs + pos);
        Reg::LoadAlign(y, dirs + rowStride + pos);
        Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
        Reg::LoadAlign(vColor, vColors + pos);
        Reg::LoadAlign(vX, vDirsAccum + pos);
        Reg::LoadAlign(vY, vDirsAccum + rowStride + pos);
        Reg::LoadAlign(vZ, vDirsAccum + 2 * rowStride + pos);
        
        // Recreate x^2/y^2/z^2 because higher-degree paths may run without
        // relying on register liveness from a previous scalar branch 
        Reg::Mul(x2, x, x, mask);
        Reg::Mul(y2, y, y, mask);
        Reg::Mul(z2, z, z, mask);

        // k=9: C33*(3*x^2*y-y^3)
        Reg::Muls(t0, x2, 3.0f, mask);
        Reg::Sub(t0, t0, y2, mask);
        Reg::Mul(t0, t0, y, mask);
        Reg::Muls(basis, t0, L3_M3_SH_PARAM, mask);
        Reg::Mul(basisDx, x, y, mask);
        Reg::Muls(basisDx, basisDx, 6.0f * L3_M3_SH_PARAM, mask);
        Reg::Sub(t1, x2, y2, mask);
        Reg::Muls(basisDy, t1, 3.0f * L3_M3_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 9 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=10: C32*z*(2*x*y)
        Reg::Mul(t0, x, y, mask);
        Reg::Muls(t0, t0, 2.0f, mask);
        Reg::Mul(basis, t0, z, mask);
        Reg::Muls(basis, basis, L3_M2_SH_PARAM, mask);
        Reg::Mul(basisDx, y, z, mask);
        Reg::Muls(basisDx, basisDx, 2.0f * L3_M2_SH_PARAM, mask);
        Reg::Mul(basisDy, x, z, mask);
        Reg::Muls(basisDy, basisDy, 2.0f * L3_M2_SH_PARAM, mask);
        Reg::Muls(basisDz, t0, L3_M2_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 10 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // Shared C31(z) = a*z^2+b.
        Reg::Muls(t0, z2, L3_M1_SH_PARAM_1, mask);
        Reg::Adds(t0, t0, L3_M1_SH_PARAM_2, mask);
        Reg::Muls(t1, z, 2.0f * L3_M1_SH_PARAM_1, mask);

        // k=11: C31(z)*y
        Reg::Mul(basis, t0, y, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Adds(basisDy, t0, 0.0f, mask);
        Reg::Mul(basisDz, t1, y, mask);
        EmitShGradient(vCoeffs, coeffs, 11 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=12: (C30a*z^2+C30b)*z
        Reg::Muls(t2, z2, L3_M0_SH_PARAM_1, mask);
        Reg::Adds(t2, t2, L3_M0_SH_PARAM_2, mask);
        Reg::Mul(basis, t2, z, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Muls(basisDz, z2, 3.0f * L3_M0_SH_PARAM_1, mask);
        Reg::Adds(basisDz, basisDz, L3_M0_SH_PARAM_2, mask);
        EmitShGradient(vCoeffs, coeffs, 12 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=13: C31(z)*x
        Reg::Mul(basis, t0, x, mask);
        Reg::Adds(basisDx, t0, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Mul(basisDz, t1, x, mask);
        EmitShGradient(vCoeffs, coeffs, 13 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=14: C32*z*(x^2-y^2)
        Reg::Sub(t0, x2, y2, mask);
        Reg::Mul(basis, t0, z, mask);
        Reg::Muls(basis, basis, L3_M2_SH_PARAM, mask);
        Reg::Mul(basisDx, x, z, mask);
        Reg::Muls(basisDx, basisDx, 2.0f * L3_M2_SH_PARAM, mask);
        Reg::Mul(basisDy, y, z, mask);
        Reg::Muls(basisDy, basisDy, -2.0f * L3_M2_SH_PARAM, mask);
        Reg::Muls(basisDz, t0, L3_M2_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 14 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // k=15: C33*(x^3-3*x*y^2)
        Reg::Muls(t0, y2, 3.0f, mask);
        Reg::Sub(t0, x2, t0, mask);
        Reg::Mul(t0, t0, x, mask);
        Reg::Muls(basis, t0, L3_M3_SH_PARAM, mask);
        Reg::Sub(t1, x2, y2, mask);
        Reg::Muls(basisDx, t1, 3.0f * L3_M3_SH_PARAM, mask);
        Reg::Mul(basisDy, x, y, mask);
        Reg::Muls(basisDy, basisDy, -6.0f * L3_M3_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 15 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        Reg::StoreAlign(vDirsAccum + pos, vX, mask);
        Reg::StoreAlign(vDirsAccum + rowStride + pos, vY, mask);
        Reg::StoreAlign(vDirsAccum + 2 * rowStride + pos, vZ, mask);
    }

// Degree 4: coefficient rows 16..24.
__simd_vf__ inline void ProcD4VF(
    __ubuf__ float* dirs, __ubuf__ float* coeffs, __ubuf__ float* vColors,
    __ubuf__ float* vDirsAccum, __ubuf__ float* vCoeffs,
    uint32_t count, uint32_t rowStride, uint32_t pos)
    {
        uint32_t remain = count;
        Reg::MaskReg mask = Reg::UpdateMask<float>(remain);

        Reg::RegTensor<float> x;
        Reg::RegTensor<float> y;
        Reg::RegTensor<float> z;
        Reg::RegTensor<float> vColor;
        Reg::RegTensor<float> vX;
        Reg::RegTensor<float> vY;
        Reg::RegTensor<float> vZ;
        Reg::RegTensor<float> basis;
        Reg::RegTensor<float> basisDx;
        Reg::RegTensor<float> basisDy;
        Reg::RegTensor<float> basisDz;
        Reg::RegTensor<float> t0;
        Reg::RegTensor<float> t1;
        Reg::RegTensor<float> t2;
        Reg::RegTensor<float> t3;
        Reg::RegTensor<float> x2;
        Reg::RegTensor<float> y2;
        Reg::RegTensor<float> z2;

        Reg::LoadAlign(x, dirs + pos);
        Reg::LoadAlign(y, dirs + rowStride + pos);
        Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
        Reg::LoadAlign(vColor, vColors + pos);
        Reg::LoadAlign(vX, vDirsAccum + pos);
        Reg::LoadAlign(vY, vDirsAccum + rowStride + pos);
        Reg::LoadAlign(vZ, vDirsAccum + 2 * rowStride + pos);
        
        Reg::Mul(x2, x, x, mask);
        Reg::Mul(y2, y, y, mask);
        Reg::Mul(z2, z, z, mask);

        // k=16: C44*(4*x^3*y-4*x*y^3)
        Reg::Sub(t0, x2, y2, mask);
        Reg::Mul(t0, t0, x, mask);
        Reg::Mul(t0, t0, y, mask);
        Reg::Muls(basis, t0, 4.0f * L4_M4_SH_PARAM, mask);
        Reg::Muls(t1, x2, 3.0f, mask);
        Reg::Sub(t1, t1, y2, mask);
        Reg::Mul(basisDx, t1, y, mask);
        Reg::Muls(basisDx, basisDx, 4.0f * L4_M4_SH_PARAM, mask);
        Reg::Muls(t1, y2, 3.0f, mask);
        Reg::Sub(t1, x2, t1, mask);
        Reg::Mul(basisDy, t1, x, mask);
        Reg::Muls(basisDy, basisDy, 4.0f * L4_M4_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 16 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // fS2=3*x^2*y-y^3, fC2=x^3-3*x*y^2.
        Reg::Muls(t0, x2, 3.0f, mask);
        Reg::Sub(t0, t0, y2, mask);
        Reg::Mul(t0, t0, y, mask); // fS2
        Reg::Muls(t1, y2, 3.0f, mask);
        Reg::Sub(t1, x2, t1, mask);
        Reg::Mul(t1, t1, x, mask); // fC2

        // k=17: C43*z*fS2
        Reg::Mul(basis, z, t0, mask);
        Reg::Muls(basis, basis, L4_M3_SH_PARAM, mask);
        Reg::Mul(basisDx, x, y, mask);
        Reg::Mul(basisDx, basisDx, z, mask);
        Reg::Muls(basisDx, basisDx, 6.0f * L4_M3_SH_PARAM, mask);
        Reg::Sub(t2, x2, y2, mask);
        Reg::Mul(basisDy, t2, z, mask);
        Reg::Muls(basisDy, basisDy, 3.0f * L4_M3_SH_PARAM, mask);
        Reg::Muls(basisDz, t0, L4_M3_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 17 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // Shared C42(z)=a*z^2+b and fS1=2*x*y.
        Reg::Muls(t2, z2, L4_M2_SH_PARAM_1, mask);
        Reg::Adds(t2, t2, L4_M2_SH_PARAM_2, mask);
        Reg::Mul(t3, x, y, mask);
        Reg::Muls(t3, t3, 2.0f, mask);

        // k=18: C42(z)*fS1
        Reg::Mul(basis, t2, t3, mask);
        Reg::Muls(basisDx, y, 2.0f, mask);
        Reg::Mul(basisDx, basisDx, t2, mask);
        Reg::Muls(basisDy, x, 2.0f, mask);
        Reg::Mul(basisDy, basisDy, t2, mask);
        Reg::Muls(basisDz, z, 2.0f * L4_M2_SH_PARAM_1, mask);
        Reg::Mul(basisDz, basisDz, t3, mask);
        EmitShGradient(vCoeffs, coeffs, 18 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        // Shared C41(z)=(a*z^2+b)*z and its z derivative
        Reg::Muls(t2, z2, L4_M1_SH_PARAM_1, mask);
        Reg::Adds(t2, t2, L4_M1_SH_PARAM_2, mask);
        Reg::Mul(t2, t2, z, mask);
        Reg::Muls(t3, z2, 3.0f * L4_M1_SH_PARAM_1, mask);
        Reg::Adds(t3, t3, L4_M1_SH_PARAM_2, mask);

        // k=19: C41(z)*y
        Reg::Mul(basis, t2, y, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Adds(basisDy, t2, 0.0f, mask);
        Reg::Mul(basisDz, t3, y, mask);
        EmitShGradient(vCoeffs, coeffs, 19 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);
                        
        // k=20: C40a*z*pSH12 + C40b*pSH6
        Reg::Muls(t0, z2, L3_M0_SH_PARAM_1, mask);
        Reg::Adds(t0, t0, L3_M0_SH_PARAM_2, mask);
        Reg::Mul(t0, t0, z, mask); // pSH12
        Reg::Muls(t1, z2, L2_M0_SH_PARAM_1, mask);
        Reg::Adds(t1, t1, L2_M0_SH_PARAM_2, mask); // pSH6
        Reg::Mul(basis, z, t0, mask);
        Reg::Muls(basis, basis, L4_M0_SH_PARAM_1, mask);
        Reg::Muls(t2, t1, L4_M0_SH_PARAM_2, mask);
        Reg::Add(basis, basis, t2, mask);
        Reg::Duplicate(basisDx, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Muls(basisDz, z, 2.0f * L2_M0_SH_PARAM_1, mask);
        Reg::Muls(basisDz, basisDz, L4_M0_SH_PARAM_2, mask);
        Reg::Muls(t2, z2, 3.0f * L3_M0_SH_PARAM_1, mask);
        Reg::Adds(t2, t2, L3_M0_SH_PARAM_2, mask); // pSH12Z
        Reg::Mul(t2, t2, z, mask);
        Reg::Add(t2, t2, t0, mask);
        Reg::Muls(t2, t2, L4_M0_SH_PARAM_1, mask);
        Reg::Add(basisDz, basisDz, t2, mask);
        EmitShGradient(vCoeffs, coeffs, 20 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);        

        // Recreate C41(z), because t2/t3 were reused by k=20.
        Reg::Muls(t2, z2, L4_M1_SH_PARAM_1, mask);
        Reg::Adds(t2, t2, L4_M1_SH_PARAM_2, mask);
        Reg::Mul(t2, t2, z, mask);
        Reg::Muls(t3, z2, 3.0f * L4_M1_SH_PARAM_1, mask);
        Reg::Adds(t3, t3, L4_M1_SH_PARAM_2, mask);

        // k=21: C41(z)*x
        Reg::Mul(basis, t2, x, mask);
        Reg::Adds(basisDx, t2, 0.0f, mask);
        Reg::Duplicate(basisDy, 0.0f, mask);
        Reg::Mul(basisDz, t3, x, mask);
        EmitShGradient(vCoeffs, coeffs, 21 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);
                        
        // k=22: C42(z)*(x^2-y^2)
        Reg::Muls(t2, z2, L4_M2_SH_PARAM_1, mask);
        Reg::Adds(t2, t2, L4_M2_SH_PARAM_2, mask);
        Reg::Sub(t3, x2, y2, mask);
        Reg::Mul(basis, t2, t3, mask);
        Reg::Muls(basisDx, x, 2.0f, mask);
        Reg::Mul(basisDx, basisDx, t2, mask);
        Reg::Muls(basisDy, y, -2.0f, mask);
        Reg::Mul(basisDy, basisDy, t2, mask);
        Reg::Muls(basisDz, z, 2.0f * L4_M2_SH_PARAM_1, mask);
        Reg::Mul(basisDz, basisDz, t3, mask);
        EmitShGradient(vCoeffs, coeffs, 22 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);
                        
        // k=23: C43*z*fC2
        Reg::Muls(t0, y2, 3.0f, mask);
        Reg::Sub(t0, x2, t0, mask);
        Reg::Mul(t0, t0, x, mask); // fC2
        Reg::Mul(basis, z, t0, mask);
        Reg::Muls(basis, basis, L4_M3_SH_PARAM, mask);
        Reg::Sub(t1, x2, y2, mask);
        Reg::Mul(basisDx, t1, z, mask);
        Reg::Muls(basisDx, basisDx, 3.0f * L4_M3_SH_PARAM, mask);
        Reg::Mul(basisDy, x, y, mask);
        Reg::Mul(basisDy, basisDy, z, mask);
        Reg::Muls(basisDy, basisDy, -6.0f * L4_M3_SH_PARAM, mask);
        Reg::Muls(basisDz, t0, L4_M3_SH_PARAM, mask);
        EmitShGradient(vCoeffs, coeffs, 23 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);
                        
        // k=24: C44*(x^4-6*x^2*y^2+y^4)
        Reg::Mul(t0, x2, x2, mask);
        Reg::Mul(t1, x2, y2, mask);
        Reg::Muls(t1, t1, 6.0f, mask);
        Reg::Sub(t0, t0, t1, mask);
        Reg::Mul(t1, y2, y2, mask);
        Reg::Add(t0, t0, t1, mask);
        Reg::Muls(basis, t0, L4_M4_SH_PARAM, mask);
        Reg::Muls(t0, y2, 3.0f, mask);
        Reg::Sub(t0, x2, t0, mask);
        Reg::Mul(basisDx, t0, x, mask);
        Reg::Muls(basisDx, basisDx, 4.0f * L4_M4_SH_PARAM, mask);
        Reg::Muls(t0, x2, 3.0f, mask);
        Reg::Sub(t0, y2, t0, mask);
        Reg::Mul(basisDy, t0, y, mask);
        Reg::Muls(basisDy, basisDy, 4.0f * L4_M4_SH_PARAM, mask);
        Reg::Duplicate(basisDz, 0.0f, mask);
        EmitShGradient(vCoeffs, coeffs, 24 * rowStride + pos, basis, basisDx, basisDy, basisDz,
                        vColor, vX, vY, vZ, mask);

        Reg::StoreAlign(vDirsAccum + pos, vX, mask);
        Reg::StoreAlign(vDirsAccum + rowStride + pos, vY, mask);
        Reg::StoreAlign(vDirsAccum + 2 * rowStride + pos, vZ, mask);
    }

__aicore__ inline void DispatchShChannel(
    __ubuf__ float* dirs, __ubuf__ float* coeffs, __ubuf__ float* vColors,
    __ubuf__ float* vDirsAccum, __ubuf__ float* vCoeffs,
    uint32_t count, uint32_t rowStride, int64_t degree, bool firstChannel)
    {
        for (uint32_t pos = 0; pos < count; pos += REG_FLOAT_PER_REPEAT) {
            const uint32_t left = count - pos;
            const uint32_t active = left < REG_FLOAT_PER_REPEAT ? left : REG_FLOAT_PER_REPEAT;
            if (firstChannel) {
                asc_vf_call<InitDirectionVF>(dirs, vDirsAccum, active, rowStride, pos);
            }
            asc_vf_call<ProcD0VF>(dirs, coeffs, vColors, vDirsAccum, vCoeffs,
                                active, rowStride, pos);
            if (degree >= 1) {
                asc_vf_call<ProcD1VF>(dirs, coeffs, vColors, vDirsAccum, vCoeffs,
                                    active, rowStride, pos);                
            }    
            if (degree >= 2) {
                asc_vf_call<ProcD2VF>(dirs, coeffs, vColors, vDirsAccum, vCoeffs,
                                    active, rowStride, pos);  
            }          
            if (degree >= 3) {
                asc_vf_call<ProcD3VF>(dirs, coeffs, vColors, vDirsAccum, vCoeffs,
                                    active, rowStride, pos);  
            }
            if (degree >= 4) {
                asc_vf_call<ProcD4VF>(dirs, coeffs, vColors, vDirsAccum, vCoeffs,
                                    active, rowStride, pos);  
            }
        }
    }

__simd_vf__ inline void ProjectDirectionVF(__ubuf__ float* dirs,
                                            __ubuf__ float* vDirsAccum,
                                            __ubuf__ float* vDirs,
                                            uint32_t count, uint32_t rowStride,
                                            int64_t degree)
    {
        uint32_t remain = count;
        for (uint32_t pos = 0; pos < count; pos += REG_FLOAT_PER_REPEAT) {
            Reg::MaskReg mask = Reg::UpdateMask<float>(remain);
            Reg::RegTensor<float> x;
            Reg::RegTensor<float> y;
            Reg::RegTensor<float> z;
            Reg::RegTensor<float> vX;
            Reg::RegTensor<float> vY;
            Reg::RegTensor<float> vZ;
            Reg::RegTensor<float> norm;
            Reg::RegTensor<float> dot;
            Reg::RegTensor<float> temp;
            Reg::RegTensor<float> out;

            if (degree == DEGREE_ZERO) {
                Reg::Duplicate(out, 0.0f, mask);
                Reg::StoreAlign(vDirs + pos, out, mask);
                Reg::StoreAlign(vDirs + rowStride + pos, out, mask);
                Reg::StoreAlign(vDirs + 2 * rowStride + pos, out, mask);
                continue;
            }

            Reg::LoadAlign(x, dirs + pos);
            Reg::LoadAlign(y, dirs + rowStride + pos);
            Reg::LoadAlign(z, dirs + 2 * rowStride + pos);
            Reg::LoadAlign(vX, vDirsAccum + pos);
            Reg::LoadAlign(vY, vDirsAccum + rowStride + pos);
            Reg::LoadAlign(vZ, vDirsAccum + 2 * rowStride + pos);
            Reg::LoadAlign(norm, vDirsAccum + 3 * rowStride + pos);

            Reg::Mul(dot, vX, x, mask);
            Reg::Mul(temp, vY, y, mask);
            Reg::Add(dot, dot, temp, mask);
            Reg::Mul(temp, vZ, z, mask);
            Reg::Add(dot, dot, temp, mask);

            Reg::Mul(temp, x, dot, mask);
            Reg::Sub(out, vX, temp, mask);
            Reg::Div(out, out, norm, mask);
            Reg::StoreAlign(vDirs + pos, out, mask);

            Reg::Mul(temp, y, dot, mask);
            Reg::Sub(out, vY, temp, mask);
            Reg::Div(out, out, norm, mask);
            Reg::StoreAlign(vDirs + rowStride + pos, out, mask);

            Reg::Mul(temp, z, dot, mask);
            Reg::Sub(out, vZ, temp, mask);
            Reg::Div(out, out, norm, mask);
            Reg::StoreAlign(vDirs + 2 * rowStride + pos, out, mask);
        }
    }

/*
 * Degree-0 read elision helpers. For degree 0 the outputs are independent of
 * dirs and coeffs (v_coeffs = Y_0 * v_colors, v_dirs = 0), so the dedicated
 * path only streams v_colors and writes the two outputs with a vector Muls and
 * a vector fill; it never touches dirsGm_ or coeffsGm_.
 */
__aicore__ inline void CopyInVColorsDegree0(
    TQue<QuePosition::VECIN, 2>& vColorsQue, GlobalTensor<float>& vColorsGm,
    int64_t gmOffset, uint32_t len)
{
    LocalTensor<float> vColorsLocal = vColorsQue.AllocTensor<float>();
    DataCopyExtParams copyParams{1, len, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(vColorsLocal, vColorsGm[gmOffset], copyParams, padParams);
    vColorsQue.EnQue(vColorsLocal);
}

__aicore__ inline void ZeroVDirsDegree0(LocalTensor<float>& vDirsLocal, uint32_t count)
{
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    Duplicate(vDirsLocal, 0.0f, static_cast<int32_t>(count));
}
/*
 * Degree-0 channel-batched DMA predicate. The batched path is bounded to the
 * low-MBU region (N_ <= BATCH_MAX_N) so the medium/large cases keep the
 * per-channel software-pipelined path byte-identical.
 */
constexpr int64_t BATCH_MAX_N = 131072;
constexpr int64_t BATCH_CHANNEL_NUM = 3;

class SphericalHarmonicsBwd {
public:
    __aicore__ inline SphericalHarmonicsBwd() {};
    __aicore__ inline void Init(GM_ADDR dirs, GM_ADDR coeffs, GM_ADDR vColors, 
                                GM_ADDR vDirs, GM_ADDR vCoeffs, TPipe *pipe, 
                                const SphericalHarmonicsBwdTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInDirs(int64_t batchIndex, int64_t loopIndex,
                                        int32_t element);
    __aicore__ inline void CopyInChannel(int64_t batchIndex, int64_t loopIndex,
                                        int32_t element, int64_t channelIndex);
    __aicore__ inline void CopyOutVCoeffsChannel(int64_t batchIndex, int64_t loopIndex,
                                                int32_t element, int64_t channelIndex);
    __aicore__ inline void CopyOutVDirs(int64_t batchIndex, int64_t loopIndex,
                                        int32_t element);
    __aicore__ inline void SubProcess(int64_t batchIndex, int64_t loopIndex,
                                        int64_t element);

    int64_t B_ = 0;
    int64_t N_ = 0;
    int64_t perCoreN_ = 0;
    int64_t curCoreN_ = 0;
    int64_t perLoopN_ = 0;
    int64_t lastLoopN_ = 0;
    int64_t perLoopMaxN_ = 0;
    int64_t loopN_ = 0;
    int64_t blockIdx_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t degree_ = 0;
    int64_t K_ = 0;
    int64_t channelNum_ = 0;

    GlobalTensor<float> dirsGm_;
    GlobalTensor<float> coeffsGm_;
    GlobalTensor<float> vColorsGm_;
    GlobalTensor<float> vDirsGm_;
    GlobalTensor<float> vCoeffsGm_;

    LocalTensor<float> dirsLocal_;
    LocalTensor<float> vDirsLocal_;

    TQue<QuePosition::VECIN, 2> dirsQue_;
    TQue<QuePosition::VECIN, 2> coeffsQue_;
    TQue<QuePosition::VECIN, 2> vColorsQue_;
    TQue<QuePosition::VECOUT, 2> vDirsQue_;
    TQue<QuePosition::VECOUT, 2> vCoeffsQue_;
};

__aicore__ inline void SphericalHarmonicsBwd::Init(GM_ADDR dirs, GM_ADDR coeffs, GM_ADDR vColors, GM_ADDR vDirs,
                                                   GM_ADDR vCoeffs, TPipe *pipe,
                                                   const SphericalHarmonicsBwdTilingData *tilingData)
{
    blockIdx_ = GetBlockIdx();
    B_ = tilingData->batchNum;
    N_ = tilingData->gaussNum;
    degree_ = tilingData->degree;
    perCoreN_ = tilingData->blockLength;
    perLoopMaxN_ = tilingData->perloopNum;
    needCoreNum_ = tilingData->needCoreNum;
    K_ = tilingData->K;
    channelNum_ = tilingData->channelNum;

    curCoreN_ = perCoreN_;
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreN_ = tilingData->lastcoreNum;
    }

    perLoopN_ = Min(perLoopMaxN_, curCoreN_);
    if (perLoopN_ == 0) {
        return;
    }
    loopN_ = curCoreN_ / perLoopN_;
    lastLoopN_ = curCoreN_ - perLoopN_ * loopN_;

    // S5: bounded single-iteration -> multi-iteration clamp. When the first
    // tile covers the whole per-core range (loopN_ == 1, lastLoopN_ == 0) the
    // degree>=1 depth-2 cross-loop pipeline branch below is skipped and the
    // dirs MTE2 / v_dirs MTE3 stay self-waited. For mid-N per-core ranges the
    // tile is halved (64-float aligned, still >= 2*REG_FLOAT_PER_REPEAT) only
    // when the pre-clamp loop is single-iteration and curCoreN_ is at least
    // MIN_MULTILOOP_N, so loopN_/lastLoopN_ are recomputed with totalIter>=2
    // and the existing depth-2 pipeline engages. Bytes, dtype, layout and
    // arithmetic order are unchanged and the degree-0 route and every
    // already-multi-iteration path keep their baseline tile byte-identically.
    constexpr int64_t MIN_MULTILOOP_N = 1024;
    if (degree_ >= DEGREE_ONE && curCoreN_ >= MIN_MULTILOOP_N &&
        loopN_ == 1 && lastLoopN_ == 0) {
        int64_t halfTile = perLoopN_ / 2;
        halfTile = halfTile / REG_FLOAT_PER_REPEAT * REG_FLOAT_PER_REPEAT;
        if (halfTile >= 2 * REG_FLOAT_PER_REPEAT) {
            perLoopN_ = halfTile;
            loopN_ = curCoreN_ / perLoopN_;
            lastLoopN_ = curCoreN_ - perLoopN_ * loopN_;
        }
    }

    dirsGm_.SetGlobalBuffer((__gm__ float *)dirs);
    coeffsGm_.SetGlobalBuffer((__gm__ float *)coeffs);
    vColorsGm_.SetGlobalBuffer((__gm__ float *)vColors);
    vDirsGm_.SetGlobalBuffer((__gm__ float *)vDirs);
    vCoeffsGm_.SetGlobalBuffer((__gm__ float *)vCoeffs);

    // dirs and accumulated v_dirs are shared by every logical channel.
    // Channel-private input/output queues use two buffers when C > 1 so MTE2,
    // Vector and MTE3 can work on adjacent channels concurrently. C == 1 uses
    // one physical buffer and gives the remaining UB capacity to a larger tile.
    uint8_t channelBufferNum = channelNum_ > 1 ? 2 : 1;
    if (degree_ == DEGREE_ZERO && channelNum_ == BATCH_CHANNEL_NUM && N_ <= BATCH_MAX_N) {
        // Degree-0 batched path: one multi-block buffer per tensor, sized to the
        // channel/axis count. dirsQue_ is still initialized because Process()
        // allocates from it (it is not otherwise touched on the degree-0 path).
        pipe->InitBuffer(dirsQue_, 1, CONSTDIM_THREE * RegAlignBytes(perLoopN_));
        pipe->InitBuffer(vColorsQue_, 1, CONSTDIM_THREE * RegAlignBytes(perLoopN_));
        pipe->InitBuffer(vCoeffsQue_, 1, CONSTDIM_THREE * RegAlignBytes(perLoopN_));
        pipe->InitBuffer(vDirsQue_, 1, CONSTDIM_THREE * RegAlignBytes(perLoopN_));
        return;
    }
    int64_t totalIter = B_ * (loopN_ + (lastLoopN_ > 0 ? 1 : 0));
    if (degree_ >= DEGREE_ONE && totalIter > 1) {
        // Cross-loop pipeline: dirs/v_dirs get a second buffer so the next
        // iteration's dirs MTE2 and the previous iteration's v_dirs MTE3
        // overlap the current Vector work. The extra buffers are reclaimed
        // from the same UB budget by shrinking the per-loop tile; the channel
        // buffers keep depth >= 2 when C > 1.
        int64_t extraRows = 3 + 4;
        int64_t rowsOrig = 7 + channelBufferNum * (2 * K_ + 1);
        int64_t rowsPipe = rowsOrig + extraRows;
        int64_t pipeTile = perLoopMaxN_ * rowsOrig / rowsPipe;
        pipeTile = pipeTile / REG_FLOAT_PER_REPEAT * REG_FLOAT_PER_REPEAT;
        if (pipeTile < REG_FLOAT_PER_REPEAT) {
            pipeTile = REG_FLOAT_PER_REPEAT;
        }
        if (perLoopN_ > pipeTile) {
            perLoopN_ = pipeTile;
        }
        loopN_ = curCoreN_ / perLoopN_;
        lastLoopN_ = curCoreN_ - perLoopN_ * loopN_;
        pipe->InitBuffer(dirsQue_, 2, 3 * RegAlignBytes(perLoopN_));
        pipe->InitBuffer(coeffsQue_, channelBufferNum, K_ * RegAlignBytes(perLoopN_));
        pipe->InitBuffer(vColorsQue_, channelBufferNum, RegAlignBytes(perLoopN_));
        pipe->InitBuffer(vDirsQue_, 2, 4 * RegAlignBytes(perLoopN_));
        pipe->InitBuffer(vCoeffsQue_, channelBufferNum, K_ * RegAlignBytes(perLoopN_));
        return;
    }
    pipe->InitBuffer(dirsQue_, 1, CONSTDIM_THREE * RegAlignBytes(perLoopN_));
    pipe->InitBuffer(coeffsQue_, channelBufferNum, K_ * RegAlignBytes(perLoopN_));
    pipe->InitBuffer(vColorsQue_, channelBufferNum, RegAlignBytes(perLoopN_));
    pipe->InitBuffer(vDirsQue_, 1, (CONSTDIM_THREE + 1) * RegAlignBytes(perLoopN_));
    pipe->InitBuffer(vCoeffsQue_, channelBufferNum, K_ * RegAlignBytes(perLoopN_));
}

/*
 * Degree>=1 cross-loop software-pipeline helpers. The dirs read and the v_dirs
 * write of adjacent loop iterations are moved by depth-2 queues so that the
 * next iteration's dirs MTE2 and the previous iteration's v_dirs MTE3 overlap
 * the current iteration's Vector work. UB row gaps use 32B units; rowStride is
 * 64-float aligned so the aligned hardware stride reconstructs the row pitch.
 */
// 7 = 3 dirs rows + 4 accumulated-v_dirs rows; the dirs/v_dirs second buffers
// add 3 + 4 rows.
__aicore__ inline uint32_t PipelineUbGap(int64_t ubStrideElems, int64_t element)
{
    return static_cast<uint32_t>((ubStrideElems - element) / INT32_ONE_BLOCK_NUM);
}

__aicore__ inline void CopyInDirsQueued(TQue<QuePosition::VECIN, 2>& dirsQue,
    GlobalTensor<float>& dirsGm, int64_t gmBase, uint32_t len, uint32_t gmGap, uint32_t ubGap)
{
    LocalTensor<float> dirsLocal = dirsQue.AllocTensor<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(CONSTDIM_THREE), len, gmGap, ubGap, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dirsLocal, dirsGm[gmBase], copyParams, padParams);
    dirsQue.EnQue(dirsLocal);
}

__aicore__ inline void DrainVDirsQueued(TQue<QuePosition::VECOUT, 2>& vDirsQue,
    GlobalTensor<float>& vDirsGm, int64_t gmBase, uint32_t len, uint32_t gmGap, uint32_t ubGap)
{
    LocalTensor<float> vDirsLocal = vDirsQue.DeQue<float>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(CONSTDIM_THREE), len, ubGap, gmGap, 0};
    DataCopyPad(vDirsGm[gmBase], vDirsLocal, copyParams);
    vDirsQue.FreeTensor(vDirsLocal);
}

__aicore__ inline void SphericalHarmonicsBwd::CopyInDirs(
    int64_t batchIndex, int64_t loopIndex, int32_t element)
{
    uint32_t len = element * sizeof(float);
    int64_t rowStride = RegAlign(element);
    int64_t taskOffset = perLoopN_ * loopIndex + perCoreN_ * blockIdx_;
    int64_t dirsBase = batchIndex * CONSTDIM_THREE * N_ + taskOffset;
    DataCopyExtParams copyParams{1, len, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    for (int64_t dirIndex = 0; dirIndex < CONSTDIM_THREE; ++dirIndex) {
        DataCopyPad(dirsLocal_[dirIndex * rowStride], dirsGm_[dirsBase + dirIndex * N_],
                    copyParams, padParams);
    }
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
}

__aicore__ inline void SphericalHarmonicsBwd::CopyInChannel(
    int64_t batchIndex, int64_t loopIndex, int32_t element, int64_t channelIndex)
{
    LocalTensor<float> coeffsLocal = coeffsQue_.AllocTensor<float>();
    LocalTensor<float> vColorsLocal = vColorsQue_.AllocTensor<float>();
    uint32_t len = element * sizeof(float);
    int64_t rowStride = RegAlign(element);
    int64_t taskOffset = perLoopN_ * loopIndex + perCoreN_ * blockIdx_;
    DataCopyExtParams copyParams{1, len, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    int64_t vColorsOffset = batchIndex * channelNum_ * N_ + channelIndex * N_ + taskOffset;
    DataCopyPad(vColorsLocal, vColorsGm_[vColorsOffset], copyParams, padParams);

    for (int64_t coeffIndex = 0; coeffIndex < K_; ++coeffIndex) {
        int64_t gmOffset = batchIndex * K_ * channelNum_ * N_ +
                            (coeffIndex * channelNum_ + channelIndex) * N_ + taskOffset;
        DataCopyPad(coeffsLocal[coeffIndex * rowStride], coeffsGm_[gmOffset], copyParams, padParams);
    }
    coeffsQue_.EnQue(coeffsLocal);
    vColorsQue_.EnQue(vColorsLocal);
}

__aicore__ inline void SphericalHarmonicsBwd::CopyOutVCoeffsChannel(
    int64_t batchIndex, int64_t loopIndex, int32_t element, int64_t channelIndex)
{
    LocalTensor<float> vCoeffsLocal = vCoeffsQue_.DeQue<float>();
    uint32_t len = element * sizeof(float);
    int64_t rowStride = RegAlign(element);
    int64_t taskOffset = perLoopN_ * loopIndex + perCoreN_ * blockIdx_;
    DataCopyExtParams copyParams{1, len, 0, 0, 0};

    for (int64_t coeffIndex = 0; coeffIndex < K_; ++coeffIndex) {
        int64_t gmOffset = batchIndex * K_ * channelNum_ * N_ +
                            (coeffIndex * channelNum_ + channelIndex) * N_ + taskOffset;
        DataCopyPad(vCoeffsGm_[gmOffset], vCoeffsLocal[coeffIndex * rowStride], copyParams);
    }
    vCoeffsQue_.FreeTensor(vCoeffsLocal);
}

__aicore__ inline void SphericalHarmonicsBwd::CopyOutVDirs(
    int64_t batchIndex, int64_t loopIndex, int32_t element)
{
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    uint32_t len = element * sizeof(float);
    int64_t rowStride = RegAlign(element);
    int64_t taskOffset = perLoopN_ * loopIndex + perCoreN_ * blockIdx_;
    int64_t gmBase = batchIndex * CONSTDIM_THREE * N_ + taskOffset;
    DataCopyExtParams copyParams{1, len, 0, 0, 0};
    for (int64_t dirIndex = 0; dirIndex < CONSTDIM_THREE; ++dirIndex) {
        DataCopyPad(vDirsGm_[gmBase + dirIndex * N_], vDirsLocal_[dirIndex * rowStride], copyParams);
    }
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
}

__aicore__ inline void SphericalHarmonicsBwd::SubProcess(
    int64_t batchIndex, int64_t loopIndex, int64_t element)
{
    if (degree_ == DEGREE_ZERO) {
        if (channelNum_ == BATCH_CHANNEL_NUM && N_ <= BATCH_MAX_N) {
            // Channel-batched degree-0: one multi-block DataCopyPad per tensor
            // (v_colors read, v_coeffs write, v_dirs write) instead of three each.
            uint32_t len = static_cast<uint32_t>(element) * sizeof(float);
            int64_t taskOffset = perLoopN_ * loopIndex + perCoreN_ * blockIdx_;
            int64_t rowStride = RegAlign(element);
            int64_t gmGapBytes = (N_ - element) * static_cast<int64_t>(sizeof(float));
            int64_t ubGapBlocks = (rowStride - element) / INT32_ONE_BLOCK_NUM;
            DataCopyExtParams inParams{static_cast<uint16_t>(BATCH_CHANNEL_NUM), len,
                                       static_cast<int64_t>(gmGapBytes), static_cast<int64_t>(ubGapBlocks), 0};
            DataCopyExtParams outParams{static_cast<uint16_t>(BATCH_CHANNEL_NUM), len,
                                        static_cast<int64_t>(ubGapBlocks), static_cast<int64_t>(gmGapBytes), 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            Duplicate(vDirsLocal_, 0.0f, static_cast<int32_t>(CONSTDIM_THREE * rowStride));
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

            int64_t vColorsBase = batchIndex * channelNum_ * N_ + taskOffset;
            LocalTensor<float> vColorsLocal = vColorsQue_.AllocTensor<float>();
            DataCopyPad(vColorsLocal, vColorsGm_[vColorsBase], inParams, padParams);
            vColorsQue_.EnQue(vColorsLocal);

            vColorsLocal = vColorsQue_.DeQue<float>();
            LocalTensor<float> vCoeffsLocal = vCoeffsQue_.AllocTensor<float>();
            Muls(vCoeffsLocal, vColorsLocal, L0_M0_SH_PARAM,
                 static_cast<int32_t>(CONSTDIM_THREE * rowStride));
            vCoeffsQue_.EnQue(vCoeffsLocal);
            vColorsQue_.FreeTensor(vColorsLocal);

            vCoeffsLocal = vCoeffsQue_.DeQue<float>();
            int64_t vCoeffsBase = batchIndex * K_ * channelNum_ * N_ + taskOffset;
            DataCopyPad(vCoeffsGm_[vCoeffsBase], vCoeffsLocal, outParams);
            vCoeffsQue_.FreeTensor(vCoeffsLocal);

            int64_t vDirsBase = batchIndex * CONSTDIM_THREE * N_ + taskOffset;
            DataCopyPad(vDirsGm_[vDirsBase], vDirsLocal_, outParams);
            SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            return;
        }
        // Degree-0 fast path: no dirs/coeffs GM traffic, no normalization VF.
        uint32_t len = static_cast<uint32_t>(element) * sizeof(float);
        int64_t taskOffset = perLoopN_ * loopIndex + perCoreN_ * blockIdx_;
        int64_t vColorsBase = batchIndex * channelNum_ * N_ + taskOffset;
        ZeroVDirsDegree0(vDirsLocal_,
                         static_cast<uint32_t>(CONSTDIM_THREE * RegAlign(element)));
        CopyInVColorsDegree0(vColorsQue_, vColorsGm_, vColorsBase, len);
        for (int64_t channelIndex = 0; channelIndex < channelNum_; ++channelIndex) {
            LocalTensor<float> vColorsLocal = vColorsQue_.DeQue<float>();
            // Prefetch the next logical channel while Vector computes this one.
            if (channelIndex + 1 < channelNum_) {
                CopyInVColorsDegree0(vColorsQue_, vColorsGm_,
                                     vColorsBase + (channelIndex + 1) * N_, len);
            }
            // Start writing the previous result while Vector computes this one.
            if (channelIndex > 0) {
                CopyOutVCoeffsChannel(batchIndex, loopIndex, element, channelIndex - 1);
            }
            LocalTensor<float> vCoeffsLocal = vCoeffsQue_.AllocTensor<float>();
            Muls(vCoeffsLocal, vColorsLocal, L0_M0_SH_PARAM,
                 static_cast<int32_t>(element));
            vCoeffsQue_.EnQue(vCoeffsLocal);
            vColorsQue_.FreeTensor(vColorsLocal);
        }
        CopyOutVCoeffsChannel(batchIndex, loopIndex, element, channelNum_ - 1);
        CopyOutVDirs(batchIndex, loopIndex, element);
        return;
    }
    uint32_t rowStride = static_cast<uint32_t>(RegAlign(element));
    CopyInDirs(batchIndex, loopIndex, element);
    CopyInChannel(batchIndex, loopIndex, element, 0);
    for (int64_t channelIndex = 0; channelIndex < channelNum_; ++channelIndex) {
        LocalTensor<float> coeffsLocal = coeffsQue_.DeQue<float>();
        LocalTensor<float> vColorsLocal = vColorsQue_.DeQue<float>();

        // Prefetch the next logical channel while Vector computes this one.
        if (channelIndex + 1 < channelNum_) {
            CopyInChannel(batchIndex, loopIndex, element, channelIndex + 1);
        }
        // Start writing the previous result while Vector computes this one.
        if (channelIndex > 0) {
            CopyOutVCoeffsChannel(batchIndex, loopIndex, element, channelIndex - 1);
        }

        LocalTensor<float> vCoeffsLocal = vCoeffsQue_.AllocTensor<float>();
        // Resolve UB pointers in the aicore domain before entering the VF.
        DispatchShChannel(
            (__ubuf__ float*)dirsLocal_.GetPhyAddr(),
            (__ubuf__ float*)coeffsLocal.GetPhyAddr(),
            (__ubuf__ float*)vColorsLocal.GetPhyAddr(),
            (__ubuf__ float*)vDirsLocal_.GetPhyAddr(),
            (__ubuf__ float*)vCoeffsLocal.GetPhyAddr(),
            static_cast<uint32_t>(element), rowStride, degree_, channelIndex == 0);

        vCoeffsQue_.EnQue(vCoeffsLocal);
        coeffsQue_.FreeTensor(coeffsLocal);
        vColorsQue_.FreeTensor(vColorsLocal);
    }
    CopyOutVCoeffsChannel(batchIndex, loopIndex, element, channelNum_ - 1);

    asc_vf_call<ProjectDirectionVF>(
                                    (__ubuf__ float*)dirsLocal_.GetPhyAddr(),
                                    (__ubuf__ float*)vDirsLocal_.GetPhyAddr(),
                                    (__ubuf__ float*)vDirsLocal_.GetPhyAddr(),
                                    static_cast<uint32_t>(element), rowStride, degree_);
    CopyOutVDirs(batchIndex, loopIndex, element);
}

__aicore__ inline void SphericalHarmonicsBwd::Process()
{
    int64_t totalIter = B_ * (loopN_ + (lastLoopN_ > 0 ? 1 : 0));
    if (degree_ >= DEGREE_ONE && totalIter > 1) {
        // Depth-2 cross-loop pipeline (degree>=1): prefetch the next
        // iteration's dirs and drain the previous iteration's v_dirs through
        // queued EnQue/DeQue pairing, so dirs MTE2 and v_dirs MTE3 overlap the
        // current iteration's Vector work instead of being self-waited.
        int64_t loopsPerBatch = loopN_ + (lastLoopN_ > 0 ? 1 : 0);
        {
            int64_t el = (loopN_ > 0) ? perLoopN_ : lastLoopN_;
            uint32_t len = static_cast<uint32_t>(el) * FLOAT_SIZE;
            CopyInDirsQueued(dirsQue_, dirsGm_, perCoreN_ * blockIdx_, len,
                             static_cast<uint32_t>((N_ - el) * FLOAT_SIZE),
                             PipelineUbGap(RegAlign(el), el));
        }
        int64_t prevBatch = 0;
        int64_t prevLoop = 0;
        int64_t prevEl = 0;
        bool hasPrev = false;
        for (int64_t it = 0; it < totalIter; ++it) {
            int64_t b = it / loopsPerBatch;
            int64_t rem = it - b * loopsPerBatch;
            int64_t lp = 0;
            int64_t el = 0;
            if (rem < loopN_) {
                lp = rem;
                el = perLoopN_;
            } else {
                lp = loopN_;
                el = lastLoopN_;
            }
            uint32_t rowStride = static_cast<uint32_t>(RegAlign(el));
            LocalTensor<float> dirsLocal = dirsQue_.DeQue<float>();
            if (it + 1 < totalIter) {
                int64_t nb = (it + 1) / loopsPerBatch;
                int64_t nrem = (it + 1) - nb * loopsPerBatch;
                int64_t nlp = 0;
                int64_t nel = 0;
                if (nrem < loopN_) {
                    nlp = nrem;
                    nel = perLoopN_;
                } else {
                    nlp = loopN_;
                    nel = lastLoopN_;
                }
                uint32_t nlen = static_cast<uint32_t>(nel) * FLOAT_SIZE;
                CopyInDirsQueued(dirsQue_, dirsGm_,
                                 nb * CONSTDIM_THREE * N_ + perCoreN_ * blockIdx_ + perLoopN_ * nlp, nlen,
                                 static_cast<uint32_t>((N_ - nel) * FLOAT_SIZE),
                                 PipelineUbGap(RegAlign(nel), nel));
            }
            if (hasPrev) {
                uint32_t plen = static_cast<uint32_t>(prevEl) * FLOAT_SIZE;
                DrainVDirsQueued(vDirsQue_, vDirsGm_,
                                 prevBatch * CONSTDIM_THREE * N_ + perCoreN_ * blockIdx_ + perLoopN_ * prevLoop,
                                 plen, static_cast<uint32_t>((N_ - prevEl) * FLOAT_SIZE),
                                 PipelineUbGap(RegAlign(prevEl), prevEl));
            }
            LocalTensor<float> vDirsLocal = vDirsQue_.AllocTensor<float>();
            CopyInChannel(b, lp, el, 0);
            for (int64_t channelIndex = 0; channelIndex < channelNum_; ++channelIndex) {
                LocalTensor<float> coeffsLocal = coeffsQue_.DeQue<float>();
                LocalTensor<float> vColorsLocal = vColorsQue_.DeQue<float>();
                if (channelIndex + 1 < channelNum_) {
                    CopyInChannel(b, lp, el, channelIndex + 1);
                }
                if (channelIndex > 0) {
                    CopyOutVCoeffsChannel(b, lp, el, channelIndex - 1);
                }
                LocalTensor<float> vCoeffsLocal = vCoeffsQue_.AllocTensor<float>();
                DispatchShChannel(
                    (__ubuf__ float*)dirsLocal.GetPhyAddr(),
                    (__ubuf__ float*)coeffsLocal.GetPhyAddr(),
                    (__ubuf__ float*)vColorsLocal.GetPhyAddr(),
                    (__ubuf__ float*)vDirsLocal.GetPhyAddr(),
                    (__ubuf__ float*)vCoeffsLocal.GetPhyAddr(),
                    static_cast<uint32_t>(el), rowStride, degree_, channelIndex == 0);
                vCoeffsQue_.EnQue(vCoeffsLocal);
                coeffsQue_.FreeTensor(coeffsLocal);
                vColorsQue_.FreeTensor(vColorsLocal);
            }
            CopyOutVCoeffsChannel(b, lp, el, channelNum_ - 1);
            asc_vf_call<ProjectDirectionVF>(
                                            (__ubuf__ float*)dirsLocal.GetPhyAddr(),
                                            (__ubuf__ float*)vDirsLocal.GetPhyAddr(),
                                            (__ubuf__ float*)vDirsLocal.GetPhyAddr(),
                                            static_cast<uint32_t>(el), rowStride, degree_);
            vDirsQue_.EnQue(vDirsLocal);
            dirsQue_.FreeTensor(dirsLocal);
            prevBatch = b;
            prevLoop = lp;
            prevEl = el;
            hasPrev = true;
        }
        {
            uint32_t plen = static_cast<uint32_t>(prevEl) * FLOAT_SIZE;
            DrainVDirsQueued(vDirsQue_, vDirsGm_,
                             prevBatch * CONSTDIM_THREE * N_ + perCoreN_ * blockIdx_ + perLoopN_ * prevLoop,
                             plen, static_cast<uint32_t>((N_ - prevEl) * FLOAT_SIZE),
                             PipelineUbGap(RegAlign(prevEl), prevEl));
        }
        return;
    }
    dirsLocal_ = dirsQue_.AllocTensor<float>();
    vDirsLocal_ = vDirsQue_.AllocTensor<float>();
    for (int64_t batchIndex = 0; batchIndex < B_; ++batchIndex) {
        int64_t loopIndex = 0;
        for (; loopIndex < loopN_; ++loopIndex) {
            SubProcess(batchIndex, loopIndex, perLoopN_);
        }
        if (lastLoopN_ > 0) {
            SubProcess(batchIndex, loopIndex, lastLoopN_);
        }
    }
    dirsQue_.FreeTensor(dirsLocal_);
    vDirsQue_.FreeTensor(vDirsLocal_);
}

} // namespace SphericalHarmonicsBwdNs
#endif // SPHERICAL_HARMONICS_BWD_H