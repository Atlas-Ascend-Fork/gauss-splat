/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "tensorutils.h"

__aicore__ inline void _accumulate_covar_direct_no_vr_from_intermediates_vf(__ubuf__ float* proj_covar_ub, __ubuf__ float* view_ub, __ubuf__ float* world_covar_ub, __ubuf__ float* covar_accum_ub){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_2;
        MicroAPI::RegTensor<float> _tmp_reg_1;
        MicroAPI::RegTensor<float> r00;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r00, view_ub);
        MicroAPI::RegTensor<float> r01;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r01, view_ub + 1);
        MicroAPI::RegTensor<float> r02;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r02, view_ub + 2);
        MicroAPI::RegTensor<float> r10;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r10, view_ub + 4);
        MicroAPI::RegTensor<float> r11;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r11, view_ub + 5);
        MicroAPI::RegTensor<float> r12;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r12, view_ub + 6);
        MicroAPI::RegTensor<float> r20;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r20, view_ub + 8);
        MicroAPI::RegTensor<float> r21;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r21, view_ub + 9);
        MicroAPI::RegTensor<float> r22;
        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(r22, view_ub + 10);
        MicroAPI::RegTensor<float> vc00;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vc00, proj_covar_ub, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vc01;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vc01, proj_covar_ub + 1 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vc11;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vc11, proj_covar_ub + 2 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> j00;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(j00, proj_covar_ub + 3 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> j11;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(j11, proj_covar_ub + 4 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> j02;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(j02, proj_covar_ub + 5 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> j12;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(j12, proj_covar_ub + 6 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> two;
        MicroAPI::Duplicate(two, 2.0f, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> s00;
        MicroAPI::Mul(s00, two, vc00, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> s01;
        MicroAPI::Mul(s01, two, vc01, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> s11;
        MicroAPI::Mul(s11, two, vc11, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> a00;
        MicroAPI::Mul(_tmp_reg_1, j00, r00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j02, r20, _tmp_maskreg_0);
        MicroAPI::Add(a00, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> a01;
        MicroAPI::Mul(_tmp_reg_1, j00, r01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j02, r21, _tmp_maskreg_0);
        MicroAPI::Add(a01, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> a02;
        MicroAPI::Mul(_tmp_reg_1, j00, r02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j02, r22, _tmp_maskreg_0);
        MicroAPI::Add(a02, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> a10;
        MicroAPI::Mul(_tmp_reg_1, j11, r10, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j12, r20, _tmp_maskreg_0);
        MicroAPI::Add(a10, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> a11;
        MicroAPI::Mul(_tmp_reg_1, j11, r11, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j12, r21, _tmp_maskreg_0);
        MicroAPI::Add(a11, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> a12;
        MicroAPI::Mul(_tmp_reg_1, j11, r12, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j12, r22, _tmp_maskreg_0);
        MicroAPI::Add(a12, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> u00;
        MicroAPI::Mul(_tmp_reg_1, s00, a00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, s01, a10, _tmp_maskreg_0);
        MicroAPI::Add(u00, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> u01;
        MicroAPI::Mul(_tmp_reg_1, s00, a01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, s01, a11, _tmp_maskreg_0);
        MicroAPI::Add(u01, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> u02;
        MicroAPI::Mul(_tmp_reg_1, s00, a02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, s01, a12, _tmp_maskreg_0);
        MicroAPI::Add(u02, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> u10;
        MicroAPI::Mul(_tmp_reg_1, s01, a00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, s11, a10, _tmp_maskreg_0);
        MicroAPI::Add(u10, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> u11;
        MicroAPI::Mul(_tmp_reg_1, s01, a01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, s11, a11, _tmp_maskreg_0);
        MicroAPI::Add(u11, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> u12;
        MicroAPI::Mul(_tmp_reg_1, s01, a02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, s11, a12, _tmp_maskreg_0);
        MicroAPI::Add(u12, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> term;
        MicroAPI::RegTensor<float> half;
        MicroAPI::Duplicate(half, 0.5f, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, a00, u00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, a10, u10, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(term, half, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(covar_accum_ub, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, a00, u01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, a10, u11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(term, half, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(covar_accum_ub + 1 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, a00, u02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, a10, u12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(term, half, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(covar_accum_ub + 2 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, a01, u01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, a11, u11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(term, half, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(covar_accum_ub + 3 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, a01, u02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, a11, u12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(term, half, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(covar_accum_ub + 4 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, a02, u02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, a12, u12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(term, half, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(covar_accum_ub + 5 * 64, term, 1, _tmp_maskreg_0);
    }
}

