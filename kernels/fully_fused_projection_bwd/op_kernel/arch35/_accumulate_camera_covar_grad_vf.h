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

__aicore__ inline void _accumulate_camera_covar_grad_vf(__ubuf__ float* proj_covar_ub, __ubuf__ float* view_ub, __ubuf__ float* world_covar_ub, __ubuf__ float* camera_grad_ub){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_1;
        MicroAPI::RegTensor<float> _tmp_reg_2;
        MicroAPI::RegTensor<float> two;
        MicroAPI::Duplicate(two, 2.0f, _tmp_maskreg_0);

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

        MicroAPI::RegTensor<float> cw00;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cw00, world_covar_ub, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cw01;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cw01, world_covar_ub + 1 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cw02;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cw02, world_covar_ub + 2 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cw11;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cw11, world_covar_ub + 3 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cw12;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cw12, world_covar_ub + 4 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cw22;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cw22, world_covar_ub + 5 * 64, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> p00;
        MicroAPI::Mul(p00, vc00, j00, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> p01;
        MicroAPI::Mul(p01, vc01, j11, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> p02;
        MicroAPI::Mul(_tmp_reg_1, vc00, j02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, vc01, j12, _tmp_maskreg_0);
        MicroAPI::Add(p02, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> p10;
        MicroAPI::Mul(p10, vc01, j00, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> p11;
        MicroAPI::Mul(p11, vc11, j11, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> p12;
        MicroAPI::Mul(_tmp_reg_1, vc01, j02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, vc11, j12, _tmp_maskreg_0);
        MicroAPI::Add(p12, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> m00;
        MicroAPI::Mul(m00, j00, p00, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> m01;
        MicroAPI::Mul(m01, j00, p01, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> m02;
        MicroAPI::Mul(m02, j00, p02, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n00;
        MicroAPI::Mul(_tmp_reg_1, m00, r00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m01, r10, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m02, r20, _tmp_maskreg_0);
        MicroAPI::Add(n00, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n01;
        MicroAPI::Mul(_tmp_reg_1, m00, r01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m01, r11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m02, r21, _tmp_maskreg_0);
        MicroAPI::Add(n01, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n02;
        MicroAPI::Mul(_tmp_reg_1, m00, r02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m01, r12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m02, r22, _tmp_maskreg_0);
        MicroAPI::Add(n02, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> g00;
        MicroAPI::Mul(_tmp_reg_1, n00, cw00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n01, cw01, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n02, cw02, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g00, two, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> g01;
        MicroAPI::Mul(_tmp_reg_1, n00, cw01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n01, cw11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n02, cw12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g01, two, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> g02;
        MicroAPI::Mul(_tmp_reg_1, n00, cw02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n01, cw12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n02, cw22, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g02, two, _tmp_reg_1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> prev;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g00, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub, prev, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 1 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g01, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 1 * 64, prev, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 2 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g02, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 2 * 64, prev, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> m10;
        MicroAPI::Mul(m10, j11, p10, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> m11;
        MicroAPI::Mul(m11, j11, p11, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> m12;
        MicroAPI::Mul(m12, j11, p12, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n10;
        MicroAPI::Mul(_tmp_reg_1, m10, r00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m11, r10, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m12, r20, _tmp_maskreg_0);
        MicroAPI::Add(n10, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n11;
        MicroAPI::Mul(_tmp_reg_1, m10, r01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m11, r11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m12, r21, _tmp_maskreg_0);
        MicroAPI::Add(n11, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n12;
        MicroAPI::Mul(_tmp_reg_1, m10, r02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m11, r12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m12, r22, _tmp_maskreg_0);
        MicroAPI::Add(n12, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> g10;
        MicroAPI::Mul(_tmp_reg_1, n10, cw00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n11, cw01, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n12, cw02, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g10, two, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> g11;
        MicroAPI::Mul(_tmp_reg_1, n10, cw01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n11, cw11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n12, cw12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g11, two, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> g12;
        MicroAPI::Mul(_tmp_reg_1, n10, cw02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n11, cw12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n12, cw22, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g12, two, _tmp_reg_1, _tmp_maskreg_0);

        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 3 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g10, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 3 * 64, prev, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 4 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g11, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 4 * 64, prev, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 5 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g12, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 5 * 64, prev, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> m20;
        MicroAPI::Mul(_tmp_reg_1, j02, p00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j12, p10, _tmp_maskreg_0);
        MicroAPI::Add(m20, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> m21;
        MicroAPI::Mul(_tmp_reg_1, j02, p01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j12, p11, _tmp_maskreg_0);
        MicroAPI::Add(m21, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> m22;
        MicroAPI::Mul(_tmp_reg_1, j02, p02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, j12, p12, _tmp_maskreg_0);
        MicroAPI::Add(m22, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n20;
        MicroAPI::Mul(_tmp_reg_1, m20, r00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m21, r10, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m22, r20, _tmp_maskreg_0);
        MicroAPI::Add(n20, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n21;
        MicroAPI::Mul(_tmp_reg_1, m20, r01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m21, r11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m22, r21, _tmp_maskreg_0);
        MicroAPI::Add(n21, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> n22;
        MicroAPI::Mul(_tmp_reg_1, m20, r02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m21, r12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, m22, r22, _tmp_maskreg_0);
        MicroAPI::Add(n22, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> g20;
        MicroAPI::Mul(_tmp_reg_1, n20, cw00, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n21, cw01, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n22, cw02, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g20, two, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> g21;
        MicroAPI::Mul(_tmp_reg_1, n20, cw01, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n21, cw11, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n22, cw12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g21, two, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> g22;
        MicroAPI::Mul(_tmp_reg_1, n20, cw02, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n21, cw12, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_2, n22, cw22, _tmp_maskreg_0);
        MicroAPI::Add(_tmp_reg_1, _tmp_reg_1, _tmp_reg_2, _tmp_maskreg_0);
        MicroAPI::Mul(g22, two, _tmp_reg_1, _tmp_maskreg_0);

        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 6 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g20, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 6 * 64, prev, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 7 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g21, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 7 * 64, prev, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(prev, camera_grad_ub + 8 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Add(prev, prev, g22, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 8 * 64, prev, 1, _tmp_maskreg_0);
    }
}
