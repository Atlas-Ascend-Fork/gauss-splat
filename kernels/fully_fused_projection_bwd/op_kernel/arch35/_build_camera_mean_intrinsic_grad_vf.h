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

__aicore__ inline void _build_camera_mean_intrinsic_grad_vf(__ubuf__ float* means_ub, __ubuf__ float* vmeans2d_ub, __ubuf__ float* proj_covar_ub, __ubuf__ float* camera_vmean_ub, __ubuf__ float* camera_grad_ub){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> _tmp_reg_1;

        MicroAPI::RegTensor<float> mx;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(mx, means_ub, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> my;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(my, means_ub + 1 * 1024, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> mz;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(mz, means_ub + 2 * 1024, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> vmx;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vmx, vmeans2d_ub, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vmy;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vmy, vmeans2d_ub + 1 * 1024, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> pcx;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(pcx, proj_covar_ub + 7 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> pcy;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(pcy, proj_covar_ub + 8 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> inv_z2;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(inv_z2, proj_covar_ub + 9 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> inv_z;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(inv_z, proj_covar_ub + 10 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> norm_x;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(norm_x, proj_covar_ub + 13 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> norm_y;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(norm_y, proj_covar_ub + 14 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vj00;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vj00, proj_covar_ub + 15 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vj11;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vj11, proj_covar_ub + 16 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vj02;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vj02, proj_covar_ub + 17 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> vj12;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vj12, proj_covar_ub + 18 * 64, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> cvx;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cvx, camera_vmean_ub, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cvy;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cvy, camera_vmean_ub + 1 * 64, 1, _tmp_maskreg_0);
        MicroAPI::RegTensor<float> cvz;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(cvz, camera_vmean_ub + 2 * 64, 1, _tmp_maskreg_0);

        MicroAPI::RegTensor<float> term;
        MicroAPI::Mul(term, cvx, mx, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvx, my, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 1 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvx, mz, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 2 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvy, mx, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 3 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvy, my, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 4 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvy, mz, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 5 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvz, mx, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 6 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvz, my, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 7 * 64, term, 1, _tmp_maskreg_0);
        MicroAPI::Mul(term, cvz, mz, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 8 * 64, term, 1, _tmp_maskreg_0);

        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 9 * 64, cvx, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 10 * 64, cvy, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 11 * 64, cvz, 1, _tmp_maskreg_0);

        MicroAPI::Mul(term, vmx, norm_x, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, vj00, inv_z, _tmp_maskreg_0);
        MicroAPI::Add(term, term, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, vj02, pcx, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, _tmp_reg_1, inv_z2, _tmp_maskreg_0);
        MicroAPI::Sub(term, term, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 12 * 64, term, 1, _tmp_maskreg_0);

        MicroAPI::Mul(term, vmy, norm_y, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, vj11, inv_z, _tmp_maskreg_0);
        MicroAPI::Add(term, term, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, vj12, pcy, _tmp_maskreg_0);
        MicroAPI::Mul(_tmp_reg_1, _tmp_reg_1, inv_z2, _tmp_maskreg_0);
        MicroAPI::Sub(term, term, _tmp_reg_1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 13 * 64, term, 1, _tmp_maskreg_0);

        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 14 * 64, vmx, 1, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(camera_grad_ub + 15 * 64, vmy, 1, _tmp_maskreg_0);
    }
}
