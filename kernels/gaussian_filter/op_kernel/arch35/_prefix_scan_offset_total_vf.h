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

__aicore__ inline void _prefix_scan_offset_total_vf(__ubuf__ float* offset_out_ub, __ubuf__ float* total_out_ub, __ubuf__ float* ws_all_ub, const float &my_idx_f, const float &active_vec_f){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> all_r;
        MicroAPI::RegTensor<float> arr_r;
        MicroAPI::RegTensor<float> zeros_r;
        MicroAPI::RegTensor<float> prefix_masked_r;
        MicroAPI::RegTensor<float> total_masked_r;
        MicroAPI::RegTensor<float> prefix_r;
        MicroAPI::RegTensor<float> total_r;
        MicroAPI::MaskReg m_prefix = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg m_active = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(all_r, ws_all_ub, 1, _tmp_maskreg_0);
        MicroAPI::Arange<float, MicroAPI::IndexOrder::INCREASE_ORDER>(arr_r, 0.0f);
        MicroAPI::Duplicate(zeros_r, 0.0f, _tmp_maskreg_0);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(m_prefix, arr_r, my_idx_f, _tmp_maskreg_0);
        MicroAPI::Select(prefix_masked_r, all_r, zeros_r, m_prefix);
        MicroAPI::ReduceSum(prefix_r, prefix_masked_r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(offset_out_ub, prefix_r, _tmp_maskreg_0);
        MicroAPI::CompareScalar<float, CMPMODE::LT>(m_active, arr_r, active_vec_f, _tmp_maskreg_0);
        MicroAPI::Select(total_masked_r, all_r, zeros_r, m_active);
        MicroAPI::ReduceSum(total_r, total_masked_r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(total_out_ub, total_r, _tmp_maskreg_0);
    }
}

