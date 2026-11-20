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

__aicore__ inline void _reduce_camera_grad_vf(__ubuf__ float* camera_grad_ub){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> r;

        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 1 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 1 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 2 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 2 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 3 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 3 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 4 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 4 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 5 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 5 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 6 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 6 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 7 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 7 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 8 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 8 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 9 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 9 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 10 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 10 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 11 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 11 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 12 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 12 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 13 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 13 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 14 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 14 * 64, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(r, camera_grad_ub + 15 * 64, 1, _tmp_maskreg_0);
        MicroAPI::ReduceSum(r, r, _tmp_maskreg_0);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(camera_grad_ub + 15 * 64, r, _tmp_maskreg_0);
    }
}
