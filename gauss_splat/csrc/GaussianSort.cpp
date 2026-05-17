/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file GaussianSort.cpp
 * \brief gaussian sort pybind adapter
 */

#include <string>

#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

at::Tensor gaussian_sort(const at::Tensor& lb_sched, const at::Tensor& gaussian_cnt, const at::Tensor& depths,
                         const at::Tensor& gs_ids, const at::Tensor& sorted_offset, int32_t max_tile_gauss)
{
    TORCH_CHECK(depths.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(depths.device() == lb_sched.device(), "Inconsistent device.");
    TORCH_CHECK(depths.sizes() == gs_ids.sizes(), "Invalid shape.");
    TORCH_CHECK(depths.scalar_type() == at::kFloat,
                "depths: float32 tensor expected but got a tensor with dtype: ", depths.scalar_type());

    auto device = depths.device();
    auto options = at::TensorOptions().dtype(at::kInt).layout(at::kStrided).device(device);
    int64_t sorted_total_nums = sorted_offset.index({-1}).item<int64_t>();
    at::Tensor sorted_gs_ids = at::empty({sorted_total_nums}, options);
    EXEC_NPU_CMD(aclnnGaussianSort, lb_sched, gaussian_cnt, depths, gs_ids, sorted_offset, max_tile_gauss,
                 sorted_gs_ids);

    return sorted_gs_ids;
}
