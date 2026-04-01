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
 
/*!
 * \file GaussianSort.cpp
 * \brief gaussian sort pybind adapter
 */

#include <string>

#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

std::tuple<at::Tensor, at::Tensor> gaussian_sort(const at::Tensor& all_in_mask, const at::Tensor& depths)
{
    TORCH_CHECK(all_in_mask.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(depths.device() == all_in_mask.device(), "Inconsistent device.");
    TORCH_CHECK(depths.sizes() == all_in_mask.sizes()[1], "Invalid shape.");
    TORCH_CHECK(all_in_mask.scalar_type() == at::kFloat,
                "all_in_mask: float32 tensor expected but got a tensor with dtype: ", all_in_mask.scalar_type());
    TORCH_CHECK(depths.scalar_type() == at::kFloat,
                "depths: float32 tensor expected but got a tensor with dtype: ", depths.scalar_type());

    auto device = all_in_mask.device();
    uint32_t tileNum = all_in_mask.sizes()[0];    // rows
    uint32_t nGaussNum = all_in_mask.sizes()[1];  // cols
    auto intOptions = at::TensorOptions().dtype(at::kInt).layout(at::kStrided).device(device);
    auto tile_sums = at::sum(all_in_mask, 1, false, at::kInt);
    TORCH_CHECK(at::all(tile_sums >= 0).item<bool>(), "tile_sums has negative! Check tile_sums.");
    auto tile_offsets = at::cumsum(tile_sums, 0, at::kInt);
    int32_t maskGauss = tile_offsets.index({-1}).item<int32_t>();
    at::Tensor sorted_gs_ids = at::empty({maskGauss}, intOptions);
    EXEC_NPU_CMD(aclnnGaussianSort, all_in_mask, tile_sums, tile_offsets, depths, sorted_gs_ids);

    return std::tie(sorted_gs_ids, tile_offsets);
}
