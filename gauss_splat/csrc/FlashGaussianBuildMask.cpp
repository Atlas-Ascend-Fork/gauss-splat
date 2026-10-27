/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>

#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

namespace {
// gather_mask is a per-core scratch: one row per active AIV. Query the real
// platform vector-core count at runtime (asc-devkit grid_config.asc pattern:
// aclrtGetDeviceInfo + ACL_DEV_ATTR_VECTOR_CORE_NUM) instead of hardcoding the
// core count, and cache it per device. Falls back to a safe upper bound when
// the query fails; tiling still clamps dispatch to min(AIV, scratch rows).
constexpr uint32_t BLOCK_NUM_FALLBACK = 64;
constexpr uint32_t GATHER_NUM = 20;

uint32_t GetBlockNum(int32_t deviceId)
{
#ifdef ASCEND_COMPUTE_UNI_A5
    static std::mutex cacheMutex;
    static std::unordered_map<int32_t, uint32_t> cache;
    const std::lock_guard<std::mutex> lock(cacheMutex);
    const auto cached = cache.find(deviceId);
    if (cached != cache.end()) {
        return cached->second;
    }
    int64_t coreNum = 0;
    const aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceId), ACL_DEV_ATTR_VECTOR_CORE_NUM, &coreNum);
    const uint32_t blockNum =
        (ret == ACL_SUCCESS && coreNum > 0) ? static_cast<uint32_t>(coreNum) : BLOCK_NUM_FALLBACK;
    cache.emplace(deviceId, blockNum);
    return blockNum;
#else
    constexpr uint32_t BLOCK_NUM = 48;
    return BLOCK_NUM;
#endif
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> flash_gaussian_build_mask(
    at::Tensor& means2d, at::Tensor& opacity, at::Tensor& conics, at::Tensor& covars2d, at::Tensor& depths,
    at::Tensor& cnt, at::Tensor& tile_grid, double image_width, double image_height, int32_t tile_size)
{
    TORCH_CHECK(torch_npu::utils::is_npu(means2d), "means2d must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(cnt), "cnt must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(opacity), "opacity must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(conics), "conics must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(covars2d), "covars2d must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(depths), "depths must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(tile_grid), "tile_grid must be NPU tensor");

    TORCH_CHECK(means2d.scalar_type() == at::kFloat,
                "means2d: float32 tensor expected but got a tensor with dtype: ", means2d.scalar_type());
    TORCH_CHECK(cnt.scalar_type() == at::kInt,
                "cnt: int32 tensor expected but got a tensor with dtype: ", cnt.scalar_type());
    TORCH_CHECK(opacity.scalar_type() == at::kFloat,
                "opacity: float32 tensor expected but got a tensor with dtype: ", opacity.scalar_type());
    TORCH_CHECK(conics.scalar_type() == at::kFloat,
                "conics: float32 tensor expected but got a tensor with dtype: ", conics.scalar_type());
    TORCH_CHECK(covars2d.scalar_type() == at::kFloat,
                "covars2d: float32 tensor expected but got a tensor with dtype: ", covars2d.scalar_type());
    TORCH_CHECK(tile_grid.scalar_type() == at::kFloat,
                "tile_grid: float32 tensor expected but got a tensor with dtype: ", tile_grid.scalar_type());
    TORCH_CHECK(depths.scalar_type() == at::kFloat,
                "depths: float32 tensor expected but got a tensor with dtype: ", depths.scalar_type());

    uint32_t batch_size = opacity.sizes()[0];
    uint32_t camera_num = opacity.sizes()[1];
    uint32_t gaussian_num = opacity.sizes()[3];
    uint32_t tile_num = tile_grid.sizes()[0];

    at::Tensor tile_sum = at::empty({batch_size, camera_num, tile_num, 1}, opacity.options().dtype(at::kInt));
    at::Tensor tile_depths =
        at::empty({batch_size, camera_num, tile_num, gaussian_num}, opacity.options().dtype(at::kFloat));
    const uint32_t blockNum = GetBlockNum(static_cast<int32_t>(opacity.get_device()));
    at::Tensor gather_mask = at::empty({blockNum, GATHER_NUM, gaussian_num}, opacity.options().dtype(at::kFloat));
    at::Tensor gauss_index =
        at::empty({batch_size, camera_num, tile_num, gaussian_num}, opacity.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnFlashGaussianBuildMask, means2d, opacity, conics, covars2d, depths, cnt, tile_grid, gather_mask, image_width,
                 image_height, tile_size, tile_sum, tile_depths, gauss_index);
    
    at::Tensor tile_offset = at::cumsum(tile_sum, 2, at::kInt);
    return std::tie(tile_sum, tile_offset, tile_depths, gauss_index);
}