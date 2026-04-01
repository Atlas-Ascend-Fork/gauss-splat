/**
 * Adapted from
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) 2019, Facebook CORPORATION.
 
 * All rights reserved.
 * Licensed under the BSD 3-Clause License  (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
#include <string>
#include <cmath>
#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

at::Tensor flash_gaussian_build_mask(at::Tensor& means2d, at::Tensor& opacity, at::Tensor& conics,
                                     at::Tensor& covars2d, at::Tensor& cnt, at::Tensor& tile_grid,
                                     double image_width, double image_height, int32_t tile_size)
{
    TORCH_CHECK(torch_npu::utils::is_npu(means2d), "means2d must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(cnt), "cnt must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(opacity), "opacity must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(conics), "conics must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(covars2d), "covars2d must be NPU tensor");
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

    uint32_t batchSize = opacity.sizes()[0];
    uint32_t cameraNum = opacity.sizes()[1];
    uint32_t gaussianNum = opacity.sizes()[3];
    uint32_t tileNum = tile_grid.sizes()[0];
    at::Tensor mask = at::zeros({batchSize, cameraNum, tileNum, gaussianNum}, opacity.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnFlashGaussianBuildMask, means2d, opacity, conics, covars2d,
                 cnt, tile_grid, image_width, image_height, tile_size, mask);
    return mask;
}