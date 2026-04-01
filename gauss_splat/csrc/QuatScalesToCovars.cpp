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
#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;
using namespace std;

at::Tensor quat_scales_to_covars(at::Tensor& quat, at::Tensor& scales)
{
    TORCH_CHECK(torch_npu::utils::is_npu(quat), "quat must be NPU tensor");
    TORCH_CHECK(torch_npu::utils::is_npu(scales), "scales must be NPU tensor");
    TORCH_CHECK(quat.scalar_type() == at::kFloat,
        "quat: float32 tensor expected but got a tensor with dtype: ", quat.scalar_type());
    TORCH_CHECK(scales.scalar_type() == at::kFloat,
        "scales: float32 tensor expected but got a tensor with dtype: ", scales.scalar_type());

    auto batch_size = quat.sizes()[0];
    auto gaussian_num = quat.sizes()[2];

    at::Tensor covars = at::zeros({batch_size, 3, 3, gaussian_num}, quat.options().dtype(at::kFloat));
    EXEC_NPU_CMD(aclnnQuatScalesToCovars, quat, scales, covars);
    return covars;
}