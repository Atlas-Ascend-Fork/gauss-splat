/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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