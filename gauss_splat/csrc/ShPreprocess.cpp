/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;

at::Tensor sh_preprocess_fwd(const at::Tensor& camtoworlds, const at::Tensor& means)
{
    TORCH_CHECK(torch_npu::utils::is_npu(camtoworlds) && torch_npu::utils::is_npu(means),
                "sh_preprocess inputs must be NPU tensors");
    TORCH_CHECK(camtoworlds.device() == means.device(), "inputs must be on the same device");
    TORCH_CHECK(camtoworlds.scalar_type() == at::kFloat && means.scalar_type() == at::kFloat,
                "inputs must be float32");
    TORCH_CHECK(camtoworlds.sizes() == at::IntArrayRef({1, 4, 4}),
                "camtoworlds must have shape [1,4,4]");
    TORCH_CHECK(means.dim() == 2 && means.size(0) == 3 && means.size(1) > 0,
                "means must have shape [3,N] with N > 0");

    at::Tensor cameras = camtoworlds.contiguous();
    at::Tensor positions = means.contiguous();
    at::Tensor rays = at::empty({1, 3, means.size(1)}, means.options());
    EXEC_NPU_CMD(aclnnShPreprocess, cameras, positions, rays);
    return rays;
}
