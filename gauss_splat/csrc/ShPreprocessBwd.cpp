/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;

at::Tensor sh_preprocess_bwd(const at::Tensor& means, const at::Tensor& camtoworlds,
                             const at::Tensor& rays_d, const at::Tensor& v_rays_d)
{
    TORCH_CHECK(torch_npu::utils::is_npu(means) && torch_npu::utils::is_npu(camtoworlds) &&
                torch_npu::utils::is_npu(rays_d) && torch_npu::utils::is_npu(v_rays_d),
                "sh_preprocess_bwd inputs must be NPU tensors");
    TORCH_CHECK(means.device() == camtoworlds.device() && means.device() == rays_d.device() &&
                means.device() == v_rays_d.device(), "inputs must be on the same device");
    TORCH_CHECK(means.scalar_type() == at::kFloat && camtoworlds.scalar_type() == at::kFloat &&
                rays_d.scalar_type() == at::kFloat && v_rays_d.scalar_type() == at::kFloat,
                "inputs must be float32");
    TORCH_CHECK(means.dim() == 2 && means.size(0) == 3 && means.size(1) > 0,
                "means must have shape [3,N] with N > 0");
    TORCH_CHECK(camtoworlds.sizes() == at::IntArrayRef({1, 4, 4}),
                "camtoworlds must have shape [1,4,4]");
    TORCH_CHECK(rays_d.sizes() == at::IntArrayRef({1, 3, means.size(1)}) &&
                v_rays_d.sizes() == rays_d.sizes(), "rays and their gradient must have shape [1,3,N]");

    at::Tensor positions = means.contiguous();
    at::Tensor cameras = camtoworlds.contiguous();
    at::Tensor rays = rays_d.contiguous();
    at::Tensor v_rays = v_rays_d.contiguous();
    at::Tensor v_means = at::empty_like(positions);
    EXEC_NPU_CMD(aclnnShPreprocessBwd, positions, cameras, rays, v_rays, v_means);
    return v_means;
}
