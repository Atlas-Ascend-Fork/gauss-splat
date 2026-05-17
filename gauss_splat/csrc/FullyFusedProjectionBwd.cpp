/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include "OpApiCommon.h"
#include "functions.h"

using namespace NPU_NAME_SPACE;
using namespace std;

namespace {
static const int64_t MEANS_DIM = 3;
static const int64_t QUATS_DIM = 3;
static const int64_t SCALES_DIM = 3;
static const int64_t CONICS_DIM = 4;
static const int64_t VMEANS2D_DIM = 4;
static const int64_t VDEPTHS_DIM = 3;
static const int64_t VCONICS_DIM = 4;
static const int64_t VCOLORSCULLING_DIM = 4;
static const int64_t VOPACITIESCULLING_DIM = 3;
static const int64_t FILTER_DIM = 3;
static const int64_t B_IDX = 0;
static const int64_t C_IDX = 1;
static const int64_t N_IDX = 2;
static const int64_t VPW_DIM = 3;
static const int64_t VQUATS_DIM = 4;
static const int64_t VSCALES_DIM = 3;
static const int64_t VR_DIM = 3;
static const int64_t VCOLOR_DIM = 3;
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> fully_fused_projection_bwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales, const at::Tensor &conics,
    const at::Tensor &viewmats, const at::Tensor &Ks, const at::Tensor &v_means2d, const at::Tensor &v_depths,
    const at::Tensor &v_conics, const at::Tensor &v_colors_culling, const at::Tensor &v_opacities_culling,
    const at::Tensor &filter, const c10::optional<at::Tensor> &compensations, int width, int height)
{
    TORCH_CHECK(means.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(quats.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(scales.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(conics.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(viewmats.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(Ks.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(v_means2d.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(v_depths.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(v_conics.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(v_colors_culling.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(v_opacities_culling.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(filter.device() == means.device(), "Inconsistent device.");

    auto device = means.device();

    TORCH_CHECK(means.dim() == MEANS_DIM, "means's dim should be 3.");
    TORCH_CHECK(quats.dim() == QUATS_DIM, "quats's dim should be 3.");
    TORCH_CHECK(scales.dim() == SCALES_DIM, "scales's dim should be 3.");
    TORCH_CHECK(conics.dim() == CONICS_DIM, "conics's dim should be 4.");
    TORCH_CHECK(v_means2d.dim() == VMEANS2D_DIM, "v_means2d's dim should be 4.");
    TORCH_CHECK(v_depths.dim() == VDEPTHS_DIM, "v_depths's dim should be 3.");
    TORCH_CHECK(v_conics.dim() == VCONICS_DIM, "v_conics's dim should be 4.");
    TORCH_CHECK(v_colors_culling.dim() == VCOLORSCULLING_DIM, "v_colors_culling's dim should be 4.");
    TORCH_CHECK(v_opacities_culling.dim() == VOPACITIESCULLING_DIM, "v_opacities_culling's dim should be 3.");
    TORCH_CHECK(filter.dim() == FILTER_DIM, "filter's dim should be 3.");

    int64_t batchSize = means.sizes()[B_IDX];
    int64_t gaussianNum = means.sizes()[N_IDX];
    int64_t cameraNum = conics.sizes()[C_IDX];

    at::Tensor v_pW = at::zeros({batchSize, gaussianNum, VPW_DIM}, means.options().dtype(at::kFloat)).contiguous();
    at::Tensor v_quats = at::zeros({batchSize, gaussianNum, VQUATS_DIM},
        means.options().dtype(at::kFloat)).contiguous();
    at::Tensor v_scales = at::zeros({batchSize, gaussianNum, VSCALES_DIM},
        means.options().dtype(at::kFloat)).contiguous();
    at::Tensor v_R = at::zeros({batchSize, cameraNum, VR_DIM, VR_DIM},
        means.options().dtype(at::kFloat)).contiguous();
    at::Tensor v_colors = at::zeros({batchSize, VCOLOR_DIM, gaussianNum},
        means.options().dtype(at::kFloat)).contiguous();
    at::Tensor v_opacities = at::zeros({batchSize, gaussianNum}, means.options().dtype(at::kFloat)).contiguous();

    EXEC_NPU_CMD(aclnnFullyFusedProjectionBwd, means, quats, scales, conics, viewmats, Ks, v_means2d, v_depths,
                 v_conics, v_colors_culling, v_opacities_culling, filter, compensations, width, height, v_pW, v_quats,
                 v_scales, v_R, v_colors, v_opacities);
    return std::tie(v_pW, v_quats, v_scales, v_R, v_colors, v_opacities);
}