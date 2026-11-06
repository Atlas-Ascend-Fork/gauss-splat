/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
static const int64_t VIEWMATS_DIM = 16;
static const int64_t KS_DIM = 9;
static const int64_t B_IDX = 0;
static const int64_t C_IDX = 1;
static const int64_t N_IDX = 2;
static const int64_t VPW_DIM = 3;
static const int64_t VQUATS_DIM = 4;
static const int64_t VSCALES_DIM = 3;
static const int64_t VR_DIM = 3;
static const int64_t VCOLOR_DIM = 3;
static const int64_t VCOLOR_EXT_DIM = 6;
static const uint32_t SF_DEPTH_CORRECT = (1 << 0);
static const uint32_t SF_PROJ_BACKWARD_CULLING = (1 << 2);
constexpr int32_t OUTPUT_NUM = 10;

void AllocOptionalInputs(
    const at::Tensor &filter, const at::Tensor &means, int32_t support_features,
    c10::optional<at::Tensor> &compensations, c10::optional<at::Tensor> &v_vi_culling,
    c10::optional<at::Tensor> &opacities, c10::optional<at::Tensor> &vi,
    c10::optional<at::Tensor> &sigma, c10::optional<at::Tensor> &depths)
{
#ifdef ASCEND_COMPUTE_UNI_A5
    TORCH_CHECK(filter.scalar_type() == at::kFloat,
                "A5 requires 'filter' to be a FLOAT mask with shape (B, C, N); "
                "the packed UINT8 bitmap is only supported on Atlas A2/A3.");
    const int64_t sliceNum = filter.size(B_IDX) * filter.size(C_IDX);
    const int64_t gaussNum = means.size(N_IDX);
    auto optOptions = means.options();

    const bool hasVi = vi.has_value() && vi->defined();
    const bool hasSigma = sigma.has_value() && sigma->defined();
    const bool hasDepths = depths.has_value() && depths->defined();

    if (!compensations.has_value() || !compensations->defined()) {
        compensations = at::ones({sliceNum, gaussNum}, optOptions);
    }
    if (!v_vi_culling.has_value() || !v_vi_culling->defined()) {
        v_vi_culling = at::zeros({sliceNum, gaussNum}, optOptions);
    }
    if (!opacities.has_value() || !opacities->defined()) {
        opacities = at::ones({sliceNum, gaussNum}, optOptions);
    }
    if (!hasVi) {
        vi = at::ones({sliceNum, gaussNum}, optOptions);
    }
    if (!hasDepths) {
        depths = at::zeros({sliceNum, gaussNum}, optOptions);
    }
    if ((support_features & SF_DEPTH_CORRECT) != 0) {
        TORCH_CHECK(hasVi && hasSigma && hasDepths,
                    "vi, sigma and depths are required when depth correction is enabled.");
    }
    if (!hasSigma) {
        sigma = at::ones({1}, optOptions);
    }
#endif
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fully_fused_projection_bwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales, const at::Tensor &conics,
    const at::Tensor &viewmats, const at::Tensor &Ks, const at::Tensor &v_means2d, const at::Tensor &v_depths,
    const at::Tensor &v_conics, const at::Tensor &v_colors_culling, const at::Tensor &v_opacities_culling,
    const at::Tensor &filter, const c10::optional<at::Tensor> &compensations,
    const c10::optional<at::Tensor> &v_vi_culling, const c10::optional<at::Tensor> &opacities,
    const c10::optional<at::Tensor> &vi, const c10::optional<at::Tensor> &sigma,
    const c10::optional<at::Tensor> &depths, const c10::optional<at::Tensor> &dense_indices, const c10::optional<at::Tensor> &cnt,
    int width, int height, int32_t max_compact_count, int32_t support_features, const std::vector<at::Tensor> &outputs)
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

    TORCH_CHECK(means.dim() == MEANS_DIM, "means's dim should be 3.");
    TORCH_CHECK(quats.dim() == QUATS_DIM, "quats's dim should be 3.");
    TORCH_CHECK(scales.dim() == SCALES_DIM, "scales's dim should be 3.");
    TORCH_CHECK(filter.dim() == FILTER_DIM, "filter's dim should be 3.");
    TORCH_CHECK(outputs.size() == OUTPUT_NUM, "output number is not 10.");

    // reshape the culling data from [B, C, ...] --> [BC, ...], so the dimension is different.
    bool proj_backword_culling = ((support_features & SF_PROJ_BACKWARD_CULLING) != 0);
    if (proj_backword_culling) {
        TORCH_CHECK(conics.dim() == 3, "conics's dim should be 3.");
        TORCH_CHECK(v_means2d.dim() == 3, "v_means2d's dim should be 3.");
        TORCH_CHECK(v_depths.dim() == 2, "v_depths's dim should be 2.");
        TORCH_CHECK(v_conics.dim() == 3, "v_conics's dim should be 3.");
        TORCH_CHECK(v_colors_culling.dim() == 3, "v_colors_culling's dim should be 3.");
        TORCH_CHECK(v_opacities_culling.dim() == 2, "v_opacities_culling's dim should be 2.");
    } else {
        TORCH_CHECK(conics.dim() == CONICS_DIM, "conics's dim should be 4.");
        TORCH_CHECK(v_means2d.dim() == VMEANS2D_DIM, "v_means2d's dim should be 4.");
        TORCH_CHECK(v_depths.dim() == VDEPTHS_DIM, "v_depths's dim should be 3.");
        TORCH_CHECK(v_conics.dim() == VCONICS_DIM, "v_conics's dim should be 4.");
        TORCH_CHECK(v_colors_culling.dim() == VCOLORSCULLING_DIM, "v_colors_culling's dim should be 4.");
        TORCH_CHECK(v_opacities_culling.dim() == VOPACITIESCULLING_DIM, "v_opacities_culling's dim should be 3.");
    }

    at::Tensor v_pW = outputs[0];
    at::Tensor v_quats = outputs[1];
    at::Tensor v_scales = outputs[2];
    at::Tensor v_R = outputs[3];
    at::Tensor v_colors = outputs[4];
    at::Tensor v_opacities = outputs[5];
    at::Tensor v_viewmats = outputs[6];
    at::Tensor v_ks = outputs[7];
    at::Tensor dL_dvi = outputs[8];
    at::Tensor dL_dsigma = outputs[9];

    // Optional inputs stay untouched on arch22 (its tiling records their presence and
    // the kernel branches accordingly). The A5 kernel has no presence branch and reads
    // all of them unconditionally, so materialize neutral defaults here (front-end) to
    // guarantee it never sees a nullptr.
    c10::optional<at::Tensor> compensationsArg = compensations;
    c10::optional<at::Tensor> vViCullingArg = v_vi_culling;
    c10::optional<at::Tensor> opacitiesArg = opacities;
    c10::optional<at::Tensor> viArg = vi;
    c10::optional<at::Tensor> sigmaArg = sigma;
    c10::optional<at::Tensor> depthsArg = depths;

    AllocOptionalInputs(filter, means, support_features, compensationsArg, vViCullingArg, opacitiesArg, viArg,
                        sigmaArg, depthsArg);

    EXEC_NPU_CMD(aclnnFullyFusedProjectionBwd, means, quats, scales, conics, viewmats, Ks, v_means2d, v_depths,
                 v_conics, v_colors_culling, v_opacities_culling, filter, compensationsArg, vViCullingArg, opacitiesArg,
                 viArg, sigmaArg, depthsArg, dense_indices, cnt, width, height, max_compact_count,
                 support_features, v_pW, v_quats, v_scales, v_R, v_colors, v_opacities, v_viewmats, v_ks, dL_dvi,
                 dL_dsigma);

    return std::tie(v_pW, v_quats, v_scales, v_R, v_colors, v_opacities, v_viewmats, v_ks, dL_dvi, dL_dsigma);
}
