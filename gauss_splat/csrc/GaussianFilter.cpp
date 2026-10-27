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
static const int64_t COLORS_DIM = 3;
static const int64_t COLORS_DIM_EXT = 7;
static const int64_t DET_DIM = 3;
static const int64_t OPACITIES_DIM = 2;
static const int64_t MEANS2D_DIM = 4;
static const int64_t MEANS2DCULLING_DIM = 2;

static const int64_t DEPTH_DIM = 3;
static const int64_t RADIUS_DIM = 4;
static const int64_t RADIUSCULLING_DIM = 2;

static const int64_t CONICS_DIM = 4;
static const int64_t CONICSCULLING_DIM = 3;

static const int64_t COVARS2D_DIM = 4;
static const int64_t COVARS2DCULLING_DIM = 3;
static const int64_t FILTER_ALIGNPAD = 7;
static const int64_t FILTER_ALIGN = 8;

constexpr int32_t OUTPUT_NUM = 16;
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
gaussian_filter(at::Tensor &means, at::Tensor &colors, at::Tensor &det, at::Tensor &opacities, at::Tensor &means2d,
                at::Tensor &depths, at::Tensor &radius, at::Tensor &conics, at::Tensor &covars2d,
                const c10::optional<at::Tensor> &compensations, const c10::optional<at::Tensor> &quat,
                const c10::optional<at::Tensor> &scales, const c10::optional<at::Tensor> &vi,
                const c10::optional<at::Tensor> &sigma, int width, int height, double near_plane, double far_plane, int support_features, const std::vector<at::Tensor> &outputs)
{
    TORCH_CHECK(means.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(colors.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(det.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(opacities.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(means2d.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(depths.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(radius.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(conics.device() == means.device(), "Inconsistent device.");
    TORCH_CHECK(covars2d.device() == means.device(), "Inconsistent device.");

    TORCH_CHECK(means.dim() == MEANS_DIM, "means's dim should be 3.");
    TORCH_CHECK(colors.dim() == COLORS_DIM, "colors's dim should be 3.");
    TORCH_CHECK(det.dim() == DET_DIM, "det's dim should be 3.");
    TORCH_CHECK(opacities.dim() == OPACITIES_DIM, "opacities's dim should be 2.");
    TORCH_CHECK(means2d.dim() == MEANS2D_DIM, "means2d's dim should be 4.");
    TORCH_CHECK(depths.dim() == DEPTH_DIM, "depths's dim should be 3.");
    TORCH_CHECK(radius.dim() == RADIUS_DIM, "radius's dim should be 4.");
    TORCH_CHECK(conics.dim() == CONICS_DIM, "conics's dim should be 4.");
    TORCH_CHECK(covars2d.dim() == COVARS2D_DIM, "covars2d's dim should be 4.");
    TORCH_CHECK(outputs.size() == OUTPUT_NUM, "output number is not 16.");

    at::Tensor meansCulling = outputs[0];
    at::Tensor colorsCulling = outputs[1];
    at::Tensor means2dCulling = outputs[2];
    at::Tensor depthsCulling = outputs[3];
    at::Tensor radiusCulling = outputs[4];
    at::Tensor covars2dCulling = outputs[5];
    at::Tensor conicsCulling = outputs[6];
    at::Tensor opacitiesCulling = outputs[7];
    at::Tensor quatsCulling = outputs[8];
    at::Tensor scalesCulling = outputs[9];
    at::Tensor viCulling = outputs[10];
    at::Tensor compensationsCulling = outputs[11];
    at::Tensor filter = outputs[12];
    at::Tensor cntFloat = outputs[13];
    at::Tensor filterFloat = outputs[14];
    at::Tensor denseIndices = outputs[15];

    EXEC_NPU_CMD(aclnnGaussianFilter, means, colors, det, opacities, means2d, depths, radius, conics, covars2d,
                compensations, quat, scales, vi, sigma, width, height, near_plane, far_plane, support_features,
                meansCulling, colorsCulling, means2dCulling, depthsCulling, radiusCulling,
                covars2dCulling, conicsCulling, opacitiesCulling, quatsCulling, scalesCulling,
                viCulling, compensationsCulling, filter, cntFloat, filterFloat, denseIndices);

    return std::tie(meansCulling, colorsCulling, means2dCulling, depthsCulling, radiusCulling, covars2dCulling, conicsCulling, opacitiesCulling,
                    quatsCulling, scalesCulling, viCulling, compensationsCulling, filter, cntFloat, filterFloat, denseIndices);
}
