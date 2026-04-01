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

namespace {
    constexpr int32_t PINHOLE_TYPE = 0;
    constexpr int32_t ORTHO_TYPE = 1;
    constexpr int32_t FISHEYE_TYPE = 2;
}

using namespace NPU_NAME_SPACE;
using namespace std;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_three_dims_gaussian_forward(
    at::Tensor& means, at::Tensor& covars, at::Tensor& opacities,
    at::Tensor& viewmats, at::Tensor& ks,
    int32_t width, int32_t height, double eps,
    bool calc_compensations, std::string camera_model)
{
    TORCH_CHECK(means.scalar_type() == at::kFloat,
        "means: float32 tensor expected but got a tensor with dtype: ", means.scalar_type());
    TORCH_CHECK(viewmats.scalar_type() == at::kFloat,
        "viewmats: float32 tensor expected but got a tensor with dtype: ", viewmats.scalar_type());
    TORCH_CHECK(ks.scalar_type() == at::kFloat,
        "ks: float32 tensor expected but got a tensor with dtype: ", ks.scalar_type());
    TORCH_CHECK(covars.scalar_type() == at::kFloat,
        "covars: float32 tensor expected but got a tensor with dtype: ", covars.scalar_type());
    
    int32_t cameraType;
    std::string upperCameraModel = camera_model;
    std::transform(upperCameraModel.begin(), upperCameraModel.end(), upperCameraModel.begin(), ::toupper);

    if (upperCameraModel == "PINHOLE") {
        cameraType = PINHOLE_TYPE;
    } else if (upperCameraModel == "ORTHO") {
        cameraType = ORTHO_TYPE;
    } else if (upperCameraModel == "FISHEYE") {
        cameraType = FISHEYE_TYPE;
    } else {
        AT_ERROR("Unsupported camera model: ", camera_model, ". Supported models: PINHOLE, ORTHO, FISHEYE");
    }

    auto batchSize = means.sizes()[0];
    auto gaussianNum = means.sizes()[2];
    auto cameraNum = viewmats.sizes()[1];

    at::Tensor means2d = at::zeros({batchSize, cameraNum, 2, gaussianNum}, means.options().dtype(at::kFloat));
    at::Tensor depths = at::zeros({batchSize, cameraNum, 1, gaussianNum}, means.options().dtype(at::kFloat));
    at::Tensor conics = at::zeros({batchSize, cameraNum, 3, gaussianNum}, means.options().dtype(at::kFloat));
    at::Tensor compensations = at::zeros({batchSize, cameraNum, 1, gaussianNum}, means.options().dtype(at::kFloat));
    at::Tensor det = at::zeros({batchSize, cameraNum, 1, gaussianNum}, means.options().dtype(at::kFloat));
    at::Tensor radius = at::zeros({batchSize, cameraNum, 2, gaussianNum}, means.options().dtype(at::kFloat));
    at::Tensor covars2d = at::zeros({batchSize, cameraNum, 3, gaussianNum}, means.options().dtype(at::kFloat));

    EXEC_NPU_CMD(aclnnProjectionThreeDimsGaussianForward, means, covars, viewmats, ks,
                 width, height, eps, calc_compensations, cameraType,
                 means2d, depths, conics, compensations, det, radius, covars2d);
    
    return std::tie(means2d, depths, conics, compensations, det, radius, covars2d);
}
