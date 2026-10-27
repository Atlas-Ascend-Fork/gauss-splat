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
#include "functions.h"
#include "OpApiCommon.h"

namespace {
    constexpr int32_t PINHOLE_TYPE = 0;
    constexpr int32_t ORTHO_TYPE = 1;
    constexpr int32_t FISHEYE_TYPE = 2;
    constexpr int32_t OUTPUT_NUM = 7;
}

using namespace NPU_NAME_SPACE;
using namespace std;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> projection_three_dims_gaussian_forward(
    at::Tensor& means, const c10::optional<at::Tensor>& covars,
    at::Tensor& opacities, at::Tensor& viewmats, at::Tensor& ks,
    const c10::optional<at::Tensor>& quat, const c10::optional<at::Tensor>& scales,
    int32_t width, int32_t height, double eps,
    bool calc_compensations, std::string camera_model, const std::vector<at::Tensor> &outputs)
{
    TORCH_CHECK(means.scalar_type() == at::kFloat,
        "means: float32 tensor expected but got a tensor with dtype: ", means.scalar_type());
    TORCH_CHECK(viewmats.scalar_type() == at::kFloat,
        "viewmats: float32 tensor expected but got a tensor with dtype: ", viewmats.scalar_type());
    TORCH_CHECK(ks.scalar_type() == at::kFloat,
        "ks: float32 tensor expected but got a tensor with dtype: ", ks.scalar_type());
    TORCH_CHECK(covars.has_value() != (quat.has_value() && scales.has_value()),
        "projection_three_dims_gaussian_forward requires either 'covars' or the "
        "('quat', 'scales') pair, but not both.");
    if (covars.has_value()) {
        TORCH_CHECK(covars->scalar_type() == at::kFloat,
            "covars: float32 tensor expected but got a tensor with dtype: ", covars->scalar_type());
    }
    if (quat.has_value()) {
        TORCH_CHECK(quat->scalar_type() == at::kFloat,
            "quat: float32 tensor expected but got a tensor with dtype: ", quat->scalar_type());
        TORCH_CHECK(scales->scalar_type() == at::kFloat,
            "scales: float32 tensor expected but got a tensor with dtype: ", scales->scalar_type());
    }
    TORCH_CHECK(outputs.size() == OUTPUT_NUM, "output number is not 6.");
    
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

    at::Tensor means2d = outputs[0];
    at::Tensor depths = outputs[1];
    at::Tensor conics = outputs[2];
    at::Tensor compensations = outputs[3];
    at::Tensor det = outputs[4];
    at::Tensor radius = outputs[5];
    at::Tensor covars2d = outputs[6];

    // The aclnn signature accepts either precomputed covars or the (quat,
    // scales) pair. Callers on arch35 pass quat/scales so the native op fuses
    // the covariance conversion, while other targets pass covars directly.
    EXEC_NPU_CMD(aclnnProjectionThreeDimsGaussianForward, means, covars, viewmats, ks,
                 quat, scales, width, height, eps, calc_compensations, cameraType,
                 means2d, depths, conics, compensations, det, radius, covars2d);
    
    return std::tie(means2d, depths, conics, compensations, det, radius, covars2d);
}
