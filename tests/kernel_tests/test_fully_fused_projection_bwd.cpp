/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_fully_fused_projection_bwd.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* meansDeviceAddr = nullptr;
    void* quatsDeviceAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* conicsDeviceAddr = nullptr;
    void* viewmatsDeviceAddr = nullptr;
    void* ksDeviceAddr = nullptr;
    void* v_means2dDeviceAddr = nullptr;
    void* v_depthsDeviceAddr = nullptr;
    void* v_conicsDeviceAddr = nullptr;
    void* v_colors_cullingDeviceAddr = nullptr;
    void* v_opacities_cullingDeviceAddr = nullptr;
    void* filterDeviceAddr = nullptr;
    void* v_pWDeviceAddr = nullptr;
    void* v_quatsDeviceAddr = nullptr;
    void* v_scalesDeviceAddr = nullptr;
    void* v_RDeviceAddr = nullptr;
    void* v_colorsDeviceAddr = nullptr;
    void* v_opacitiesDeviceAddr = nullptr;
    void* v_viewmatsDeviceAddr = nullptr;
    void* v_ksDeviceAddr = nullptr;
    void* dL_dviDeviceAddr = nullptr;
    void* dL_dsigmaDeviceAddr = nullptr;

    aclTensor* meansTensor = nullptr;
    aclTensor* quatsTensor = nullptr;
    aclTensor* scalesTensor = nullptr;
    aclTensor* conicsTensor = nullptr;
    aclTensor* viewmatsTensor = nullptr;
    aclTensor* ksTensor = nullptr;
    aclTensor* v_means2dTensor = nullptr;
    aclTensor* v_depthsTensor = nullptr;
    aclTensor* v_conicsTensor = nullptr;
    aclTensor* v_colors_cullingTensor = nullptr;
    aclTensor* v_opacities_cullingTensor = nullptr;
    aclTensor* filterTensor = nullptr;
    aclTensor* v_pWTensor = nullptr;
    aclTensor* v_quatsTensor = nullptr;
    aclTensor* v_scalesTensor = nullptr;
    aclTensor* v_RTensor = nullptr;
    aclTensor* v_colorsTensor = nullptr;
    aclTensor* v_opacitiesTensor = nullptr;
    aclTensor* v_viewmatsTensor = nullptr;
    aclTensor* v_ksTensor = nullptr;
    aclTensor* dL_dviTensor = nullptr;
    aclTensor* dL_dsigmaTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t cameraNum = 1;
    int64_t gaussNum = 256;
    
    std::vector<int64_t> meansShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> quatsShape = {batchSize, 4, gaussNum};
    std::vector<int64_t> scalesShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> conicsShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> viewmatsShape = {cameraNum, 4, 4};
    std::vector<int64_t> ksShape = {cameraNum, 3, 3};
    std::vector<int64_t> v_means2dShape = {batchSize, cameraNum, 2, gaussNum};
    std::vector<int64_t> v_depthsShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> v_conicsShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> v_colors_cullingShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> v_opacities_cullingShape = {batchSize, gaussNum};
    std::vector<int64_t> filterShape = {batchSize, cameraNum, gaussNum};
    std::vector<int64_t> v_pWShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> v_quatsShape = {batchSize, 4, gaussNum};
    std::vector<int64_t> v_scalesShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> v_RShape = {batchSize, 3, 3, gaussNum};
    std::vector<int64_t> v_colorsShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> v_opacitiesShape = {batchSize, gaussNum};
    std::vector<int64_t> v_viewmatsShape = {batchSize, cameraNum, 16};
    std::vector<int64_t> v_ksShape = {batchSize, cameraNum, 9};
    std::vector<int64_t> dL_dviShape = {batchSize * cameraNum, gaussNum};
    std::vector<int64_t> dL_dsigmaShape = {batchSize * cameraNum, gaussNum};

    std::vector<float> meansHostData(GetShapeSize(meansShape), 0.0f);
    std::vector<float> quatsHostData(GetShapeSize(quatsShape), 0.707f);
    std::vector<float> scalesHostData(GetShapeSize(scalesShape), 1.0f);
    std::vector<float> conicsHostData(GetShapeSize(conicsShape), 0.1f);
    std::vector<float> viewmatsHostData(GetShapeSize(viewmatsShape), 1.0f);
    std::vector<float> ksHostData(GetShapeSize(ksShape), 500.0f);
    std::vector<float> v_means2dHostData(GetShapeSize(v_means2dShape), 1.0f);
    std::vector<float> v_depthsHostData(GetShapeSize(v_depthsShape), 1.0f);
    std::vector<float> v_conicsHostData(GetShapeSize(v_conicsShape), 1.0f);
    std::vector<float> v_colors_cullingHostData(GetShapeSize(v_colors_cullingShape), 1.0f);
    std::vector<float> v_opacities_cullingHostData(GetShapeSize(v_opacities_cullingShape), 1.0f);
    std::vector<uint8_t> filterHostData(GetShapeSize(filterShape), 1);
    std::vector<float> v_pWHostData(GetShapeSize(v_pWShape), 0.0f);
    std::vector<float> v_quatsHostData(GetShapeSize(v_quatsShape), 0.0f);
    std::vector<float> v_scalesHostData(GetShapeSize(v_scalesShape), 0.0f);
    std::vector<float> v_RHostData(GetShapeSize(v_RShape), 0.0f);
    std::vector<float> v_colorsHostData(GetShapeSize(v_colorsShape), 0.0f);
    std::vector<float> v_opacitiesHostData(GetShapeSize(v_opacitiesShape), 0.0f);
    std::vector<float> v_viewmatsHostData(GetShapeSize(v_viewmatsShape), 0.0f);
    std::vector<float> v_ksHostData(GetShapeSize(v_ksShape), 0.0f);
    std::vector<float> dL_dviHostData(GetShapeSize(dL_dviShape), 0.0f);
    std::vector<float> dL_dsigmaHostData(GetShapeSize(dL_dsigmaShape), 0.0f);

    int ret = CreateAclTensor(meansHostData, meansShape, &resources.meansDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.meansTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(quatsHostData, quatsShape, &resources.quatsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.quatsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(scalesHostData, scalesShape, &resources.scalesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.scalesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(conicsHostData, conicsShape, &resources.conicsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.conicsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(viewmatsHostData, viewmatsShape, &resources.viewmatsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.viewmatsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(ksHostData, ksShape, &resources.ksDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.ksTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_means2dHostData, v_means2dShape, &resources.v_means2dDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_means2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_depthsHostData, v_depthsShape, &resources.v_depthsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_depthsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_conicsHostData, v_conicsShape, &resources.v_conicsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_conicsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_colors_cullingHostData, v_colors_cullingShape, &resources.v_colors_cullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_colors_cullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_opacities_cullingHostData, v_opacities_cullingShape, &resources.v_opacities_cullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_opacities_cullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(filterHostData, filterShape, &resources.filterDeviceAddr,
                          aclDataType::ACL_UINT8, &resources.filterTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_pWHostData, v_pWShape, &resources.v_pWDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_pWTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_quatsHostData, v_quatsShape, &resources.v_quatsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_quatsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_scalesHostData, v_scalesShape, &resources.v_scalesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_scalesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_RHostData, v_RShape, &resources.v_RDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_RTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_colorsHostData, v_colorsShape, &resources.v_colorsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_colorsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_opacitiesHostData, v_opacitiesShape, &resources.v_opacitiesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_opacitiesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_viewmatsHostData, v_viewmatsShape, &resources.v_viewmatsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_viewmatsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(v_ksHostData, v_ksShape, &resources.v_ksDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.v_ksTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(dL_dviHostData, dL_dviShape, &resources.dL_dviDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.dL_dviTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(dL_dsigmaHostData, dL_dsigmaShape, &resources.dL_dsigmaDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.dL_dsigmaTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteFullyFusedProjectionBwd(TensorResources& resources, aclrtStream stream,
                                   void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t width = 800;
    int64_t height = 600;
    aclOpExecutor* executor;

    int ret = aclnnFullyFusedProjectionBwdGetWorkspaceSize(
        resources.meansTensor, resources.quatsTensor, resources.scalesTensor,
        resources.conicsTensor, resources.viewmatsTensor, resources.ksTensor,
        resources.v_means2dTensor, resources.v_depthsTensor, resources.v_conicsTensor,
        resources.v_colors_cullingTensor, resources.v_opacities_cullingTensor,
        resources.filterTensor,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        width, height, 0 /* maxCompactCount */, 0 /* supportFeatures */,
        resources.v_pWTensor, resources.v_quatsTensor, resources.v_scalesTensor,
        resources.v_RTensor, resources.v_colorsTensor, resources.v_opacitiesTensor,
        resources.v_viewmatsTensor, resources.v_ksTensor,
        resources.dL_dviTensor, resources.dL_dsigmaTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFullyFusedProjectionBwdGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnFullyFusedProjectionBwd(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFullyFusedProjectionBwd failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.meansTensor) aclDestroyTensor(resources.meansTensor);
    if (resources.quatsTensor) aclDestroyTensor(resources.quatsTensor);
    if (resources.scalesTensor) aclDestroyTensor(resources.scalesTensor);
    if (resources.conicsTensor) aclDestroyTensor(resources.conicsTensor);
    if (resources.viewmatsTensor) aclDestroyTensor(resources.viewmatsTensor);
    if (resources.ksTensor) aclDestroyTensor(resources.ksTensor);
    if (resources.v_means2dTensor) aclDestroyTensor(resources.v_means2dTensor);
    if (resources.v_depthsTensor) aclDestroyTensor(resources.v_depthsTensor);
    if (resources.v_conicsTensor) aclDestroyTensor(resources.v_conicsTensor);
    if (resources.v_colors_cullingTensor) aclDestroyTensor(resources.v_colors_cullingTensor);
    if (resources.v_opacities_cullingTensor) aclDestroyTensor(resources.v_opacities_cullingTensor);
    if (resources.filterTensor) aclDestroyTensor(resources.filterTensor);
    if (resources.v_pWTensor) aclDestroyTensor(resources.v_pWTensor);
    if (resources.v_quatsTensor) aclDestroyTensor(resources.v_quatsTensor);
    if (resources.v_scalesTensor) aclDestroyTensor(resources.v_scalesTensor);
    if (resources.v_RTensor) aclDestroyTensor(resources.v_RTensor);
    if (resources.v_colorsTensor) aclDestroyTensor(resources.v_colorsTensor);
    if (resources.v_opacitiesTensor) aclDestroyTensor(resources.v_opacitiesTensor);
    if (resources.v_viewmatsTensor) aclDestroyTensor(resources.v_viewmatsTensor);
    if (resources.v_ksTensor) aclDestroyTensor(resources.v_ksTensor);
    if (resources.dL_dviTensor) aclDestroyTensor(resources.dL_dviTensor);
    if (resources.dL_dsigmaTensor) aclDestroyTensor(resources.dL_dsigmaTensor);

    if (resources.meansDeviceAddr) aclrtFree(resources.meansDeviceAddr);
    if (resources.quatsDeviceAddr) aclrtFree(resources.quatsDeviceAddr);
    if (resources.scalesDeviceAddr) aclrtFree(resources.scalesDeviceAddr);
    if (resources.conicsDeviceAddr) aclrtFree(resources.conicsDeviceAddr);
    if (resources.viewmatsDeviceAddr) aclrtFree(resources.viewmatsDeviceAddr);
    if (resources.ksDeviceAddr) aclrtFree(resources.ksDeviceAddr);
    if (resources.v_means2dDeviceAddr) aclrtFree(resources.v_means2dDeviceAddr);
    if (resources.v_depthsDeviceAddr) aclrtFree(resources.v_depthsDeviceAddr);
    if (resources.v_conicsDeviceAddr) aclrtFree(resources.v_conicsDeviceAddr);
    if (resources.v_colors_cullingDeviceAddr) aclrtFree(resources.v_colors_cullingDeviceAddr);
    if (resources.v_opacities_cullingDeviceAddr) aclrtFree(resources.v_opacities_cullingDeviceAddr);
    if (resources.filterDeviceAddr) aclrtFree(resources.filterDeviceAddr);
    if (resources.v_pWDeviceAddr) aclrtFree(resources.v_pWDeviceAddr);
    if (resources.v_quatsDeviceAddr) aclrtFree(resources.v_quatsDeviceAddr);
    if (resources.v_scalesDeviceAddr) aclrtFree(resources.v_scalesDeviceAddr);
    if (resources.v_RDeviceAddr) aclrtFree(resources.v_RDeviceAddr);
    if (resources.v_colorsDeviceAddr) aclrtFree(resources.v_colorsDeviceAddr);
    if (resources.v_opacitiesDeviceAddr) aclrtFree(resources.v_opacitiesDeviceAddr);
    if (resources.v_viewmatsDeviceAddr) aclrtFree(resources.v_viewmatsDeviceAddr);
    if (resources.v_ksDeviceAddr) aclrtFree(resources.v_ksDeviceAddr);
    if (resources.dL_dviDeviceAddr) aclrtFree(resources.dL_dviDeviceAddr);
    if (resources.dL_dsigmaDeviceAddr) aclrtFree(resources.dL_dsigmaDeviceAddr);

    if (workspaceAddr) aclrtFree(workspaceAddr);
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    std::vector<int64_t> v_pWShape = {1, 3, 256};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteFullyFusedProjectionBwd(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    PrintResult<float>(v_pWShape, &resources.v_pWDeviceAddr, "v_pW");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}