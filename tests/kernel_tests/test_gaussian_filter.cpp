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
#include "aclnnop/aclnn_gaussian_filter.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* meansDeviceAddr = nullptr;
    void* colorsDeviceAddr = nullptr;
    void* detDeviceAddr = nullptr;
    void* opacitiesDeviceAddr = nullptr;
    void* means2dDeviceAddr = nullptr;
    void* depthsDeviceAddr = nullptr;
    void* radiusDeviceAddr = nullptr;
    void* conicsDeviceAddr = nullptr;
    void* covars2dDeviceAddr = nullptr;
    void* compensationsDeviceAddr = nullptr;
    void* meansCullingDeviceAddr = nullptr;
    void* colorsCullingDeviceAddr = nullptr;
    void* means2dCullingDeviceAddr = nullptr;
    void* depthsCullingDeviceAddr = nullptr;
    void* radiusCullingDeviceAddr = nullptr;
    void* covars2dCullingDeviceAddr = nullptr;
    void* conicsCullingDeviceAddr = nullptr;
    void* opacitiesCullingDeviceAddr = nullptr;
    void* filterDeviceAddr = nullptr;
    void* cntDeviceAddr = nullptr;

    aclTensor* meansTensor = nullptr;
    aclTensor* colorsTensor = nullptr;
    aclTensor* detTensor = nullptr;
    aclTensor* opacitiesTensor = nullptr;
    aclTensor* means2dTensor = nullptr;
    aclTensor* depthsTensor = nullptr;
    aclTensor* radiusTensor = nullptr;
    aclTensor* conicsTensor = nullptr;
    aclTensor* covars2dTensor = nullptr;
    aclTensor* compensationsTensor = nullptr;
    aclTensor* meansCullingTensor = nullptr;
    aclTensor* colorsCullingTensor = nullptr;
    aclTensor* means2dCullingTensor = nullptr;
    aclTensor* depthsCullingTensor = nullptr;
    aclTensor* radiusCullingTensor = nullptr;
    aclTensor* covars2dCullingTensor = nullptr;
    aclTensor* conicsCullingTensor = nullptr;
    aclTensor* opacitiesCullingTensor = nullptr;
    aclTensor* filterTensor = nullptr;
    aclTensor* cntTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t cameraNum = 1;
    int64_t gaussNum = 1024;
    
    std::vector<int64_t> meansShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> colorsShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> detShape = {batchSize, cameraNum, gaussNum};
    std::vector<int64_t> opacitiesShape = {batchSize, gaussNum};
    std::vector<int64_t> means2dShape = {batchSize, cameraNum, 2, gaussNum};
    std::vector<int64_t> depthsShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> radiusShape = {batchSize, cameraNum, 2, gaussNum};
    std::vector<int64_t> conicsShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> covars2dShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> compensationsShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> filterShape = {batchSize, cameraNum, gaussNum};
    std::vector<int64_t> cntShape = {batchSize, cameraNum};

    std::vector<float> meansHostData(GetShapeSize(meansShape), 0.0f);
    std::vector<float> colorsHostData(GetShapeSize(colorsShape), 1.0f);
    std::vector<float> detHostData(GetShapeSize(detShape), 1.0f);
    std::vector<float> opacitiesHostData(GetShapeSize(opacitiesShape), 0.5f);
    std::vector<float> means2dHostData(GetShapeSize(means2dShape), 400.0f);
    std::vector<float> depthsHostData(GetShapeSize(depthsShape), 5.0f);
    std::vector<float> radiusHostData(GetShapeSize(radiusShape), 10.0f);
    std::vector<float> conicsHostData(GetShapeSize(conicsShape), 0.1f);
    std::vector<float> covars2dHostData(GetShapeSize(covars2dShape), 0.1f);
    std::vector<uint8_t> filterHostData(GetShapeSize(filterShape), 0);
    std::vector<int32_t> cntHostData(GetShapeSize(cntShape), 0);

    int ret = CreateAclTensor(meansHostData, meansShape, &resources.meansDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.meansTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(colorsHostData, colorsShape, &resources.colorsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.colorsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(detHostData, detShape, &resources.detDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.detTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(opacitiesHostData, opacitiesShape, &resources.opacitiesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.opacitiesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(means2dHostData, means2dShape, &resources.means2dDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.means2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(depthsHostData, depthsShape, &resources.depthsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.depthsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(radiusHostData, radiusShape, &resources.radiusDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.radiusTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(conicsHostData, conicsShape, &resources.conicsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.conicsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(covars2dHostData, covars2dShape, &resources.covars2dDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.covars2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(meansHostData, meansShape, &resources.meansCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.meansCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(colorsHostData, colorsShape, &resources.colorsCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.colorsCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(means2dHostData, means2dShape, &resources.means2dCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.means2dCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(depthsHostData, depthsShape, &resources.depthsCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.depthsCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(radiusHostData, radiusShape, &resources.radiusCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.radiusCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(covars2dHostData, covars2dShape, &resources.covars2dCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.covars2dCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(conicsHostData, conicsShape, &resources.conicsCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.conicsCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(opacitiesHostData, opacitiesShape, &resources.opacitiesCullingDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.opacitiesCullingTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(filterHostData, filterShape, &resources.filterDeviceAddr,
                          aclDataType::ACL_UINT8, &resources.filterTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(cntHostData, cntShape, &resources.cntDeviceAddr,
                          aclDataType::ACL_INT32, &resources.cntTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteGaussianFilter(TensorResources& resources, aclrtStream stream,
                          void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t width = 800;
    int64_t height = 600;
    float nearPlane = 0.01f;
    float farPlane = 100.0f;
    aclOpExecutor* executor;

    int ret = aclnnGaussianFilterGetWorkspaceSize(
        resources.meansTensor, resources.colorsTensor, resources.detTensor, resources.opacitiesTensor,
        resources.means2dTensor, resources.depthsTensor, resources.radiusTensor, resources.conicsTensor,
        resources.covars2dTensor, nullptr, width, height, nearPlane, farPlane,
        resources.meansCullingTensor, resources.colorsCullingTensor, resources.means2dCullingTensor,
        resources.depthsCullingTensor, resources.radiusCullingTensor, resources.covars2dCullingTensor,
        resources.conicsCullingTensor, resources.opacitiesCullingTensor, resources.filterTensor,
        resources.cntTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGaussianFilterGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnGaussianFilter(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGaussianFilter failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.meansTensor) aclDestroyTensor(resources.meansTensor);
    if (resources.colorsTensor) aclDestroyTensor(resources.colorsTensor);
    if (resources.detTensor) aclDestroyTensor(resources.detTensor);
    if (resources.opacitiesTensor) aclDestroyTensor(resources.opacitiesTensor);
    if (resources.means2dTensor) aclDestroyTensor(resources.means2dTensor);
    if (resources.depthsTensor) aclDestroyTensor(resources.depthsTensor);
    if (resources.radiusTensor) aclDestroyTensor(resources.radiusTensor);
    if (resources.conicsTensor) aclDestroyTensor(resources.conicsTensor);
    if (resources.covars2dTensor) aclDestroyTensor(resources.covars2dTensor);
    if (resources.meansCullingTensor) aclDestroyTensor(resources.meansCullingTensor);
    if (resources.colorsCullingTensor) aclDestroyTensor(resources.colorsCullingTensor);
    if (resources.means2dCullingTensor) aclDestroyTensor(resources.means2dCullingTensor);
    if (resources.depthsCullingTensor) aclDestroyTensor(resources.depthsCullingTensor);
    if (resources.radiusCullingTensor) aclDestroyTensor(resources.radiusCullingTensor);
    if (resources.covars2dCullingTensor) aclDestroyTensor(resources.covars2dCullingTensor);
    if (resources.conicsCullingTensor) aclDestroyTensor(resources.conicsCullingTensor);
    if (resources.opacitiesCullingTensor) aclDestroyTensor(resources.opacitiesCullingTensor);
    if (resources.filterTensor) aclDestroyTensor(resources.filterTensor);
    if (resources.cntTensor) aclDestroyTensor(resources.cntTensor);

    if (resources.meansDeviceAddr) aclrtFree(resources.meansDeviceAddr);
    if (resources.colorsDeviceAddr) aclrtFree(resources.colorsDeviceAddr);
    if (resources.detDeviceAddr) aclrtFree(resources.detDeviceAddr);
    if (resources.opacitiesDeviceAddr) aclrtFree(resources.opacitiesDeviceAddr);
    if (resources.means2dDeviceAddr) aclrtFree(resources.means2dDeviceAddr);
    if (resources.depthsDeviceAddr) aclrtFree(resources.depthsDeviceAddr);
    if (resources.radiusDeviceAddr) aclrtFree(resources.radiusDeviceAddr);
    if (resources.conicsDeviceAddr) aclrtFree(resources.conicsDeviceAddr);
    if (resources.covars2dDeviceAddr) aclrtFree(resources.covars2dDeviceAddr);
    if (resources.meansCullingDeviceAddr) aclrtFree(resources.meansCullingDeviceAddr);
    if (resources.colorsCullingDeviceAddr) aclrtFree(resources.colorsCullingDeviceAddr);
    if (resources.means2dCullingDeviceAddr) aclrtFree(resources.means2dCullingDeviceAddr);
    if (resources.depthsCullingDeviceAddr) aclrtFree(resources.depthsCullingDeviceAddr);
    if (resources.radiusCullingDeviceAddr) aclrtFree(resources.radiusCullingDeviceAddr);
    if (resources.covars2dCullingDeviceAddr) aclrtFree(resources.covars2dCullingDeviceAddr);
    if (resources.conicsCullingDeviceAddr) aclrtFree(resources.conicsCullingDeviceAddr);
    if (resources.opacitiesCullingDeviceAddr) aclrtFree(resources.opacitiesCullingDeviceAddr);
    if (resources.filterDeviceAddr) aclrtFree(resources.filterDeviceAddr);
    if (resources.cntDeviceAddr) aclrtFree(resources.cntDeviceAddr);

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
    std::vector<int64_t> cntShape = {1, 1};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteGaussianFilter(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    PrintIntResult(cntShape, &resources.cntDeviceAddr, "cnt");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}