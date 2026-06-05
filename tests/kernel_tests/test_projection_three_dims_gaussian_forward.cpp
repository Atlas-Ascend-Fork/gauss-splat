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
#include "aclnnop/aclnn_projection_three_dims_gaussian_forward.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* meansDeviceAddr = nullptr;
    void* covarsDeviceAddr = nullptr;
    void* viewmatsDeviceAddr = nullptr;
    void* ksDeviceAddr = nullptr;
    void* means2dDeviceAddr = nullptr;
    void* depthsDeviceAddr = nullptr;
    void* conicsDeviceAddr = nullptr;
    void* compensationsDeviceAddr = nullptr;
    void* detDeviceAddr = nullptr;
    void* radiusDeviceAddr = nullptr;
    void* covars2dDeviceAddr = nullptr;

    aclTensor* meansTensor = nullptr;
    aclTensor* covarsTensor = nullptr;
    aclTensor* viewmatsTensor = nullptr;
    aclTensor* ksTensor = nullptr;
    aclTensor* means2dTensor = nullptr;
    aclTensor* depthsTensor = nullptr;
    aclTensor* conicsTensor = nullptr;
    aclTensor* compensationsTensor = nullptr;
    aclTensor* detTensor = nullptr;
    aclTensor* radiusTensor = nullptr;
    aclTensor* covars2dTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t cameraNum = 1;
    int64_t gaussNum = 1024;
    
    std::vector<int64_t> meansShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> covarsShape = {batchSize, 3, 3, gaussNum};
    std::vector<int64_t> viewmatsShape = {batchSize, cameraNum, 4, 4};
    std::vector<int64_t> ksShape = {batchSize, cameraNum, 3, 3};
    std::vector<int64_t> means2dShape = {batchSize, cameraNum, 2, gaussNum};
    std::vector<int64_t> depthsShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> conicsShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> compensationsShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> detShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> radiusShape = {batchSize, cameraNum, 2, gaussNum};
    std::vector<int64_t> covars2dShape = {batchSize, cameraNum, 3, gaussNum};

    std::vector<float> meansHostData(GetShapeSize(meansShape), 0.0f);
    std::vector<float> covarsHostData(GetShapeSize(covarsShape), 1.0f);
    std::vector<float> viewmatsHostData(GetShapeSize(viewmatsShape), 1.0f);
    std::vector<float> ksHostData(GetShapeSize(ksShape), 500.0f);
    std::vector<float> means2dHostData(GetShapeSize(means2dShape), 0.0f);
    std::vector<float> depthsHostData(GetShapeSize(depthsShape), 0.0f);
    std::vector<float> conicsHostData(GetShapeSize(conicsShape), 0.0f);
    std::vector<float> compensationsHostData(GetShapeSize(compensationsShape), 0.0f);
    std::vector<float> detHostData(GetShapeSize(detShape), 0.0f);
    std::vector<float> radiusHostData(GetShapeSize(radiusShape), 0.0f);
    std::vector<float> covars2dHostData(GetShapeSize(covars2dShape), 0.0f);

    int ret = CreateAclTensor(meansHostData, meansShape, &resources.meansDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.meansTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(covarsHostData, covarsShape, &resources.covarsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.covarsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(viewmatsHostData, viewmatsShape, &resources.viewmatsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.viewmatsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(ksHostData, ksShape, &resources.ksDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.ksTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(means2dHostData, means2dShape, &resources.means2dDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.means2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(depthsHostData, depthsShape, &resources.depthsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.depthsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(conicsHostData, conicsShape, &resources.conicsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.conicsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(compensationsHostData, compensationsShape, &resources.compensationsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.compensationsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(detHostData, detShape, &resources.detDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.detTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(radiusHostData, radiusShape, &resources.radiusDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.radiusTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(covars2dHostData, covars2dShape, &resources.covars2dDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.covars2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteProjectionThreeDimsGaussianForward(TensorResources& resources, aclrtStream stream,
                                               void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t imageWidth = 800;
    int64_t imageHeight = 600;
    float eps2d = 0.3f;
    bool calcCompensations = false;
    int64_t cameraModel = 0;
    aclOpExecutor* executor;

    int ret = aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize(
        resources.meansTensor, resources.covarsTensor, resources.viewmatsTensor, resources.ksTensor,
        imageWidth, imageHeight, eps2d, calcCompensations, cameraModel,
        resources.means2dTensor, resources.depthsTensor, resources.conicsTensor,
        resources.compensationsTensor, resources.detTensor, resources.radiusTensor, resources.covars2dTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnProjectionThreeDimsGaussianForward(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnProjectionThreeDimsGaussianForward failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.meansTensor) aclDestroyTensor(resources.meansTensor);
    if (resources.covarsTensor) aclDestroyTensor(resources.covarsTensor);
    if (resources.viewmatsTensor) aclDestroyTensor(resources.viewmatsTensor);
    if (resources.ksTensor) aclDestroyTensor(resources.ksTensor);
    if (resources.means2dTensor) aclDestroyTensor(resources.means2dTensor);
    if (resources.depthsTensor) aclDestroyTensor(resources.depthsTensor);
    if (resources.conicsTensor) aclDestroyTensor(resources.conicsTensor);
    if (resources.compensationsTensor) aclDestroyTensor(resources.compensationsTensor);
    if (resources.detTensor) aclDestroyTensor(resources.detTensor);
    if (resources.radiusTensor) aclDestroyTensor(resources.radiusTensor);
    if (resources.covars2dTensor) aclDestroyTensor(resources.covars2dTensor);

    if (resources.meansDeviceAddr) aclrtFree(resources.meansDeviceAddr);
    if (resources.covarsDeviceAddr) aclrtFree(resources.covarsDeviceAddr);
    if (resources.viewmatsDeviceAddr) aclrtFree(resources.viewmatsDeviceAddr);
    if (resources.ksDeviceAddr) aclrtFree(resources.ksDeviceAddr);
    if (resources.means2dDeviceAddr) aclrtFree(resources.means2dDeviceAddr);
    if (resources.depthsDeviceAddr) aclrtFree(resources.depthsDeviceAddr);
    if (resources.conicsDeviceAddr) aclrtFree(resources.conicsDeviceAddr);
    if (resources.compensationsDeviceAddr) aclrtFree(resources.compensationsDeviceAddr);
    if (resources.detDeviceAddr) aclrtFree(resources.detDeviceAddr);
    if (resources.radiusDeviceAddr) aclrtFree(resources.radiusDeviceAddr);
    if (resources.covars2dDeviceAddr) aclrtFree(resources.covars2dDeviceAddr);

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
    std::vector<int64_t> means2dShape = {1, 1, 2, 1024};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteProjectionThreeDimsGaussianForward(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    PrintResult<float>(means2dShape, &resources.means2dDeviceAddr, "means2d");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}