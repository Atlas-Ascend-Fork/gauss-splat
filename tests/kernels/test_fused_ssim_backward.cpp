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
#include "aclnnop/aclnn_fused_ssim_backward.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* gradSsimMapDeviceAddr = nullptr;
    void* img1DeviceAddr = nullptr;
    void* img2DeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* mu1DeviceAddr = nullptr;
    void* mu2DeviceAddr = nullptr;
    void* s11DeviceAddr = nullptr;
    void* s22DeviceAddr = nullptr;
    void* s12DeviceAddr = nullptr;
    void* gradImg1DeviceAddr = nullptr;
    void* gradImg2DeviceAddr = nullptr;

    aclTensor* gradSsimMapTensor = nullptr;
    aclTensor* img1Tensor = nullptr;
    aclTensor* img2Tensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* mu1Tensor = nullptr;
    aclTensor* mu2Tensor = nullptr;
    aclTensor* s11Tensor = nullptr;
    aclTensor* s22Tensor = nullptr;
    aclTensor* s12Tensor = nullptr;
    aclTensor* gradImg1Tensor = nullptr;
    aclTensor* gradImg2Tensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t channelNum = 3;
    int64_t height = 64;
    int64_t width = 70;
    int64_t windowSize = 7;
    int64_t weightLen = 16;

    std::vector<int64_t> imgShape = {batchSize, channelNum, height, width};
    std::vector<int64_t> weightShape = {weightLen};

    std::vector<float> img1HostData(GetShapeSize(imgShape), 0.25f);
    std::vector<float> img2HostData(GetShapeSize(imgShape), 0.30f);
    std::vector<float> gradSsimMapHostData(GetShapeSize(imgShape), 1.0f);
    std::vector<float> muHostData(GetShapeSize(imgShape), 0.25f);
    std::vector<float> sHostData(GetShapeSize(imgShape), 0.10f);
    std::vector<float> outHostData(GetShapeSize(imgShape), 0.0f);
    std::vector<float> weightHostData(weightLen, 0.0f);
    float weightSum = 0.0f;
    for (int64_t i = 0; i < windowSize; i++) {
        float offset = static_cast<float>(i - windowSize / 2);
        weightHostData[i] = std::exp(-(offset * offset) / (2.0f * 0.75f * 0.75f));
        weightSum += weightHostData[i];
    }
    for (int64_t i = 0; i < windowSize; i++) {
        weightHostData[i] /= weightSum;
    }

    int ret = CreateAclTensor(gradSsimMapHostData, imgShape, &resources.gradSsimMapDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.gradSsimMapTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(img1HostData, imgShape, &resources.img1DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.img1Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(img2HostData, imgShape, &resources.img2DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.img2Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(weightHostData, weightShape, &resources.weightDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.weightTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(muHostData, imgShape, &resources.mu1DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.mu1Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(muHostData, imgShape, &resources.mu2DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.mu2Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(sHostData, imgShape, &resources.s11DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.s11Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(sHostData, imgShape, &resources.s22DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.s22Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(sHostData, imgShape, &resources.s12DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.s12Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(outHostData, imgShape, &resources.gradImg1DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.gradImg1Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(outHostData, imgShape, &resources.gradImg2DeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.gradImg2Tensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteFusedSsimBackward(TensorResources& resources, aclrtStream stream,
                             void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t windowSize = 7;
    int64_t needGradImg2 = 1;
    aclOpExecutor* executor;

    int ret = aclnnFusedSsimBackwardGetWorkspaceSize(
        resources.gradSsimMapTensor, resources.img1Tensor, resources.img2Tensor, resources.weightTensor,
        resources.mu1Tensor, resources.mu2Tensor, resources.s11Tensor, resources.s22Tensor, resources.s12Tensor,
        windowSize, needGradImg2, resources.gradImg1Tensor, resources.gradImg2Tensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedSsimBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnFusedSsimBackward(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFusedSsimBackward failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                      aclrtStream stream, int32_t deviceId) {
    if (resources.gradSsimMapTensor) aclDestroyTensor(resources.gradSsimMapTensor);
    if (resources.img1Tensor) aclDestroyTensor(resources.img1Tensor);
    if (resources.img2Tensor) aclDestroyTensor(resources.img2Tensor);
    if (resources.weightTensor) aclDestroyTensor(resources.weightTensor);
    if (resources.mu1Tensor) aclDestroyTensor(resources.mu1Tensor);
    if (resources.mu2Tensor) aclDestroyTensor(resources.mu2Tensor);
    if (resources.s11Tensor) aclDestroyTensor(resources.s11Tensor);
    if (resources.s22Tensor) aclDestroyTensor(resources.s22Tensor);
    if (resources.s12Tensor) aclDestroyTensor(resources.s12Tensor);
    if (resources.gradImg1Tensor) aclDestroyTensor(resources.gradImg1Tensor);
    if (resources.gradImg2Tensor) aclDestroyTensor(resources.gradImg2Tensor);

    if (resources.gradSsimMapDeviceAddr) aclrtFree(resources.gradSsimMapDeviceAddr);
    if (resources.img1DeviceAddr) aclrtFree(resources.img1DeviceAddr);
    if (resources.img2DeviceAddr) aclrtFree(resources.img2DeviceAddr);
    if (resources.weightDeviceAddr) aclrtFree(resources.weightDeviceAddr);
    if (resources.mu1DeviceAddr) aclrtFree(resources.mu1DeviceAddr);
    if (resources.mu2DeviceAddr) aclrtFree(resources.mu2DeviceAddr);
    if (resources.s11DeviceAddr) aclrtFree(resources.s11DeviceAddr);
    if (resources.s22DeviceAddr) aclrtFree(resources.s22DeviceAddr);
    if (resources.s12DeviceAddr) aclrtFree(resources.s12DeviceAddr);
    if (resources.gradImg1DeviceAddr) aclrtFree(resources.gradImg1DeviceAddr);
    if (resources.gradImg2DeviceAddr) aclrtFree(resources.gradImg2DeviceAddr);

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
    std::vector<int64_t> gradImg1Shape = {1, 3, 64, 70};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteFusedSsimBackward(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintResult<float>(gradImg1Shape, &resources.gradImg1DeviceAddr, "gradImg1");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
