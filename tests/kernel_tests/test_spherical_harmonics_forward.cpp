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
#include "aclnnop/aclnn_spherical_harmonics_forward.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* dirsDeviceAddr = nullptr;
    void* coeffsDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;

    aclTensor* dirsTensor = nullptr;
    aclTensor* coeffsTensor = nullptr;
    aclTensor* outputTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t gaussNum = 1024;
    int64_t degreesToUse = 3;
    int64_t K = (degreesToUse + 1) * (degreesToUse + 1);
    
    std::vector<int64_t> dirsShape =   {3, gaussNum};
    std::vector<int64_t> coeffsShape = {K, 3, gaussNum};
    std::vector<int64_t> outputShape = {3, gaussNum};

    int64_t dirsShapeSize = GetShapeSize(dirsShape);
    int64_t coeffsShapeSize = GetShapeSize(coeffsShape);
    int64_t outputShapeSize = GetShapeSize(outputShape);

    std::vector<float> dirsHostData(dirsShapeSize, 0.5f);
    std::vector<float> coeffsHostData(coeffsShapeSize, 0.1f);
    std::vector<float> outputHostData(outputShapeSize, 0.0f);

    int ret = CreateAclTensor(dirsHostData, dirsShape, &resources.dirsDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.dirsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(coeffsHostData, coeffsShape, &resources.coeffsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.coeffsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(outputHostData, outputShape, &resources.outputDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.outputTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }
    return ACL_SUCCESS;
}

int ExecuteSphericalHarmonicsForward(TensorResources& resources, aclrtStream stream,
                                      void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t degreesToUse = 3;
    aclOpExecutor* executor;

    int ret = aclnnSphericalHarmonicsForwardGetWorkspaceSize(
        resources.dirsTensor, resources.coeffsTensor, degreesToUse,
        resources.outputTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSphericalHarmonicsForwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnSphericalHarmonicsForward(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSphericalHarmonicsForward failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.dirsTensor) {
      aclDestroyTensor(resources.dirsTensor);
    }
    if (resources.coeffsTensor) {
      aclDestroyTensor(resources.coeffsTensor);
    }
    if (resources.outputTensor) {
      aclDestroyTensor(resources.outputTensor);
    }

    if (resources.dirsDeviceAddr) {
      aclrtFree(resources.dirsDeviceAddr);
    }
    if (resources.coeffsDeviceAddr) {
      aclrtFree(resources.coeffsDeviceAddr);
    }
    if (resources.outputDeviceAddr) {
      aclrtFree(resources.outputDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
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
    std::vector<int64_t> outputShape = {1, 1024, 3};
    int ret = ACL_SUCCESS;

    // 1. Initialize device and stream
    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    // 2. Initialize tensors
    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 3. Execute the operation
    ret = ExecuteSphericalHarmonicsForward(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 4. Synchronize stream
    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    // 5. Process results
    PrintResult<float>(outputShape, &resources.outputDeviceAddr, "output");

    // 6. Cleanup resources
    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}