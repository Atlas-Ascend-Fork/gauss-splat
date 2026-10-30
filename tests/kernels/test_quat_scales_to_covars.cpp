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
#include "aclnnop/aclnn_quat_scales_to_covars.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* quatDeviceAddr = nullptr;
    void* scalesDeviceAddr = nullptr;
    void* covarsDeviceAddr = nullptr;

    aclTensor* quatTensor = nullptr;
    aclTensor* scalesTensor = nullptr;
    aclTensor* covarsTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t gaussNum = 1024;
    
    std::vector<int64_t> quatShape = {batchSize, 4, gaussNum};
    std::vector<int64_t> scalesShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> covarsShape = {batchSize, 3, 3, gaussNum};

    std::vector<float> quatHostData(GetShapeSize(quatShape), 0.707f);
    std::vector<float> scalesHostData(GetShapeSize(scalesShape), 1.0f);
    std::vector<float> covarsHostData(GetShapeSize(covarsShape), 0.0f);

    int ret = CreateAclTensor(quatHostData, quatShape, &resources.quatDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.quatTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(scalesHostData, scalesShape, &resources.scalesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.scalesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(covarsHostData, covarsShape, &resources.covarsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.covarsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteQuatScalesToCovars(TensorResources& resources, aclrtStream stream,
                               void** workspaceAddr, uint64_t* workspaceSize) {
    aclOpExecutor* executor;

    int ret = aclnnQuatScalesToCovarsGetWorkspaceSize(
        resources.quatTensor, resources.scalesTensor, resources.covarsTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuatScalesToCovarsGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnQuatScalesToCovars(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnQuatScalesToCovars failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.quatTensor) aclDestroyTensor(resources.quatTensor);
    if (resources.scalesTensor) aclDestroyTensor(resources.scalesTensor);
    if (resources.covarsTensor) aclDestroyTensor(resources.covarsTensor);

    if (resources.quatDeviceAddr) aclrtFree(resources.quatDeviceAddr);
    if (resources.scalesDeviceAddr) aclrtFree(resources.scalesDeviceAddr);
    if (resources.covarsDeviceAddr) aclrtFree(resources.covarsDeviceAddr);

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
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteQuatScalesToCovars(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    std::vector<int64_t> covarsShape = {1, 3, 3, 1024};
    PrintResult<float>(covarsShape, &resources.covarsDeviceAddr, "covars");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}