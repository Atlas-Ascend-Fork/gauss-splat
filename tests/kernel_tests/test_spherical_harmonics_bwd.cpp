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
#include "aclnnop/aclnn_spherical_harmonics_bwd.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* dirsDeviceAddr = nullptr;
    void* coeffsDeviceAddr = nullptr;
    void* vColorsDeviceAddr = nullptr;
    void* vDirsDeviceAddr = nullptr;
    void* vCoeffsDeviceAddr = nullptr;

    aclTensor* dirsTensor = nullptr;
    aclTensor* coeffsTensor = nullptr;
    aclTensor* vColorsTensor = nullptr;
    aclTensor* vDirsTensor = nullptr;
    aclTensor* vCoeffsTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t gaussNum = 1024;
    int64_t degree = 3;
    int64_t K = (degree + 1) * (degree + 1);
    
    std::vector<int64_t> dirsShape =    {batchSize, 3, gaussNum};
    std::vector<int64_t> coeffsShape =  {batchSize, K, 3, gaussNum};
    std::vector<int64_t> vColorsShape = {batchSize, 3, gaussNum};
    std::vector<int64_t> vDirsShape =   {batchSize, 3, gaussNum};
    std::vector<int64_t> vCoeffsShape = {batchSize, K, 3, gaussNum};

    std::vector<float> dirsHostData(GetShapeSize(dirsShape), 0.5f);
    std::vector<float> coeffsHostData(GetShapeSize(coeffsShape), 0.1f);
    std::vector<float> vColorsHostData(GetShapeSize(vColorsShape), 1.0f);
    std::vector<float> vDirsHostData(GetShapeSize(vDirsShape), 0.0f);
    std::vector<float> vCoeffsHostData(GetShapeSize(vCoeffsShape), 0.0f);

    int ret = CreateAclTensor(dirsHostData, dirsShape, &resources.dirsDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.dirsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(coeffsHostData, coeffsShape, &resources.coeffsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.coeffsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(vColorsHostData, vColorsShape, &resources.vColorsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.vColorsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(vDirsHostData, vDirsShape, &resources.vDirsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.vDirsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(vCoeffsHostData, vCoeffsShape, &resources.vCoeffsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.vCoeffsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteSphericalHarmonicsBwd(TensorResources& resources, aclrtStream stream,
                                  void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t degree = 3;
    aclOpExecutor* executor;

    int ret = aclnnSphericalHarmonicsBwdGetWorkspaceSize(
        resources.dirsTensor, resources.coeffsTensor, resources.vColorsTensor, degree,
        resources.vDirsTensor, resources.vCoeffsTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSphericalHarmonicsBwdGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnSphericalHarmonicsBwd(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnSphericalHarmonicsBwd failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.dirsTensor) aclDestroyTensor(resources.dirsTensor);
    if (resources.coeffsTensor) aclDestroyTensor(resources.coeffsTensor);
    if (resources.vColorsTensor) aclDestroyTensor(resources.vColorsTensor);
    if (resources.vDirsTensor) aclDestroyTensor(resources.vDirsTensor);
    if (resources.vCoeffsTensor) aclDestroyTensor(resources.vCoeffsTensor);

    if (resources.dirsDeviceAddr) aclrtFree(resources.dirsDeviceAddr);
    if (resources.coeffsDeviceAddr) aclrtFree(resources.coeffsDeviceAddr);
    if (resources.vColorsDeviceAddr) aclrtFree(resources.vColorsDeviceAddr);
    if (resources.vDirsDeviceAddr) aclrtFree(resources.vDirsDeviceAddr);
    if (resources.vCoeffsDeviceAddr) aclrtFree(resources.vCoeffsDeviceAddr);

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

    ret = ExecuteSphericalHarmonicsBwd(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}