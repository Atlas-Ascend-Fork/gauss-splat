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
#include "aclnnop/aclnn_calc_render_simt_fwd.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* gsDeviceAddr = nullptr;
    void* tileCoordsDeviceAddr = nullptr;
    void* offsetsDeviceAddr = nullptr;
    void* gsIdsDeviceAddr = nullptr;
    void* colorDeviceAddr = nullptr;
    void* depthDeviceAddr = nullptr;
    void* lastCumsumDeviceAddr = nullptr;
    void* errorDeviceAddr = nullptr;
    void* lastIdsDeviceAddr = nullptr;

    aclTensor* gsTensor = nullptr;
    aclTensor* tileCoordsTensor = nullptr;
    aclTensor* offsetsTensor = nullptr;
    aclTensor* gsIdsTensor = nullptr;
    aclTensor* colorTensor = nullptr;
    aclTensor* depthTensor = nullptr;
    aclTensor* lastCumsumTensor = nullptr;
    aclTensor* errorTensor = nullptr;
    aclTensor* lastIdsTensor = nullptr;
};

int InitializeTensors(TensorResources& resources, int32_t deviceId) {
    int64_t nGauss = 256;
    int64_t tileNum = 16;
    int64_t nPixel = 1024;
    int64_t totalGauss = tileNum * nGauss;

    // The tiling contract requires len(offsets) == vecNum + 2 * tileNum, where
    // vecNum is the AIV core count reported by the platform.
    int64_t vectorCoreNum = 0;
    auto capRet = aclGetDeviceCapability(deviceId, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &vectorCoreNum);
    if (!CHECK_RET(capRet == ACL_SUCCESS) || vectorCoreNum <= 0) {
        LOG_PRINT("aclGetDeviceCapability failed. ERROR: %d\n", capRet);
        return capRet == ACL_SUCCESS ? ACL_ERROR_FAILURE : capRet;
    }

    std::vector<int64_t> gsShape = {nGauss, 10};
    std::vector<int64_t> tileCoordsShape = {tileNum, 2, nPixel};
    std::vector<int64_t> offsetsShape = {vectorCoreNum + 2 * tileNum};
    std::vector<int64_t> gsIdsShape = {totalGauss};
    std::vector<int64_t> colorShape = {3, tileNum, nPixel};
    std::vector<int64_t> depthShape = {tileNum, nPixel};
    std::vector<int64_t> lastCumsumShape = {tileNum, nPixel};
    std::vector<int64_t> errorShape = {tileNum, nPixel};
    std::vector<int64_t> lastIdsShape = {tileNum, nPixel};

    std::vector<float> gsHostData(GetShapeSize(gsShape), 0.5f);
    std::vector<float> tileCoordsHostData(GetShapeSize(tileCoordsShape), 0.0f);
    std::vector<int64_t> offsetsHostData(GetShapeSize(offsetsShape), 0);
    std::vector<int64_t> gsIdsHostData(GetShapeSize(gsIdsShape), 0);
    std::vector<float> colorHostData(GetShapeSize(colorShape), 0.0f);
    std::vector<float> depthHostData(GetShapeSize(depthShape), 0.0f);
    std::vector<float> lastCumsumHostData(GetShapeSize(lastCumsumShape), 0.0f);
    std::vector<float> errorHostData(GetShapeSize(errorShape), 0.0f);
    std::vector<int32_t> lastIdsHostData(GetShapeSize(lastIdsShape), -1);

    int ret = CreateAclTensor(gsHostData, gsShape, &resources.gsDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.gsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(tileCoordsHostData, tileCoordsShape, &resources.tileCoordsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.tileCoordsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(offsetsHostData, offsetsShape, &resources.offsetsDeviceAddr,
                          aclDataType::ACL_INT64, &resources.offsetsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gsIdsHostData, gsIdsShape, &resources.gsIdsDeviceAddr,
                          aclDataType::ACL_INT64, &resources.gsIdsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(colorHostData, colorShape, &resources.colorDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.colorTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(depthHostData, depthShape, &resources.depthDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.depthTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(lastCumsumHostData, lastCumsumShape, &resources.lastCumsumDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.lastCumsumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(errorHostData, errorShape, &resources.errorDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.errorTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(lastIdsHostData, lastIdsShape, &resources.lastIdsDeviceAddr,
                          aclDataType::ACL_INT32, &resources.lastIdsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteCalcRenderSimtFwd(TensorResources& resources, aclrtStream stream,
                             void** workspaceAddr, uint64_t* workspaceSize) {
    aclOpExecutor* executor;

    int ret = aclnnCalcRenderSimtFwdGetWorkspaceSize(
        resources.gsTensor, resources.tileCoordsTensor, resources.offsetsTensor, resources.gsIdsTensor,
        resources.colorTensor, resources.depthTensor, resources.lastCumsumTensor, resources.errorTensor,
        resources.lastIdsTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnCalcRenderSimtFwdGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnCalcRenderSimtFwd(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnCalcRenderSimtFwd failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                      aclrtStream stream, int32_t deviceId) {
    if (resources.gsTensor) aclDestroyTensor(resources.gsTensor);
    if (resources.tileCoordsTensor) aclDestroyTensor(resources.tileCoordsTensor);
    if (resources.offsetsTensor) aclDestroyTensor(resources.offsetsTensor);
    if (resources.gsIdsTensor) aclDestroyTensor(resources.gsIdsTensor);
    if (resources.colorTensor) aclDestroyTensor(resources.colorTensor);
    if (resources.depthTensor) aclDestroyTensor(resources.depthTensor);
    if (resources.lastCumsumTensor) aclDestroyTensor(resources.lastCumsumTensor);
    if (resources.errorTensor) aclDestroyTensor(resources.errorTensor);
    if (resources.lastIdsTensor) aclDestroyTensor(resources.lastIdsTensor);

    if (resources.gsDeviceAddr) aclrtFree(resources.gsDeviceAddr);
    if (resources.tileCoordsDeviceAddr) aclrtFree(resources.tileCoordsDeviceAddr);
    if (resources.offsetsDeviceAddr) aclrtFree(resources.offsetsDeviceAddr);
    if (resources.gsIdsDeviceAddr) aclrtFree(resources.gsIdsDeviceAddr);
    if (resources.colorDeviceAddr) aclrtFree(resources.colorDeviceAddr);
    if (resources.depthDeviceAddr) aclrtFree(resources.depthDeviceAddr);
    if (resources.lastCumsumDeviceAddr) aclrtFree(resources.lastCumsumDeviceAddr);
    if (resources.errorDeviceAddr) aclrtFree(resources.errorDeviceAddr);
    if (resources.lastIdsDeviceAddr) aclrtFree(resources.lastIdsDeviceAddr);

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
    std::vector<int64_t> colorShape = {3, 16, 1024};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources, deviceId);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteCalcRenderSimtFwd(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    PrintResult<float>(colorShape, &resources.colorDeviceAddr, "color");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
