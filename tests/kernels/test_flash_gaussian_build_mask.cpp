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
#include "aclnnop/aclnn_flash_gaussian_build_mask.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* means2dDeviceAddr = nullptr;
    void* opacityDeviceAddr = nullptr;
    void* conicsDeviceAddr = nullptr;
    void* covars2dDeviceAddr = nullptr;
    void* depthsDeviceAddr = nullptr;
    void* cntDeviceAddr = nullptr;
    void* tileGridDeviceAddr = nullptr;
    void* gatherMaskDeviceAddr = nullptr;
    void* tileSumDeviceAddr = nullptr;
    void* tileDepthsDeviceAddr = nullptr;
    void* gaussIndexDeviceAddr = nullptr;

    aclTensor* means2dTensor = nullptr;
    aclTensor* opacityTensor = nullptr;
    aclTensor* conicsTensor = nullptr;
    aclTensor* covars2dTensor = nullptr;
    aclTensor* depthsTensor = nullptr;
    aclTensor* cntTensor = nullptr;
    aclTensor* tileGridTensor = nullptr;
    aclTensor* gatherMaskTensor = nullptr;
    aclTensor* tileSumTensor = nullptr;
    aclTensor* tileDepthsTensor = nullptr;
    aclTensor* gaussIndexTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t cameraNum = 1;
    int64_t gaussNum = 256;
    int64_t numTile = 50;
    
    std::vector<int64_t> means2dShape = {batchSize, cameraNum, 2, gaussNum};
    std::vector<int64_t> opacityShape = {batchSize, cameraNum, 1, gaussNum};
    std::vector<int64_t> conicsShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> covars2dShape = {batchSize, cameraNum, 3, gaussNum};
    std::vector<int64_t> depthsShape = {batchSize, cameraNum, gaussNum};
    std::vector<int64_t> cntShape = {batchSize, cameraNum, 1};
    std::vector<int64_t> tileGridShape = {numTile, 2};
    std::vector<int64_t> gatherMaskShape = {48, 20, gaussNum};
    std::vector<int64_t> tileSumShape = {1, 1, numTile, 1};
    std::vector<int64_t> tileDepthsShape = {batchSize, cameraNum, numTile, gaussNum};
    std::vector<int64_t> gaussIndexShape = {batchSize, cameraNum, numTile, gaussNum};

    std::vector<float> means2dHostData(GetShapeSize(means2dShape), 100.0f);
    std::vector<float> opacityHostData(GetShapeSize(opacityShape), 0.5f);
    std::vector<float> conicsHostData(GetShapeSize(conicsShape), 0.0f);
    std::vector<float> covars2dHostData(GetShapeSize(covars2dShape), 0.0f);
    std::vector<float> depthsHostData(GetShapeSize(depthsShape), 5.0f);
    std::vector<int32_t> cntHostData(GetShapeSize(cntShape), 100);
    std::vector<float> tileGridHostData(GetShapeSize(tileGridShape), 0.0f);
    std::vector<float> gatherMaskHostData(GetShapeSize(gatherMaskShape), 1.0f);
    std::vector<int32_t> tileSumHostData(GetShapeSize(tileSumShape), 0);
    std::vector<float> tileDepthsHostData(GetShapeSize(tileDepthsShape), 0.0f);
    std::vector<float> gaussIndexHostData(GetShapeSize(gaussIndexShape), 0.0f);

    int ret = CreateAclTensor(means2dHostData, means2dShape, &resources.means2dDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.means2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(opacityHostData, opacityShape, &resources.opacityDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.opacityTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(conicsHostData, conicsShape, &resources.conicsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.conicsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(covars2dHostData, covars2dShape, &resources.covars2dDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.covars2dTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(depthsHostData, depthsShape, &resources.depthsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.depthsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(cntHostData, cntShape, &resources.cntDeviceAddr,
                          aclDataType::ACL_INT32, &resources.cntTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(tileGridHostData, tileGridShape, &resources.tileGridDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.tileGridTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gatherMaskHostData, gatherMaskShape, &resources.gatherMaskDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.gatherMaskTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(tileSumHostData, tileSumShape, &resources.tileSumDeviceAddr,
                          aclDataType::ACL_INT32, &resources.tileSumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(tileDepthsHostData, tileDepthsShape, &resources.tileDepthsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.tileDepthsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gaussIndexHostData, gaussIndexShape, &resources.gaussIndexDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.gaussIndexTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteFlashGaussianBuildMask(TensorResources& resources, aclrtStream stream,
                                   void** workspaceAddr, uint64_t* workspaceSize) {
    float imageWidth = 800.0f;
    float imageHeight = 600.0f;
    int64_t tileSize = 16;
    aclOpExecutor* executor;

    int ret = aclnnFlashGaussianBuildMaskGetWorkspaceSize(
        resources.means2dTensor, resources.opacityTensor, resources.conicsTensor,
        resources.covars2dTensor, resources.depthsTensor, resources.cntTensor,
        resources.tileGridTensor, resources.gatherMaskTensor, imageWidth, imageHeight, tileSize,
        resources.tileSumTensor, resources.tileDepthsTensor, resources.gaussIndexTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFlashGaussianBuildMaskGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnFlashGaussianBuildMask(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnFlashGaussianBuildMask failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.means2dTensor) aclDestroyTensor(resources.means2dTensor);
    if (resources.opacityTensor) aclDestroyTensor(resources.opacityTensor);
    if (resources.conicsTensor) aclDestroyTensor(resources.conicsTensor);
    if (resources.covars2dTensor) aclDestroyTensor(resources.covars2dTensor);
    if (resources.depthsTensor) aclDestroyTensor(resources.depthsTensor);
    if (resources.cntTensor) aclDestroyTensor(resources.cntTensor);
    if (resources.tileGridTensor) aclDestroyTensor(resources.tileGridTensor);
    if (resources.gatherMaskTensor) aclDestroyTensor(resources.gatherMaskTensor);
    if (resources.tileSumTensor) aclDestroyTensor(resources.tileSumTensor);
    if (resources.tileDepthsTensor) aclDestroyTensor(resources.tileDepthsTensor);
    if (resources.gaussIndexTensor) aclDestroyTensor(resources.gaussIndexTensor);

    if (resources.means2dDeviceAddr) aclrtFree(resources.means2dDeviceAddr);
    if (resources.opacityDeviceAddr) aclrtFree(resources.opacityDeviceAddr);
    if (resources.conicsDeviceAddr) aclrtFree(resources.conicsDeviceAddr);
    if (resources.covars2dDeviceAddr) aclrtFree(resources.covars2dDeviceAddr);
    if (resources.depthsDeviceAddr) aclrtFree(resources.depthsDeviceAddr);
    if (resources.cntDeviceAddr) aclrtFree(resources.cntDeviceAddr);
    if (resources.tileGridDeviceAddr) aclrtFree(resources.tileGridDeviceAddr);
    if (resources.gatherMaskDeviceAddr) aclrtFree(resources.gatherMaskDeviceAddr);
    if (resources.tileSumDeviceAddr) aclrtFree(resources.tileSumDeviceAddr);
    if (resources.tileDepthsDeviceAddr) aclrtFree(resources.tileDepthsDeviceAddr);
    if (resources.gaussIndexDeviceAddr) aclrtFree(resources.gaussIndexDeviceAddr);

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
    std::vector<int64_t> tileSumShape = {50};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteFlashGaussianBuildMask(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    PrintIntResult(tileSumShape, &resources.tileSumDeviceAddr, "tileSum");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}