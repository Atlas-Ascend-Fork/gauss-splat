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
#include "aclnnop/aclnn_gaussian_sort.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* lbSchedDeviceAddr = nullptr;
    void* gaussianCntDeviceAddr = nullptr;
    void* gsIdsDeviceAddr = nullptr;
    void* depthsDeviceAddr = nullptr;
    void* sortedOffsetDeviceAddr = nullptr;
    void* sortedGsIdsDeviceAddr = nullptr;

    aclTensor* lbSchedTensor = nullptr;
    aclTensor* gaussianCntTensor = nullptr;
    aclTensor* gsIdsTensor = nullptr;
    aclTensor* depthsTensor = nullptr;
    aclTensor* sortedOffsetTensor = nullptr;
    aclTensor* sortedGsIdsTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t batchSize = 1;
    int64_t cameraNum = 1;
    int64_t tileNum = 16;
    int64_t gaussNum = 256;
    int64_t cntPerTile = 10;
    // schedule layout: [num_bins bin end offsets][tileNum scheduled tile ids][tileNum cumulative offsets]
    int64_t numBins = tileNum;
    int64_t scheduleNum = numBins + 2 * tileNum;

    std::vector<int64_t> lbSchedShape = {batchSize, cameraNum, scheduleNum};
    std::vector<int64_t> gaussianCntShape = {batchSize, cameraNum, tileNum};
    std::vector<int64_t> gsIdsShape = {batchSize, cameraNum, tileNum, gaussNum};
    std::vector<int64_t> depthsShape = {batchSize, cameraNum, tileNum, gaussNum};
    std::vector<int64_t> sortedOffsetShape = {batchSize, cameraNum, 1};
    std::vector<int64_t> sortedGsIdsShape = {tileNum*gaussNum};

    std::vector<int64_t> lbSchedHostData(GetShapeSize(lbSchedShape), 0);
    for (int64_t i = 0; i < numBins; i++) {
        lbSchedHostData[i] = i + 1;                              // bin end offsets: one tile per bin
        lbSchedHostData[numBins + i] = i;                       // scheduled tile ids
        lbSchedHostData[numBins + tileNum + i] = (i + 1) * cntPerTile; // cumulative gauss counts
    }
    std::vector<int32_t> gaussianCntHostData(GetShapeSize(gaussianCntShape), static_cast<int32_t>(cntPerTile));
    std::vector<float> gsIdsHostData(GetShapeSize(gsIdsShape), 0.0f);
    std::vector<float> depthsHostData(GetShapeSize(depthsShape), 1.0f);
    std::vector<int64_t> sortedOffsetHostData(GetShapeSize(sortedOffsetShape), 0);
    std::vector<int32_t> sortedGsIdsHostData(GetShapeSize(sortedGsIdsShape), 0);

    int ret = CreateAclTensor(lbSchedHostData, lbSchedShape, &resources.lbSchedDeviceAddr,
                              aclDataType::ACL_INT64, &resources.lbSchedTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gaussianCntHostData, gaussianCntShape, &resources.gaussianCntDeviceAddr,
                          aclDataType::ACL_INT32, &resources.gaussianCntTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gsIdsHostData, gsIdsShape, &resources.gsIdsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.gsIdsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(depthsHostData, depthsShape, &resources.depthsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.depthsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(sortedOffsetHostData, sortedOffsetShape, &resources.sortedOffsetDeviceAddr,
                          aclDataType::ACL_INT64, &resources.sortedOffsetTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(sortedGsIdsHostData, sortedGsIdsShape, &resources.sortedGsIdsDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sortedGsIdsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteGaussianSort(TensorResources& resources, aclrtStream stream,
                        void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t maxTileGauss = 64;
    aclOpExecutor* executor;

    int ret = aclnnGaussianSortGetWorkspaceSize(
        resources.lbSchedTensor, resources.gaussianCntTensor, resources.gsIdsTensor,
        resources.depthsTensor, resources.sortedOffsetTensor, maxTileGauss,
        resources.sortedGsIdsTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGaussianSortGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnGaussianSort(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnGaussianSort failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.lbSchedTensor) aclDestroyTensor(resources.lbSchedTensor);
    if (resources.gaussianCntTensor) aclDestroyTensor(resources.gaussianCntTensor);
    if (resources.gsIdsTensor) aclDestroyTensor(resources.gsIdsTensor);
    if (resources.depthsTensor) aclDestroyTensor(resources.depthsTensor);
    if (resources.sortedOffsetTensor) aclDestroyTensor(resources.sortedOffsetTensor);
    if (resources.sortedGsIdsTensor) aclDestroyTensor(resources.sortedGsIdsTensor);

    if (resources.lbSchedDeviceAddr) aclrtFree(resources.lbSchedDeviceAddr);
    if (resources.gaussianCntDeviceAddr) aclrtFree(resources.gaussianCntDeviceAddr);
    if (resources.gsIdsDeviceAddr) aclrtFree(resources.gsIdsDeviceAddr);
    if (resources.depthsDeviceAddr) aclrtFree(resources.depthsDeviceAddr);
    if (resources.sortedOffsetDeviceAddr) aclrtFree(resources.sortedOffsetDeviceAddr);
    if (resources.sortedGsIdsDeviceAddr) aclrtFree(resources.sortedGsIdsDeviceAddr);

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

    ret = ExecuteGaussianSort(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    aclrtSynchronizeStream(stream);
    
    std::vector<int64_t> sortedGsIdsShape = {1, 1, 16, 256};
    PrintIntResult(sortedGsIdsShape, &resources.sortedGsIdsDeviceAddr, "sortedGsIds");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}