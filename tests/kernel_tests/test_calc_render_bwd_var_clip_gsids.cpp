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
#include "aclnnop/aclnn_calc_render_bwd_var_clip_gsids.h"
#include "test_utils.h"

using namespace std;

namespace {

struct TensorResources {
    void* vColorDeviceAddr = nullptr;
    void* vDepthDeviceAddr = nullptr;
    void* lastCumsumDeviceAddr = nullptr;
    void* errorDeviceAddr = nullptr;
    void* gsDeviceAddr = nullptr;
    void* tileCoordsDeviceAddr = nullptr;
    void* offsetsDeviceAddr = nullptr;
    void* gsClipIndex_gsIdsDeviceAddr = nullptr;
    void* alphaClipIndexDeviceAddr = nullptr;
    void* vGsDeviceAddr = nullptr;

    aclTensor* vColorTensor = nullptr;
    aclTensor* vDepthTensor = nullptr;
    aclTensor* lastCumsumTensor = nullptr;
    aclTensor* errorTensor = nullptr;
    aclTensor* gsTensor = nullptr;
    aclTensor* tileCoordsTensor = nullptr;
    aclTensor* offsetsTensor = nullptr;
    aclTensor* gsClipIndex_gsIdsTensor = nullptr;
    aclTensor* alphaClipIndexTensor = nullptr;
    aclTensor* vGsTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t nGauss = 100;
    int64_t tileNum = 4;
    int64_t nPixel = 64;
    
    std::vector<int64_t> vColorShape = {nPixel, 3};
    std::vector<int64_t> vDepthShape = {nPixel, 1};
    std::vector<int64_t> lastCumsumShape = {nPixel, 1};
    std::vector<int64_t> errorShape = {nPixel, 1};
    std::vector<int64_t> gsShape = {nGauss, 10};
    std::vector<int64_t> tileCoordsShape = {tileNum, 2, nPixel};
    std::vector<int64_t> offsetsShape = {tileNum + 1};
    std::vector<int64_t> gsClipIndex_gsIdsShape = {tileNum, nGauss};
    std::vector<int64_t> alphaClipIndexShape = {nPixel};
    std::vector<int64_t> vGsShape = {nGauss, 10};

    std::vector<float> vColorHostData(GetShapeSize(vColorShape), 1.0f);
    std::vector<float> vDepthHostData(GetShapeSize(vDepthShape), 1.0f);
    std::vector<float> lastCumsumHostData(GetShapeSize(lastCumsumShape), 0.5f);
    std::vector<float> errorHostData(GetShapeSize(errorShape), 0.0f);
    std::vector<float> gsHostData(GetShapeSize(gsShape), 0.0f);
    std::vector<float> tileCoordsHostData(GetShapeSize(tileCoordsShape), 0.0f);

    std::vector<int64_t> offsetsHostData(tileNum + 1, 0);
    for (int64_t i = 0; i <= tileNum; i++) {
        offsetsHostData[i] = i * (nGauss / tileNum);
    }

    std::vector<int64_t> gsClipIndexGsIdsHostData(GetShapeSize(gsClipIndex_gsIdsShape), 0);
    std::vector<uint8_t> alphaClipIndexHostData(GetShapeSize(alphaClipIndexShape), 0);
    std::vector<float> vGsHostData(GetShapeSize(vGsShape), 0.0f);

    int ret = CreateAclTensor(vColorHostData, vColorShape, &resources.vColorDeviceAddr,
                              aclDataType::ACL_FLOAT, &resources.vColorTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(vDepthHostData, vDepthShape, &resources.vDepthDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.vDepthTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(lastCumsumHostData, lastCumsumShape, &resources.lastCumsumDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.lastCumsumTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(errorHostData, errorShape, &resources.errorDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.errorTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gsHostData, gsShape, &resources.gsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.gsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(tileCoordsHostData, tileCoordsShape, &resources.tileCoordsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.tileCoordsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(offsetsHostData, offsetsShape, &resources.offsetsDeviceAddr,
                          aclDataType::ACL_INT64, &resources.offsetsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(gsClipIndexGsIdsHostData, gsClipIndex_gsIdsShape, &resources.gsClipIndex_gsIdsDeviceAddr,
                          aclDataType::ACL_INT64, &resources.gsClipIndex_gsIdsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(alphaClipIndexHostData, alphaClipIndexShape, &resources.alphaClipIndexDeviceAddr,
                          aclDataType::ACL_UINT8, &resources.alphaClipIndexTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = CreateAclTensor(vGsHostData, vGsShape, &resources.vGsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.vGsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    return ACL_SUCCESS;
}

int ExecuteCalcRenderBwdVarClipGsids(TensorResources& resources, aclrtStream stream,
                                     void** workspaceAddr, uint64_t* workspaceSize) {
    aclOpExecutor* executor;

    int ret = aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize(
        resources.vColorTensor, resources.vDepthTensor, resources.lastCumsumTensor, resources.errorTensor,
        resources.gsTensor, resources.tileCoordsTensor, resources.offsetsTensor,
        resources.gsClipIndex_gsIdsTensor, resources.alphaClipIndexTensor,
        resources.vGsTensor, workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;
    }

    ret = aclnnCalcRenderBwdVarClipGsids(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnCalcRenderBwdVarClipGsids failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.vColorTensor) aclDestroyTensor(resources.vColorTensor);
    if (resources.vDepthTensor) aclDestroyTensor(resources.vDepthTensor);
    if (resources.lastCumsumTensor) aclDestroyTensor(resources.lastCumsumTensor);
    if (resources.errorTensor) aclDestroyTensor(resources.errorTensor);
    if (resources.gsTensor) aclDestroyTensor(resources.gsTensor);
    if (resources.tileCoordsTensor) aclDestroyTensor(resources.tileCoordsTensor);
    if (resources.offsetsTensor) aclDestroyTensor(resources.offsetsTensor);
    if (resources.gsClipIndex_gsIdsTensor) aclDestroyTensor(resources.gsClipIndex_gsIdsTensor);
    if (resources.alphaClipIndexTensor) aclDestroyTensor(resources.alphaClipIndexTensor);
    if (resources.vGsTensor) aclDestroyTensor(resources.vGsTensor);

    if (resources.vColorDeviceAddr) aclrtFree(resources.vColorDeviceAddr);
    if (resources.vDepthDeviceAddr) aclrtFree(resources.vDepthDeviceAddr);
    if (resources.lastCumsumDeviceAddr) aclrtFree(resources.lastCumsumDeviceAddr);
    if (resources.errorDeviceAddr) aclrtFree(resources.errorDeviceAddr);
    if (resources.gsDeviceAddr) aclrtFree(resources.gsDeviceAddr);
    if (resources.tileCoordsDeviceAddr) aclrtFree(resources.tileCoordsDeviceAddr);
    if (resources.offsetsDeviceAddr) aclrtFree(resources.offsetsDeviceAddr);
    if (resources.gsClipIndex_gsIdsDeviceAddr) aclrtFree(resources.gsClipIndex_gsIdsDeviceAddr);
    if (resources.alphaClipIndexDeviceAddr) aclrtFree(resources.alphaClipIndexDeviceAddr);
    if (resources.vGsDeviceAddr) aclrtFree(resources.vGsDeviceAddr);

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
    std::vector<int64_t> vGsShape = {100, 8};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) return ret;

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteCalcRenderBwdVarClipGsids(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
    }

    PrintResult<float>(vGsShape, &resources.vGsDeviceAddr, "vGs");

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}