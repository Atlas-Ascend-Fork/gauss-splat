/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "spherical_harmonics_forward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t DIRS_PTR_INDEX = 0;
const uint32_t COEFFS_PTR_INDEX = 1;
const uint32_t OUTPUT_PTR_INDEX = 0;

const uint32_t L2_SH_DEGREE = 2;
const uint32_t L3_SH_DEGREE = 3;
const uint32_t L4_SH_DEGREE = 4;
const uint32_t L2_SH_BUFFER_NUM = 6;
const uint32_t L3_SH_BUFFER_NUM = 10;
const uint32_t L4_SH_BUFFER_NUM = 16;

const uint32_t BATCH_DIM_INDEX = 0;
const uint32_t CHANNEL_DIM_INDEX = 1;
const uint32_t GAUSSIAN_DIM_INDEX = 2;
const uint32_t COEFF_NUM_DIM_INDEX = 1;
const uint32_t COEFF_CHANNEL_DIM_INDEX = 2;

// const uint32_t DEGREE_USED_INDEX = 0;
const uint32_t AXIS_NUM = 3;
const uint32_t MIN_CHANNEL_NUM = 1;
const uint32_t MAX_CHANNEL_NUM = 4;
const uint32_t ALIGN_VALUE = 8;

// const uint32_t BN_NUM = 6;
// const uint32_t BC_NUM = 4;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.8;
}

namespace optiling {
static ge::graphStatus TilingForSphericalHarmonicsForward(gert::TilingContext* context)
{
    SphericalHarmonicsForwardTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dirsTensorPtr = context->GetInputTensor(DIRS_PTR_INDEX);
    auto coeffsTensorPtr = context->GetInputTensor(COEFFS_PTR_INDEX);
    if (dirsTensorPtr == nullptr || coeffsTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dirsShape = context->GetInputShape(DIRS_PTR_INDEX);
    auto coeffsShape = context->GetInputShape(COEFFS_PTR_INDEX);
    if (dirsShape == nullptr || coeffsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // dirs: [B, 3, N], coeffs: [B, K, C, N], C in [1, 4]
    uint32_t batchSize = dirsShape->GetStorageShape().GetDim(BATCH_DIM_INDEX);
    uint32_t gaussianNum = dirsShape->GetStorageShape().GetDim(GAUSSIAN_DIM_INDEX);
    uint32_t taskNum = gaussianNum;
    uint32_t coeffNum = coeffsShape->GetStorageShape().GetDim(COEFF_NUM_DIM_INDEX);
    uint32_t channelNum = coeffsShape->GetStorageShape().GetDim(COEFF_CHANNEL_DIM_INDEX);
    if (channelNum < MIN_CHANNEL_NUM || channelNum > MAX_CHANNEL_NUM) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t degreeUsed = *(attrsPtr->GetAttrPointer<int32_t>(0));
    uint32_t degreeNum = (degreeUsed + 1) * (degreeUsed + 1);

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint64_t ubTotalSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotalSize);
    uint32_t blockDim = platformInfo.GetCoreNumAiv();
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalTaskNum;
    if (static_cast<uint32_t>((taskNum) % ALIGN_VALUE) == 0) {
        totalTaskNum = taskNum;
    } else {
        totalTaskNum = (static_cast<uint32_t>(taskNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    if (degreeNum == 1) {
        const uint32_t minTaskPerCore = 1170;
        uint32_t gridFromTaskNum = (totalTaskNum + minTaskPerCore - 1) / minTaskPerCore;
        if (gridFromTaskNum < 1) {
            gridFromTaskNum = 1;
        }
        if (gridFromTaskNum < blockDim) {
            blockDim = gridFromTaskNum;
        }
    }

    uint32_t tailNum = totalTaskNum - taskNum;
    uint32_t taskNumPerScore = (totalTaskNum / blockDim / ALIGN_VALUE) * ALIGN_VALUE;
    uint32_t taskNumPerLcore = taskNumPerScore + ALIGN_VALUE;
    uint32_t numScore = (blockDim * taskNumPerLcore - totalTaskNum) / ALIGN_VALUE;
    uint32_t numLcore = blockDim - numScore;

    if (taskNumPerScore == 0) {
        blockDim = blockDim - numScore;
    }
    if (taskNumPerLcore == 0) {
        blockDim = blockDim - numLcore;
    }

    // degree 1-4 UB budget (s1-degree14-tile-pipeline): dirs (3) + out (C) +
    // coeffs (C*K) per task. The K-per-task basis staging buffer, the 6/10/16-deep
    // computing pool and any recurrence reserve are gone (the basis runs in
    // registers); the depth-2 ping-pong splits a tile into two half-size slots, so
    // the physical queue footprint equals this per-task budget.
    uint32_t bufferSize;
    if (degreeNum == 1) {
        // degree-0-only UB budget: coeffs (C*K) + output (C) per task; the kernel
        // splits the tile into a depth-2 ping-pong of the same total footprint, so
        // no dirs/result/computing tensors are reserved for the degree-0 path.
        bufferSize = channelNum * degreeNum + channelNum;
    } else {
        bufferSize = AXIS_NUM + channelNum + channelNum * degreeNum;
    }
    uint32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (bufferSize * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_batchSize(batchSize);
    tiling.set_gaussianNum(gaussianNum);
    tiling.set_taskNum(taskNum);
    tiling.set_coeffNum(coeffNum);
    tiling.set_channelNum(channelNum);
    tiling.set_degreeUsed(degreeUsed);
    tiling.set_degreeNum(degreeNum);
    tiling.set_totalTaskNum(totalTaskNum);
    tiling.set_tailNum(tailNum);
    tiling.set_taskNumPerScore(taskNumPerScore);
    tiling.set_taskNumPerLcore(taskNumPerLcore);
    tiling.set_numScore(numScore);
    tiling.set_numLcore(numLcore);
    tiling.set_taskNumPerLoop(taskNumPerLoop);
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1);
    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* dirsShape = context->GetInputShape(DIRS_PTR_INDEX);
    // if (dirsShape == nullptr) {
    //         return ge::GRAPH_FAILED;
    //     }

    const gert::Shape* coeffsShape = context->GetInputShape(COEFFS_PTR_INDEX);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_PTR_INDEX);

    if (dirsShape == nullptr || coeffsShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t channelNum = coeffsShape->GetDim(COEFF_CHANNEL_DIM_INDEX);
    if (channelNum < MIN_CHANNEL_NUM || channelNum > MAX_CHANNEL_NUM) {
        return ge::GRAPH_FAILED;
    }

    // output: [B, C, N]
    *outputShape = *dirsShape;
    outputShape->SetDim(CHANNEL_DIM_INDEX, channelNum);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SphericalHarmonicsForward : public OpDef {
public:
    explicit SphericalHarmonicsForward(const char* name) : OpDef(name)
    {
        this->Input("dirs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("coeffs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("degrees_to_use").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForSphericalHarmonicsForward);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SphericalHarmonicsForward);
}
