/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quat_scales_to_covars_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t QUAT_PTR_INDEX = 0;
const uint32_t SCALES_PTR_INDEX = 1;
const uint32_t COVARS_PTR_INDEX = 0;

const uint32_t BATCH_SIZE_INDEX = 0;
const uint32_t GAUSSIAN_NUM_INDEX = 2;
const uint32_t ALIGN_VALUE = 8;

const uint32_t BN_NUM = 25;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.8;
}

namespace optiling {
static ge::graphStatus TilingForQuatScalesToCovars(gert::TilingContext* context)
{
    QuatScalesToCovarsTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto quatTensorPtr = context->GetInputTensor(QUAT_PTR_INDEX);
    auto scalesTensorPtr = context->GetInputTensor(SCALES_PTR_INDEX);
    if (quatTensorPtr == nullptr || scalesTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto quatShape = context->GetInputShape(QUAT_PTR_INDEX);
    auto scalesShape = context->GetInputShape(SCALES_PTR_INDEX);
    if (quatShape == nullptr || scalesShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSizeNum = quatShape->GetStorageShape().GetDim(BATCH_SIZE_INDEX);
    uint32_t gaussianNum = quatShape->GetStorageShape().GetDim(GAUSSIAN_NUM_INDEX);

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
    if (static_cast<uint32_t>((gaussianNum) % ALIGN_VALUE) == 0) {
        totalTaskNum = gaussianNum;
    } else {
        totalTaskNum = (static_cast<uint32_t>(gaussianNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    uint32_t tailNum = totalTaskNum - gaussianNum;
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

    uint32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (BN_NUM * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_batchSizeNum(batchSizeNum);
    tiling.set_gaussianNum(gaussianNum);
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

    const gert::Shape* quatShape = context->GetInputShape(QUAT_PTR_INDEX);
    const gert::Shape* scalesShape = context->GetInputShape(SCALES_PTR_INDEX);
    if (quatShape == nullptr || scalesShape == nullptr) {
            return ge::GRAPH_FAILED;
        }

    gert::Shape* covarsShape = context->GetOutputShape(COVARS_PTR_INDEX);
    uint32_t batchSizeNum = quatShape->GetDim(BATCH_SIZE_INDEX);
    uint32_t gaussianNum = quatShape->GetDim(GAUSSIAN_NUM_INDEX);
    *covarsShape = {batchSizeNum, 3, 3, gaussianNum};
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
class QuatScalesToCovars : public OpDef {
public:
    explicit QuatScalesToCovars(const char* name) : OpDef(name)
    {
        this->Input("quat")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scales")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("covars")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForQuatScalesToCovars);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(QuatScalesToCovars);
}
