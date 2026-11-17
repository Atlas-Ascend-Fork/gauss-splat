/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "color_exposure_fwd_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t COLORS_PTR_INDEX = 0;
const uint32_t MEANS2D_PTR_INDEX = 1;
const uint32_t EXPOSURE_PTR_INDEX = 2;

const uint32_t BATCH_DIM_INDEX = 0;
const uint32_t GAUSS_DIM_INDEX = 2;

const uint32_t GRID_WIDTH_DIM_INDEX = 0;
const uint32_t GRID_HEIGHT_DIM_INDEX = 1;
const uint32_t GRID_DEPTH_DIM_INDEX = 2;

const uint32_t WIDTH_INDEX = 0;
const uint32_t HEIGHT_INDEX = 1;
const uint32_t CLAMP_INDEX = 2;

const uint32_t ALIGN_VALUE = 8;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.8;

const uint32_t EXPOSURE_NUM = 12;
const uint32_t COLOR_NUM = 3;

constexpr uint32_t SF_USE_CLAMP = (1 << 0);
constexpr uint32_t SF_USE_GRID = (1 << 2);
constexpr uint32_t DCACHE_SIZE = 32 * 1024;
constexpr uint32_t MAX_GRID_DIMS = 16 * 16 * 8 * 12;
}

namespace optiling {
static ge::graphStatus InitColorExposureFwdTilingData(gert::TilingContext* context,
    ColorExposureFwdTilingData &tiling)
{
    auto colorsTensorPtr = context->GetInputTensor(COLORS_PTR_INDEX);
    auto means2dTensorPtr = context->GetInputTensor(MEANS2D_PTR_INDEX);
    auto exposureTensorPtr = context->GetInputTensor(EXPOSURE_PTR_INDEX);
    if (colorsTensorPtr == nullptr || means2dTensorPtr == nullptr
        || exposureTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto colorsShape = context->GetInputShape(COLORS_PTR_INDEX);
    auto expShape = context->GetInputShape(EXPOSURE_PTR_INDEX);
    if (colorsShape == nullptr || expShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t numelExp = 1;
    for (size_t i = 0; i < expShape->GetStorageShape().GetDimNum(); ++i) {
        numelExp *= expShape->GetStorageShape().GetDim(i);
    }

    uint32_t batchSize = colorsShape->GetStorageShape().GetDim(BATCH_DIM_INDEX);
    uint32_t gaussNum = colorsShape->GetStorageShape().GetDim(GAUSS_DIM_INDEX);

    auto attrsPtr = context->GetAttrs();
    uint32_t width = *(attrsPtr->GetAttrPointer<uint32_t>(WIDTH_INDEX));
    uint32_t height = *(attrsPtr->GetAttrPointer<uint32_t>(HEIGHT_INDEX));
    uint32_t clamp = *(attrsPtr->GetAttrPointer<uint32_t>(CLAMP_INDEX));
    uint32_t supportFeatures = clamp > 0 ? SF_USE_CLAMP : 0;
    uint32_t gridWidth = 1;
    uint32_t gridHeight = 1;
    uint32_t gridDepth = 1;
    bool useGrid = false;
    if (numelExp > EXPOSURE_NUM) {
        useGrid = true;
        supportFeatures |= SF_USE_GRID;
        gridWidth = expShape->GetStorageShape().GetDim(GRID_WIDTH_DIM_INDEX);
        gridHeight = expShape->GetStorageShape().GetDim(GRID_HEIGHT_DIM_INDEX);
        gridDepth = expShape->GetStorageShape().GetDim(GRID_DEPTH_DIM_INDEX);
    }

    if (useGrid && (gridWidth <= 1 || gridHeight <= 1 || gridDepth <= 1)) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_batchSize(batchSize);
    tiling.set_gaussNum(gaussNum);
    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_gridWidth(gridWidth);
    tiling.set_gridHeight(gridHeight);
    tiling.set_gridDepth(gridDepth);
    tiling.set_supportFeatures(supportFeatures);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalcColorExposureFwdTilingData(gert::TilingContext* context,
    ColorExposureFwdTilingData &tiling)
{
    uint32_t blockDim = tiling.get_blockDim();
    uint32_t gaussNum = tiling.get_gaussNum();
    uint64_t ubTotalSize = tiling.get_ubTotalSize();
    uint32_t totalTaskNum;
    if (static_cast<uint32_t>((gaussNum) % ALIGN_VALUE) == 0) {
        totalTaskNum = gaussNum;
    } else {
        totalTaskNum = (static_cast<uint32_t>(gaussNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    uint32_t tailNum = totalTaskNum - gaussNum;
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

    uint32_t gridWidth = tiling.get_gridWidth();
    uint32_t gridHeight = tiling.get_gridHeight();
    uint32_t gridDepth = tiling.get_gridDepth();
    bool useGrid = ((tiling.get_supportFeatures() & SF_USE_GRID) != 0);
    uint32_t expSize = (gridWidth * gridHeight * gridDepth) * EXPOSURE_NUM;
    if (expSize > MAX_GRID_DIMS) {
        return ge::GRAPH_FAILED;
    }
    expSize = expSize * FLOAT_SIZE;
    uint32_t bufferSize = 0;
    if (useGrid) {
        // colors input/output, ids, means2d, grey, exposure
        expSize += DCACHE_SIZE;
        context->SetDynUBufSize(ubTotalSize - DCACHE_SIZE);
        bufferSize = COLOR_NUM * 2 * 2 + 8;
    } else {
        // colors input/output
        bufferSize = COLOR_NUM * 2 * 2;
    }
    uint32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO - expSize)/ (bufferSize * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_totalTaskNum(totalTaskNum);
    tiling.set_tailNum(tailNum);
    tiling.set_taskNumPerScore(taskNumPerScore);
    tiling.set_taskNumPerLcore(taskNumPerLcore);
    tiling.set_numScore(numScore);
    tiling.set_numLcore(numLcore);
    tiling.set_taskNumPerLoop(taskNumPerLoop);
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForColorExposureFwd(gert::TilingContext* context)
{
    ColorExposureFwdTilingData tiling;
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto status = InitColorExposureFwdTilingData(context, tiling);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

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
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);
    status = CalcColorExposureFwdTilingData(context, tiling);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

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

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(COLORS_PTR_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class ColorExposureFwd : public OpDef {
public:
    explicit ColorExposureFwd(const char* name) : OpDef(name)
    {
        this->Input("colors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("means2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("exposure")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("width").Int();
        this->Attr("height").Int();
        this->Attr("clamp").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForColorExposureFwd);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(ColorExposureFwd);
}