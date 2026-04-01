/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include "flash_gaussian_build_mask_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t MEANS2D_PTR_INDEX = 0;
const uint32_t OPACITY_PTR_INDEX = 1;
const uint32_t CONICS_PTR_INDEX = 2;
const uint32_t COVARS2D_PTR_INDEX = 3;
const uint32_t CNT_PTR_INDEX = 4;
const uint32_t TILE_GRID_PTR_INDEX = 5;
const uint32_t MASK_PTR_INDEX = 0;

const uint32_t BATCH_SIZE_INDEX = 0;
const uint32_t CAMERA_NUM_INDEX = 1;
const uint32_t GAUSS_NUM_INDEX = 3;
const uint32_t TILE_NUM_INDEX = 0;
const uint32_t IMAGE_WIDTH_INDEX = 0;
const uint32_t IMAGE_HEIGHT_INDEX = 1;
const uint32_t TILE_SIZE_INDEX = 2;
const uint32_t ALIGN_VALUE = 64;

const uint32_t TASK_BUFFER_NUM = 16;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.85;
}

namespace optiling {
static ge::graphStatus TilingForFlashGaussianBuildMask(gert::TilingContext* context)
{
    FlashGaussianBuildMaskTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto means2dTensorPtr = context->GetInputTensor(MEANS2D_PTR_INDEX);
    auto opacityTensorPtr = context->GetInputTensor(OPACITY_PTR_INDEX);
    auto conicsTensorPtr = context->GetInputTensor(CONICS_PTR_INDEX);
    auto covars2dTensorPtr = context->GetInputTensor(COVARS2D_PTR_INDEX);
    auto cntTensorPtr = context->GetInputTensor(CNT_PTR_INDEX);
    auto tilegridTensorPtr = context->GetInputTensor(TILE_GRID_PTR_INDEX);
    if (means2dTensorPtr == nullptr || opacityTensorPtr == nullptr || \
        conicsTensorPtr == nullptr || covars2dTensorPtr == nullptr || \
        cntTensorPtr == nullptr || tilegridTensorPtr ==nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto means2dShape = context->GetInputShape(MEANS2D_PTR_INDEX);
    auto opacityShape = context->GetInputShape(OPACITY_PTR_INDEX);
    auto conicsShape = context->GetInputShape(CONICS_PTR_INDEX);
    auto covars2dShape = context->GetInputShape(COVARS2D_PTR_INDEX);
    auto cntShape = context->GetInputShape(CNT_PTR_INDEX);
    auto tilegridShape = context->GetInputShape(TILE_GRID_PTR_INDEX);
    if (means2dShape == nullptr || opacityShape == nullptr || \
        conicsShape == nullptr || covars2dShape == nullptr || \
        cntShape == nullptr || tilegridShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = opacityShape->GetStorageShape().GetDim(BATCH_SIZE_INDEX);
    uint32_t cameraNum = opacityShape->GetStorageShape().GetDim(CAMERA_NUM_INDEX);
    uint32_t gaussNum = opacityShape->GetStorageShape().GetDim(GAUSS_NUM_INDEX);
    uint32_t numTile = tilegridShape->GetStorageShape().GetDim(TILE_NUM_INDEX);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float imageWidth = *(attrsPtr->GetAttrPointer<uint32_t>(IMAGE_WIDTH_INDEX));
    float imageHeight = *(attrsPtr->GetAttrPointer<uint32_t>(IMAGE_HEIGHT_INDEX));
    float tileSize = static_cast<float>(*(attrsPtr->GetAttrPointer<uint32_t>(TILE_SIZE_INDEX)));

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

    // 以numTile为总任务数进行分核
    uint32_t tileNumPerScore = (numTile / blockDim);
    uint32_t tileNumPerLcore = tileNumPerScore + 1;
    uint32_t numScore = (blockDim * tileNumPerLcore - numTile);
    uint32_t numLcore = blockDim - numScore;
    if (tileNumPerScore == 0) {
        blockDim = blockDim - numScore;
    }
    if (tileNumPerLcore == 0) {
        blockDim = blockDim - numLcore;
    }

    // 确定单核上的高斯球循环次数，避免ub溢出，向上对齐到32B，方便进行搬运。
    uint32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (TASK_BUFFER_NUM * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_numTile(numTile);
    tiling.set_batchSize(batchSize);
    tiling.set_cameraNum(cameraNum);
    tiling.set_gaussNum(gaussNum);
    tiling.set_tileNumPerScore(tileNumPerScore);
    tiling.set_tileNumPerLcore(tileNumPerLcore);
    tiling.set_numScore(numScore);
    tiling.set_numLcore(numLcore);
    tiling.set_taskNumPerLoop(taskNumPerLoop);
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);
    tiling.set_imageWidth(imageWidth);
    tiling.set_imageHeight(imageHeight);
    tiling.set_tileSize(tileSize);

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

    const gert::Shape* means2dShape = context->GetInputShape(MEANS2D_PTR_INDEX);
    const gert::Shape* cntShape = context->GetInputShape(CNT_PTR_INDEX);
    const gert::Shape* opacityShape = context->GetInputShape(OPACITY_PTR_INDEX);
    const gert::Shape* conicsShape = context->GetInputShape(CONICS_PTR_INDEX);
    const gert::Shape* covars2dShape = context->GetInputShape(COVARS2D_PTR_INDEX);
    const gert::Shape* tilegridShape = context->GetInputShape(TILE_GRID_PTR_INDEX);
    if (means2dShape == nullptr || cntShape == nullptr || opacityShape == nullptr \
        || conicsShape == nullptr || covars2dShape == nullptr || tilegridShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
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
class FlashGaussianBuildMask : public OpDef {
public:
    explicit FlashGaussianBuildMask(const char* name) : OpDef(name)
    {
        this->Input("means2d")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("opacity")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conics")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("covars2d")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tile_grid")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("image_width").Float();
        this->Attr("image_height").Float();
        this->Attr("tile_size").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForFlashGaussianBuildMask);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FlashGaussianBuildMask);
}
