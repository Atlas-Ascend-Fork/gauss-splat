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

#include "projection_three_dims_gaussian_forward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t MEANS_PTR_INDEX = 0;
const uint32_t COVARS_PTR_INDEX = 1;
const uint32_t VIEWMATS_PTR_INDEX = 2;
const uint32_t KS_PTR_INDEX = 3;

const uint32_t MEANS2D_PTR_INDEX = 0;
const uint32_t DEPTHS_PTR_INDEX = 1;
const uint32_t CONICS_PTR_INDEX = 2;
const uint32_t COMPENSATIOONS_PTR_INDEX = 3;
const uint32_t DET_PTR_INDEX = 4;
const uint32_t RADIUS_PTR_INDEX = 5;
const uint32_t COVARS2D_PTR_INDEX = 6;

const uint32_t IMAGE_WIDTH_INDEX = 0;
const uint32_t IMAGE_HEIGHT_INDEX = 1;
const uint32_t EPS2D_INDEX = 2;
const uint32_t CALC_COMPENSATIONS_INDEX = 3;
const uint32_t CAMERA_MODEL_INDEX = 4;

const uint32_t BATCH_SIZE_INDEX = 0;
const uint32_t CHANNEL_NUM_INDEX = 1;
const uint32_t GAUSSIAN_NUM_INDEX = 2;
const uint32_t ALIGN_VALUE = 64;

const uint32_t BUFFER_NUM = 42;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.85;
}

namespace optiling {
static ge::graphStatus TilingForProjectionThreeDimsGaussianForward(gert::TilingContext* context)
{
    ProjectionThreeDimsGaussianForwardTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto meansTensorPtr = context->GetInputTensor(MEANS_PTR_INDEX);
    auto covarsTensorPtr = context->GetInputTensor(COVARS_PTR_INDEX);
    auto viewmatsTensorPtr = context->GetInputTensor(VIEWMATS_PTR_INDEX);
    auto ksTensorPtr = context->GetInputTensor(KS_PTR_INDEX);
    if (meansTensorPtr == nullptr || covarsTensorPtr == nullptr \
        || viewmatsTensorPtr == nullptr || ksTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto meansShape = context->GetInputShape(MEANS_PTR_INDEX);
    auto covarsShape = context->GetInputShape(COVARS_PTR_INDEX);
    auto viewmatsShape = context->GetInputShape(VIEWMATS_PTR_INDEX);
    auto ksShape = context->GetInputShape(KS_PTR_INDEX);
    if (meansShape == nullptr || covarsShape == nullptr \
        || viewmatsShape == nullptr || ksShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t imageWidth = *(attrsPtr->GetAttrPointer<uint32_t>(IMAGE_WIDTH_INDEX));
    int32_t imageHeight = *(attrsPtr->GetAttrPointer<uint32_t>(IMAGE_HEIGHT_INDEX));
    float eps2d = *(attrsPtr->GetAttrPointer<float>(EPS2D_INDEX));
    bool calcCompensations = *(attrsPtr->GetAttrPointer<bool>(CALC_COMPENSATIONS_INDEX));
    uint32_t cameraModel = *(attrsPtr->GetAttrPointer<uint32_t>(CAMERA_MODEL_INDEX));

    uint32_t batchSizeNum = meansShape->GetStorageShape().GetDim(BATCH_SIZE_INDEX);
    uint32_t gaussianNum = meansShape->GetStorageShape().GetDim(GAUSSIAN_NUM_INDEX);
    uint32_t cameraNum = viewmatsShape->GetStorageShape().GetDim(CHANNEL_NUM_INDEX);

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

    uint32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (BUFFER_NUM * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_imageWidth(static_cast<float>(imageWidth));
    tiling.set_imageHeight(static_cast<float>(imageHeight));
    tiling.set_cameraModel(cameraModel);
    tiling.set_eps2d(eps2d);
    tiling.set_calcCompensations(calcCompensations);

    tiling.set_batchSizeNum(batchSizeNum);
    tiling.set_cameraNum(cameraNum);
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

    const gert::Shape* meansShape = context->GetInputShape(MEANS_PTR_INDEX);
    const gert::Shape* covarsShape = context->GetInputShape(COVARS_PTR_INDEX);
    const gert::Shape* viewmatsShape = context->GetInputShape(VIEWMATS_PTR_INDEX);
    const gert::Shape* ksShape = context->GetInputShape(KS_PTR_INDEX);

    if (meansShape == nullptr || covarsShape == nullptr \
        || viewmatsShape == nullptr || ksShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* means2dShape = context->GetOutputShape(MEANS2D_PTR_INDEX);
    gert::Shape* depthsShape = context->GetOutputShape(DEPTHS_PTR_INDEX);
    gert::Shape* conicsShape = context->GetOutputShape(CONICS_PTR_INDEX);
    gert::Shape* compensationsShape = context->GetOutputShape(COMPENSATIOONS_PTR_INDEX);
    gert::Shape* detShape = context->GetOutputShape(DET_PTR_INDEX);
    gert::Shape* radiusShape = context->GetOutputShape(RADIUS_PTR_INDEX);
    gert::Shape* covars2dShape = context->GetOutputShape(COVARS2D_PTR_INDEX);

    uint32_t batchSizeNum = meansShape->GetDim(BATCH_SIZE_INDEX);
    uint32_t gaussianNum = meansShape->GetDim(GAUSSIAN_NUM_INDEX);
    uint32_t cameraNum = viewmatsShape->GetDim(CHANNEL_NUM_INDEX);
    *means2dShape = {batchSizeNum, cameraNum, 2, gaussianNum};
    *depthsShape = {batchSizeNum, cameraNum, 1, gaussianNum};
    *conicsShape = {batchSizeNum, cameraNum, 3, gaussianNum};
    *compensationsShape = {batchSizeNum, cameraNum, 1, gaussianNum};
    *detShape = {batchSizeNum, cameraNum, 1, gaussianNum};
    *radiusShape = {batchSizeNum, cameraNum, 2, gaussianNum};
    *covars2dShape = {batchSizeNum, cameraNum, 3, gaussianNum};
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
class ProjectionThreeDimsGaussianForward : public OpDef {
public:
    explicit ProjectionThreeDimsGaussianForward(const char* name) : OpDef(name)
    {
        this->Input("means")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("covars")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("viewmats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("ks")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("means2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("depths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("conics")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("compensations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("det")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("radius")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("covars2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("image_width").Int();
        this->Attr("image_height").Int();
        this->Attr("eps2d").Float();
        this->Attr("calc_compensations").Bool();
        this->Attr("camera_model").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForProjectionThreeDimsGaussianForward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ProjectionThreeDimsGaussianForward);
}
