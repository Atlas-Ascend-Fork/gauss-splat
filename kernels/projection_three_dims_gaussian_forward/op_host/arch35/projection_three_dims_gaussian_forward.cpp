/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "projection_three_dims_gaussian_forward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t MEANS_PTR_INDEX = 0;
const uint32_t COVARS_PTR_INDEX = 1;
const uint32_t VIEWMATS_PTR_INDEX = 2;
const uint32_t KS_PTR_INDEX = 3;
// Optional quat (B,N,4) / scales (B,N,3) inputs feeding the inlined
// quat_scales_to_covars path. Either covars OR (quat AND scales) must be
// provided; if both, quat/scales wins (inputIsQuatScales=1).
const uint32_t QUAT_PTR_INDEX = 4;
const uint32_t SCALES_PTR_INDEX = 5;

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

const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.85;
// Per-task UB budget, each buffer taskNumPerLoop floats (see op_kernel Init):
//   MeansTensor 3 + CovarsTensor 9 + ComputingTensor 12
//   + MeansWorld 3 + CovarsWorld 9 = 36
// Plus the fixed 64-float CameraParam slot.
const uint32_t PER_TASK_FLOAT_BUFFERS = 36;
// The quat+scales path overlays QuatTensor/ScalesTensor onto CovarsTensor's
// UB region (time-multiplex via a LocalTensor alias in kernel Init()), so it
// needs no extra per-task floats and keeps the same tile as the covars path.
const uint32_t PER_TASK_FLOAT_BUFFERS_QS = 36;
const uint32_t FIXED_UB_BYTES = 64 * FLOAT_SIZE;
}

namespace optiling {
static ge::graphStatus TilingForProjectionThreeDimsGaussianForward(gert::TilingContext* context)
{
    ProjectionThreeDimsGaussianForwardTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto meansTensorPtr = context->GetInputTensor(MEANS_PTR_INDEX);
    auto viewmatsTensorPtr = context->GetInputTensor(VIEWMATS_PTR_INDEX);
    auto ksTensorPtr = context->GetInputTensor(KS_PTR_INDEX);
    if (meansTensorPtr == nullptr || viewmatsTensorPtr == nullptr || ksTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto meansShape = context->GetInputShape(MEANS_PTR_INDEX);
    auto viewmatsShape = context->GetInputShape(VIEWMATS_PTR_INDEX);
    auto ksShape = context->GetInputShape(KS_PTR_INDEX);
    // covarsShape unused (output shapes come from means/viewmats); covars
    // is optional and may be null on the fused path.
    if (meansShape == nullptr || viewmatsShape == nullptr || ksShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int32_t imageWidth = *(attrsPtr->GetAttrPointer<int32_t>(IMAGE_WIDTH_INDEX));
    int32_t imageHeight = *(attrsPtr->GetAttrPointer<int32_t>(IMAGE_HEIGHT_INDEX));
    float eps2d = *(attrsPtr->GetAttrPointer<float>(EPS2D_INDEX));
    bool calcCompensations = *(attrsPtr->GetAttrPointer<bool>(CALC_COMPENSATIONS_INDEX));
    uint32_t cameraModel = static_cast<uint32_t>(*(attrsPtr->GetAttrPointer<int32_t>(CAMERA_MODEL_INDEX)));

    // optional quat/scales; either covars or (quat AND scales) required.
    auto quatTensorPtr = context->GetInputTensor(QUAT_PTR_INDEX);
    auto scalesTensorPtr = context->GetInputTensor(SCALES_PTR_INDEX);
    auto covarsTensorPtr = context->GetInputTensor(COVARS_PTR_INDEX);
    uint32_t inputIsQuatScales = (quatTensorPtr != nullptr && scalesTensorPtr != nullptr) ? 1u : 0u;
    if (inputIsQuatScales == 0 && covarsTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

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

    // Subtract the fixed CameraParam slot, divide by the per-task byte cost,
    // then align DOWN (align-up could push the tile past the UB_RATIO budget).
    // The quat+scales path uses PER_TASK_FLOAT_BUFFERS_QS.
    uint32_t perTaskBuffers = (inputIsQuatScales == 1)
        ? PER_TASK_FLOAT_BUFFERS_QS : PER_TASK_FLOAT_BUFFERS;
    uint32_t taskNumPerLoop = static_cast<int32_t>(
        (ubTotalSize * UB_RATIO - FIXED_UB_BYTES) / (perTaskBuffers * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>(taskNumPerLoop / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_imageWidth(static_cast<float>(imageWidth));
    tiling.set_imageHeight(static_cast<float>(imageHeight));
    tiling.set_cameraModel(cameraModel);
    tiling.set_eps2d(eps2d);
    tiling.set_calcCompensations(calcCompensations);
    tiling.set_inputIsQuatScales(inputIsQuatScales);

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
    // cameraNum == 1 -> key 2 (single-camera, camera-major);
    // cameraNum > 1 -> key 1 (multi-camera, taskLoop-major).
    context->SetTilingKey(cameraNum == 1 ? 2 : 1);
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
    const gert::Shape* viewmatsShape = context->GetInputShape(VIEWMATS_PTR_INDEX);
    const gert::Shape* ksShape = context->GetInputShape(KS_PTR_INDEX);

    // covars optional (replaced by quat/scales on the fused path);
    // output shapes derive from means + viewmats only.
    if (meansShape == nullptr || viewmatsShape == nullptr || ksShape == nullptr) {
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
        // optional quat/scales -- alternative to covars; the kernel inlines
        // quat_scales_to_covars when both are provided.
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
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(ProjectionThreeDimsGaussianForward);
}
