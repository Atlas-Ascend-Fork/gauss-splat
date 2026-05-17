/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gaussian_sort.cpp
 * \brief gaussian sort op host
 */

#include "gaussian_sort_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t LBSCHED_PTR_INDEX = 0;
constexpr uint32_t CNT_PTR_INDEX = 1;
constexpr uint32_t DEPTHS_PTR_INDEX = 2;
constexpr uint32_t GSIDS_PTR_INDEX = 3;
constexpr uint32_t SORTED_OFFSETS_PTR_INDEX = 4;
constexpr uint32_t BATCH_SIZE_INDEX = 0;
constexpr uint32_t CAMERA_NUM_INDEX = 1;
constexpr uint32_t TILE_NUM_INDEX = 2;
constexpr uint32_t GAUSS_NUM_INDEX = 3;
constexpr uint32_t SCHEDULE_NUM_INDEX = 2;
constexpr uint32_t TILE_MAX_GAUSSIAN_INDEX = 0;
constexpr uint32_t SIZE_OF_FLOAT = 4;
constexpr uint32_t ALIGN_NUM = 32;
constexpr uint32_t SORT_TENSOR_NUM = 8;
constexpr uint32_t WS_TENSOR_NUM = 2;
}  // namespace

namespace optiling {
static ge::graphStatus TilingForGaussianSort(gert::TilingContext* context)
{
    GaussianSortTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto lbSchedTensorPtr = context->GetInputTensor(LBSCHED_PTR_INDEX);
    auto cntTensorPtr = context->GetInputTensor(CNT_PTR_INDEX);
    auto depthsTensorPtr = context->GetInputTensor(DEPTHS_PTR_INDEX);
    auto gsIdsTensorPtr = context->GetInputTensor(GSIDS_PTR_INDEX);
    auto sortedOffsetTensorPtr = context->GetInputTensor(SORTED_OFFSETS_PTR_INDEX);
    if (lbSchedTensorPtr == nullptr || cntTensorPtr == nullptr || depthsTensorPtr == nullptr ||
        gsIdsTensorPtr == nullptr || sortedOffsetTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto lbSchedShape = context->GetInputShape(LBSCHED_PTR_INDEX);
    auto cntShape = context->GetInputShape(CNT_PTR_INDEX);
    auto depthsShape = context->GetInputShape(DEPTHS_PTR_INDEX);
    auto gsIdsShape = context->GetInputShape(GSIDS_PTR_INDEX);
    auto sortedOffsetShape = context->GetInputShape(SORTED_OFFSETS_PTR_INDEX);
    if (lbSchedShape == nullptr || cntShape == nullptr || depthsShape == nullptr || gsIdsShape == nullptr ||
        sortedOffsetShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = depthsShape->GetStorageShape().GetDim(BATCH_SIZE_INDEX);
    uint32_t cameraNum = depthsShape->GetStorageShape().GetDim(CAMERA_NUM_INDEX);
    uint32_t tileNum = depthsShape->GetStorageShape().GetDim(TILE_NUM_INDEX);
    uint32_t gaussNum = depthsShape->GetStorageShape().GetDim(GAUSS_NUM_INDEX);
    uint32_t scheduleNum = lbSchedShape->GetStorageShape().GetDim(SCHEDULE_NUM_INDEX);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t tileMaxGaussian = *(attrsPtr->GetAttrPointer<uint32_t>(TILE_MAX_GAUSSIAN_INDEX));
    uint32_t maxMaskNum = (tileMaxGaussian % ALIGN_NUM == 0)
                              ? tileMaxGaussian
                              : ((tileMaxGaussian + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint32_t vectorNum = platformInfo.GetCoreNumAiv();
    if (vectorNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockDim = (tileNum > vectorNum) ? vectorNum : tileNum;
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // sort阶段，核内大小数排序UB支持最大数计算
    uint32_t maxSortNum = ubSize / (SORT_TENSOR_NUM * SIZE_OF_FLOAT);

    tiling.set_batchSize(batchSize);
    tiling.set_cameraNum(cameraNum);
    tiling.set_tileNum(tileNum);
    tiling.set_scheduleNum(scheduleNum);
    tiling.set_gaussNum(gaussNum);
    tiling.set_maxSortNum(maxSortNum);
    tiling.set_maxMaskNum(maxMaskNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);

    // workspace 空间申请，引入 B，C 维度，外加负载均衡策略，无法精细化处理，故选择全局tile中最大高斯球数作为空间申请
    size_t userWorkspaceSize = maxMaskNum * SIZE_OF_FLOAT * WS_TENSOR_NUM * blockDim;
    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = systemWorkspaceSize + userWorkspaceSize;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class GaussianSort : public OpDef {
public:
    explicit GaussianSort(const char* name) : OpDef(name)
    {
        this->Input("lb_sched")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gaussian_cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gs_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sorted_offset")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sorted_gs_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("max_tile_gauss").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingForGaussianSort);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GaussianSort);
}  // namespace ops
