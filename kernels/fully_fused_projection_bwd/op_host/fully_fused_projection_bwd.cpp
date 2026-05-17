/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fully_fused_projection_bwd_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const int32_t RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;
const int64_t RESERVED_BUFFER = 1024;
const int64_t FLOAT_SIZE = 4;
const int64_t BUFFER_SIZE = 100;
const int64_t ONE_BLOCK_FLOAT = 8;
const int64_t MEANS_PTR_INDEX = 0;
const int64_t CONICS_PTR_INDEX = 3;
const int64_t COMPENSATIONS_PTR_INDEX = 12;

const int64_t BATCH_SIZE_INDEX = 0;
const int64_t CAMERA_NUM_INDEX = 1;
const int64_t GAUSSIAN_NUM_INDEX = 2;

const int64_t UB_MAX_GAUSSIAN = 480;
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t maxCoreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto compensationsDesc = context->GetOptionalInputDesc(COMPENSATIONS_PTR_INDEX);

    const gert::StorageShape *meansShape = context->GetInputShape(MEANS_PTR_INDEX);
    const gert::StorageShape *conicsShape = context->GetInputShape(CONICS_PTR_INDEX);
    FullyFusedProjectionBwdTilingData tiling;
    int64_t hasCompensations = 0;
    if (compensationsDesc != nullptr) {
        hasCompensations = 1;
    }
    tiling.set_hasCompensations(hasCompensations);

    int64_t batchNum = meansShape->GetStorageShape().GetDim(BATCH_SIZE_INDEX);
    int64_t cameraNum = conicsShape->GetStorageShape().GetDim(CAMERA_NUM_INDEX);
    int64_t gaussNum = meansShape->GetStorageShape().GetDim(GAUSSIAN_NUM_INDEX);

    tiling.set_batchNum(batchNum);
    tiling.set_cameraNum(cameraNum);
    tiling.set_gaussNum(gaussNum);

    int64_t blockLength = 0;
    if (maxCoreNum != 0) {
        blockLength = ((gaussNum + maxCoreNum - 1) / maxCoreNum + ONE_BLOCK_FLOAT - 1) /
                        ONE_BLOCK_FLOAT * ONE_BLOCK_FLOAT;
    } else {
        return ge::GRAPH_FAILED;
    }
    if (blockLength != 0) {
        maxCoreNum = (gaussNum + blockLength - 1) / blockLength;
    } else {
        return ge::GRAPH_FAILED;
    }

    int64_t lastblockLength = gaussNum - blockLength * (maxCoreNum - 1);
    tiling.set_blockLength(blockLength);
    tiling.set_lastcoreNum(lastblockLength);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t width = *(attrs->GetInt(0));
    int64_t height = *(attrs->GetInt(1));
    tiling.set_width(width);
    tiling.set_height(height);

    int64_t ubSize_ = static_cast<int64_t>(ubSize) - RESERVED_BUFFER;
    int64_t perloopNum = ubSize_ / (BUFFER_SIZE * FLOAT_SIZE) / ONE_BLOCK_FLOAT * ONE_BLOCK_FLOAT;
    tiling.set_perloopNum(perloopNum);
    tiling.set_needCoreNum(maxCoreNum);
    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize + RESERVED_WORKSPACE_SIZE;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    context->SetBlockDim(maxCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) { return GRAPH_SUCCESS; }

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) { return ge::GRAPH_SUCCESS; }
} // namespace ge

namespace ops {
class FullyFusedProjectionBwd : public OpDef {
public:
    explicit FullyFusedProjectionBwd(const char *name) : OpDef(name)
    {
        this->Input("means")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conics")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("viewmats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("Ks")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_means2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_depths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_conics")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_colors_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_opacities_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compesations")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Output("v_pW")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("v_quats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("v_scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("v_R")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("v_colors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("v_opacities")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FullyFusedProjectionBwd);
} // namespace ops
