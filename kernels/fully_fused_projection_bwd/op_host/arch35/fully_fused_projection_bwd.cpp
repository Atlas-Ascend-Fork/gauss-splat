/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fully_fused_projection_bwd_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace {

static const int64_t MEANS_PTR_INDEX = 0;
static const int64_t FILTER_PTR_INDEX = 11;

static const int64_t WIDTH_INDEX = 0;
static const int64_t HEIGHT_INDEX = 1;
static const int64_t MAX_COMPACT_COUNT_INDEX = 2;
static const int64_t SUPPORT_FEATURES_INDEX = 3;

static const int64_t V_PW_PTR_INDEX = 0;
static const int64_t V_QUATS_PTR_INDEX = 1;
static const int64_t V_SCALES_PTR_INDEX = 2;
static const int64_t V_R_PTR_INDEX = 3;
static const int64_t V_COLORS_PTR_INDEX = 4;
static const int64_t V_OPACITIES_PTR_INDEX = 5;
static const int64_t V_VIEWMATS_PTR_INDEX = 6;
static const int64_t V_KS_PTR_INDEX = 7;
static const int64_t DL_DVI_PTR_INDEX = 8;
static const int64_t DL_DSIGMA_PTR_INDEX = 9;

static const int64_t DIM_0 = 0;
static const int64_t DIM_1 = 1;
static const int64_t DIM_2 = 2;

static const uint32_t SF_MAX_ALPHA_GRAD_ABS_OUTPUT = (1 << 3);
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // set block size
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t vecCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (vecCoreNum == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(vecCoreNum);
    context->SetDynUBufSize(221184);

    FullyFusedProjectionBwdTilingData tiling;
    auto attrs = context->GetAttrs();
    // always const attributes.
    tiling.set_scalar_one(1);
    tiling.set_feature2(2);
    tiling.set_feature3(3);
    tiling.set_feature4(4);
    tiling.set_view_cols(16);
    tiling.set_k_cols(9);

    // dimension relate attributes
    const gert::StorageShape *meansShape = context->GetInputShape(MEANS_PTR_INDEX);
    const gert::StorageShape *filterShape = context->GetInputShape(FILTER_PTR_INDEX);
    if (meansShape == nullptr || filterShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int32_t slice_count = meansShape->GetStorageShape().GetDim(DIM_0);
    int32_t gaussian_num = meansShape->GetStorageShape().GetDim(DIM_2);
    int32_t batch_count = filterShape->GetStorageShape().GetDim(DIM_0);
    int32_t camera_count = filterShape->GetStorageShape().GetDim(DIM_1);
    tiling.set_slice_count(slice_count);
    tiling.set_dense_g(gaussian_num);
    tiling.set_batch_count(batch_count);
    tiling.set_camera_count(camera_count);

    // new attributes
    int32_t width = *(attrs->GetAttrPointer<int32_t>(WIDTH_INDEX));
    int32_t height = *(attrs->GetAttrPointer<int32_t>(HEIGHT_INDEX));
    int32_t max_compact_count = *(attrs->GetAttrPointer<int32_t>(MAX_COMPACT_COUNT_INDEX));
    int32_t chunk_stride = std::max(1, (max_compact_count + 63) / 64);
    int32_t support_features = *(attrs->GetAttrPointer<int32_t>(SUPPORT_FEATURES_INDEX));
    bool max_alpha_abs_grad_output = ((support_features & SF_MAX_ALPHA_GRAD_ABS_OUTPUT) != 0);
    tiling.set_image_width(width);
    tiling.set_image_height(height);
    tiling.set_chunk_stride(chunk_stride);
    tiling.set_total_tasks(slice_count * chunk_stride);
    tiling.set_max_compact_count(max_compact_count);
    tiling.set_color_dims(max_alpha_abs_grad_output ? 6 : 3);
    tiling.set_support_features(support_features);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = 0;
    size_t sysWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(V_PW_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_QUATS_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_SCALES_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_R_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_COLORS_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_OPACITIES_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_VIEWMATS_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(V_KS_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DL_DVI_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DL_DSIGMA_PTR_INDEX, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FullyFusedProjectionBwd : public OpDef {
public:
    explicit FullyFusedProjectionBwd(const char* name) : OpDef(name)
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
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compesations")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_vi_culling")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("opacities")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("vi")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sigma")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depths")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("dense_indices")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cnt")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("max_compact_count").AttrType(REQUIRED).Int(1);
        this->Attr("support_features").AttrType(REQUIRED).Int(0);
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
        this->Output("v_viewmats")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("v_ks")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dL_dvi")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("dL_dsigma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(FullyFusedProjectionBwd);
}
