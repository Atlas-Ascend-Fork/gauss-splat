/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gaussian_filter_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace {
static const int64_t MEANS_PTR_INDEX = 0;
static const int64_t COLORS_PTR_INDEX = 1;
static const int64_t DET_PTR_INDEX = 2;
static const int64_t OPACITIES_PTR_INDEX = 3;
static const int64_t MEANS2D_PTR_INDEX = 4;
static const int64_t DEPTHS_PTR_INDEX = 5;
static const int64_t RADIUS_PTR_INDEX = 6;
static const int64_t CONICS_PTR_INDEX = 7;
static const int64_t COVARS2D_PTR_INDEX = 8;
static const int64_t COMPENSATIONS_PTR_INDEX = 9;

static const int64_t MEANS_CULLING_PTR_INDEX = 0;
static const int64_t COLORS_CULLING_PTR_INDEX = 1;
static const int64_t MEANS2D_CULLING_PTR_INDEX = 2;
static const int64_t DEPTHS_CULLING_PTR_INDEX = 3;
static const int64_t RADIUS_CULLING_PTR_INDEX = 4;
static const int64_t COVARS2D_CULLING_PTR_INDEX = 5;
static const int64_t CONICS_CULLING_PTR_INDEX = 6;
static const int64_t OPACITIES_CULLING_PTR_INDEX = 7;
static const int64_t QUATS_CULLING_PTR_INDEX = 8;
static const int64_t SCALES_CULLING_PTR_INDEX = 9;
static const int64_t VI_CULLING_PTR_INDEX = 10;
static const int64_t COMPENSATIONS_CULLING_PTR_INDEX = 11;
static const int64_t FILTER_PTR_INDEX = 12;
static const int64_t CNT_PTR_INDEX = 13;
static const int64_t FILTER_FLOAT_PTR_INDEX = 14;
static const int64_t DENSE_INDICES_PTR_INDEX = 15;

static const int64_t WIDTH_INDEX = 0;
static const int64_t HEIGHT_INDEX = 1;
static const int64_t NEAR_PLANE_INDEX = 2;
static const int64_t FAR_PLANE_INDEX = 3;
static const int64_t SUPPORT_FEATURES_INDEX = 4;

static const int64_t DIM_0 = 0;
static const int64_t DIM_1 = 1;
static const int64_t DIM_2 = 2;

static const int32_t SIGMA_LEN = 1;
static const int64_t PACKED_ALIGN = 8;
static const int64_t PACKED_PAD = 7;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // set block size
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t cubeCoreNum = ascendcPlatform.GetCoreNumAic();
    if (cubeCoreNum == 0) return ge::GRAPH_FAILED;
    context->SetBlockDim(cubeCoreNum < 32 ? cubeCoreNum : 32);
    context->SetDynUBufSize(221184);

    const gert::StorageShape *detShape = context->GetInputShape(DET_PTR_INDEX);
    const gert::StorageShape *colorsShape = context->GetInputShape(COLORS_PTR_INDEX);
    if (detShape == nullptr || colorsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    GaussianFilterTilingData tiling;
    auto attrs = context->GetAttrs();
    int32_t batchNum = detShape->GetStorageShape().GetDim(DIM_0);
    int32_t cameraNum = detShape->GetStorageShape().GetDim(DIM_1);
    int32_t gaussNum = detShape->GetStorageShape().GetDim(DIM_2);
    int32_t colorDim = colorsShape->GetStorageShape().GetDim(DIM_1);
    tiling.set_batchNum(batchNum);
    tiling.set_cameraNum(cameraNum);
    tiling.set_gaussNum(gaussNum);
    tiling.set_colorDims(colorDim);
    tiling.set_sigmaLen(SIGMA_LEN);
    tiling.set_M((gaussNum + PACKED_PAD) / PACKED_ALIGN);

    float width = static_cast<float>(*(attrs->GetAttrPointer<int32_t>(WIDTH_INDEX)));
    float height = static_cast<float>(*(attrs->GetAttrPointer<int32_t>(HEIGHT_INDEX)));
    float near_plane = *(attrs->GetAttrPointer<float>(NEAR_PLANE_INDEX));
    float far_plane = *(attrs->GetAttrPointer<float>(FAR_PLANE_INDEX));
    int32_t support_features = *(attrs->GetAttrPointer<int32_t>(SUPPORT_FEATURES_INDEX));
    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_nearPlane(near_plane);
    tiling.set_farPlane(far_plane);
    tiling.set_supportFeatures(support_features);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t userWorkspaceSize = 64 * sizeof(float);
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
    context->SetOutputDataType(MEANS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(COLORS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(MEANS2D_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DEPTHS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(RADIUS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(COVARS2D_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(CONICS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OPACITIES_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(QUATS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(SCALES_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(VI_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(COMPENSATIONS_CULLING_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(FILTER_PTR_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(CNT_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(FILTER_FLOAT_PTR_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DENSE_INDICES_PTR_INDEX, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class GaussianFilter : public OpDef {
public:
    explicit GaussianFilter(const char* name) : OpDef(name)
    {
        this->Input("means")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("colors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("det")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("opacities")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("means2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("radius")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conics")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("covars2d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("compensations")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("quats")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scales")
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
        this->Output("means_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("colors_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("means2d_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("depths_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("radius_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("covars2d_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("conics_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("opacities_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("quats_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("scales_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("vi_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("compensations_culling")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("filter_float")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("dense_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("near_plane").AttrType(REQUIRED).Float();
        this->Attr("far_plane").AttrType(REQUIRED).Float();
        this->Attr("support_features").AttrType(REQUIRED).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(GaussianFilter);
}
