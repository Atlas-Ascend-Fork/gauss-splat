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

#include "gaussian_filter_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
static constexpr int32_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const int64_t RESERVED_BUFFER = 1024;
static const int64_t ONE_BLK_FLOAT = 8;
static const int64_t SIZE_OF_FILTER = 8;
static const int64_t BLOCK_BYTES = 32;
static const int64_t INT_BYTES = 4;
static const int64_t FLOAT_BYTES = 4;
static const int64_t BUFFER_LEN = 24;
static const int64_t MAX_CORE_NUM = 48;

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
static const int64_t FILTER_PTR_INDEX = 8;
static const int64_t CNT_PTR_INDEX = 9;

static const int64_t WIDTH_INDEX = 0;
static const int64_t HEIGHT_INDEX = 1;
static const int64_t NEAR_PLANE_INDEX = 2;
static const int64_t FAR_PLANE_INDEX = 3;

static const int64_t DIM_0 = 0;
static const int64_t DIM_1 = 1;
static const int64_t DIM_2 = 2;
} // namespace

namespace optiling {
static ge::graphStatus Tiling4GaussianFilter(gert::TilingContext *context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t maxCoreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto compensationsDesc = context->GetOptionalInputDesc(COMPENSATIONS_PTR_INDEX);
    const gert::StorageShape *meansShape = context->GetInputShape(MEANS_PTR_INDEX);
    const gert::StorageShape *colorsShape = context->GetInputShape(COLORS_PTR_INDEX);
    const gert::StorageShape *detShape = context->GetInputShape(DET_PTR_INDEX);
    const gert::StorageShape *opacitiesShape = context->GetInputShape(OPACITIES_PTR_INDEX);
    const gert::StorageShape *means2dShape = context->GetInputShape(MEANS2D_PTR_INDEX);
    const gert::StorageShape *depthsShape = context->GetInputShape(DEPTHS_PTR_INDEX);
    const gert::StorageShape *radiusShape = context->GetInputShape(RADIUS_PTR_INDEX);
    const gert::StorageShape *conicsShape = context->GetInputShape(CONICS_PTR_INDEX);
    const gert::StorageShape *covars2dShape = context->GetInputShape(COVARS2D_PTR_INDEX);

    const gert::StorageShape *meansCullingShape = context->GetOutputShape(MEANS_CULLING_PTR_INDEX);
    const gert::StorageShape *colorsCullingShape = context->GetOutputShape(COLORS_CULLING_PTR_INDEX);
    const gert::StorageShape *means2dCullingShape = context->GetOutputShape(MEANS2D_CULLING_PTR_INDEX);
    const gert::StorageShape *depthsCullingShape = context->GetOutputShape(DEPTHS_CULLING_PTR_INDEX);
    const gert::StorageShape *radiusCullingShape = context->GetOutputShape(RADIUS_CULLING_PTR_INDEX);
    const gert::StorageShape *covars2dCullingShape = context->GetOutputShape(COVARS2D_CULLING_PTR_INDEX);
    const gert::StorageShape *conicsCullingShape = context->GetOutputShape(CONICS_CULLING_PTR_INDEX);
    const gert::StorageShape *opacitiesCullingShape = context->GetOutputShape(OPACITIES_CULLING_PTR_INDEX);
    const gert::StorageShape *filterShape = context->GetOutputShape(FILTER_PTR_INDEX);
    const gert::StorageShape *cntShape = context->GetOutputShape(CNT_PTR_INDEX);

    if (meansShape == nullptr || colorsShape == nullptr || detShape == nullptr || opacitiesShape == nullptr ||
        means2dShape == nullptr || depthsShape == nullptr || radiusShape == nullptr || conicsShape == nullptr ||
        covars2dShape == nullptr || means2dCullingShape == nullptr || depthsCullingShape == nullptr ||
        radiusCullingShape == nullptr || covars2dCullingShape == nullptr || conicsCullingShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t hasCompensations = 0;

    if (compensationsDesc != nullptr) {
        hasCompensations = 1;
    }

    GaussianFilterTilingData tiling;
    tiling.set_hasCompensations(hasCompensations);
    int64_t batchNum = detShape->GetStorageShape().GetDim(DIM_0);
    int64_t cameraNum = detShape->GetStorageShape().GetDim(DIM_1);
    int64_t gaussNum = detShape->GetStorageShape().GetDim(DIM_2);

    tiling.set_batchNum(batchNum);
    tiling.set_cameraNum(cameraNum);
    tiling.set_gaussNum(gaussNum);
    if (maxCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t blockLength =
        ((gaussNum + maxCoreNum - 1) / maxCoreNum / SIZE_OF_FILTER + SIZE_OF_FILTER - 1) * SIZE_OF_FILTER;
    maxCoreNum = (gaussNum + blockLength - 1) / blockLength;
    int64_t lastBlockLength = gaussNum - blockLength * (maxCoreNum - 1);
    tiling.set_blockLength(blockLength);
    tiling.set_lastcoreNum(lastBlockLength);

    auto attrs = context->GetAttrs();
    int64_t width = *(attrs->GetAttrPointer<int64_t>(WIDTH_INDEX));
    int64_t height = *(attrs->GetAttrPointer<int64_t>(HEIGHT_INDEX));
    float nearPlane = *(attrs->GetAttrPointer<float>(NEAR_PLANE_INDEX));
    float farPlane = *(attrs->GetAttrPointer<float>(FAR_PLANE_INDEX));

    tiling.set_width(width);
    tiling.set_height(height);
    tiling.set_nearPlane(nearPlane);
    tiling.set_farPlane(farPlane);

    int64_t ubSize_ = (static_cast<int64_t>(ubSize) - RESERVED_BUFFER);
    tiling.set_needCoreNum(maxCoreNum);

    int64_t cntBuffer = MAX_CORE_NUM * INT_BYTES / BLOCK_BYTES * BLOCK_BYTES * 2;
    int64_t perloopNum =
        ((ubSize_ - cntBuffer) / BUFFER_LEN - ONE_BLK_FLOAT) / FLOAT_BYTES / SIZE_OF_FILTER * SIZE_OF_FILTER;

    tiling.set_perloopNum(perloopNum);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    size_t sysWorkspaceByteSize = static_cast<size_t>(platformInfo.GetLibApiWorkSpaceSize());
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceByteSize + RESERVED_WORKSPACE_SIZE);
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
class GaussianFilter : public OpDef {
public:
    explicit GaussianFilter(const char *name) : OpDef(name)
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
        this->Output("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("cnt")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("width").AttrType(REQUIRED).Int();
        this->Attr("height").AttrType(REQUIRED).Int();
        this->Attr("near_plane").AttrType(REQUIRED).Float();
        this->Attr("far_plane").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::Tiling4GaussianFilter);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GaussianFilter);
} // namespace ops
