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

#include "register/op_def_registry.h"
#include "spherical_harmonics_bwd_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
static constexpr int32_t RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;
static const int64_t RESERVED_BUFFER = 1024;
static const int64_t DIRS_PTR_INDEX = 0;
static const int64_t COEFFS_PTR_INDEX = 1;
static const int64_t VCOLORS_PTR_INDEX = 2;
static const int64_t BUFFER_LEN_D0 = 5;
static const int64_t BUFFER_LEN_D1 = 5;
static const int64_t BUFFER_LEN_D2 = 13;
static const int64_t BUFFER_LEN_D3 = 21;
static const int64_t BUFFER_LEN_D4 = 27;
static const int64_t PERLOOPNUM_D0 = 1520;
static const int64_t PERLOOPNUM_D1 = 968;
static const int64_t PERLOOPNUM_D2 = 528;
static const int64_t PERLOOPNUM_D3 = 352;
static const int64_t PERLOOPNUM_D4 = 240;
static const int64_t DIRS_DIM = 3;
static const int64_t COEFFS_DIM = 4;
static const int64_t VCOLORS_DIM = 3;
static const int64_t DEGREE_ZERO = 0;
static const int64_t DEGREE_ONE = 1;
static const int64_t DEGREE_TWO = 2;
static const int64_t DEGREE_THREE = 3;
} // namespace

namespace optiling {
static ge::graphStatus Tiling4SphericalHarmonicsBwd(gert::TilingContext *context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t maxCoreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const gert::StorageShape *dirsShape = context->GetInputShape(DIRS_PTR_INDEX);
    const gert::StorageShape *coeffsShape = context->GetInputShape(COEFFS_PTR_INDEX);
    const gert::StorageShape *vColorsShape = context->GetInputShape(VCOLORS_PTR_INDEX);
    if (dirsShape == nullptr || coeffsShape == nullptr || vColorsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (dirsShape->GetStorageShape().GetDimNum() != DIRS_DIM ||
        coeffsShape->GetStorageShape().GetDimNum() != COEFFS_DIM ||
        vColorsShape->GetStorageShape().GetDimNum() != VCOLORS_DIM) {
        return ge::GRAPH_FAILED;
    }

    SphericalHarmonicsBwdTilingData tiling;
    int64_t batchNum = dirsShape->GetStorageShape().GetDim(0);
    int64_t gaussNum = dirsShape->GetStorageShape().GetDim(2);
    int64_t K = coeffsShape->GetStorageShape().GetDim(1);

    tiling.set_batchNum(batchNum);
    tiling.set_gaussNum(gaussNum);

    int64_t blockLength = 0;
    if (maxCoreNum != 0) {
        blockLength = (gaussNum + maxCoreNum - 1) / maxCoreNum;
    } else {
        return ge::GRAPH_FAILED;
    }

    int64_t lastblockLength = gaussNum - blockLength * (maxCoreNum - 1);
    tiling.set_blockLength(blockLength);
    tiling.set_lastcoreNum(lastblockLength);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t degree = *(attrs->GetInt(0));
    if (K != (degree + 1) * (degree + 1)) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_degree(degree);

    int64_t ubSize_ = static_cast<int64_t>(ubSize) - RESERVED_BUFFER;
    int64_t perloopNum = 240;
    tiling.set_needCoreNum(maxCoreNum);
    tiling.set_K(K);
    int64_t bufferLen = 0;
    if (degree == DEGREE_ZERO) {
        bufferLen = BUFFER_LEN_D0;
        perloopNum = PERLOOPNUM_D0;
    } else if (degree == DEGREE_ONE) {
        bufferLen = BUFFER_LEN_D1;
        perloopNum = PERLOOPNUM_D1;
    } else if (degree == DEGREE_TWO) {
        bufferLen = BUFFER_LEN_D2;
        perloopNum = PERLOOPNUM_D2;
    } else if (degree == DEGREE_THREE) {
        bufferLen = BUFFER_LEN_D3;
        perloopNum = PERLOOPNUM_D3;
    } else {
        bufferLen = BUFFER_LEN_D4;
        perloopNum = PERLOOPNUM_D4;
    }
    tiling.set_perloopNum(perloopNum);
    tiling.set_bufferLen(bufferLen);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(RESERVED_WORKSPACE_SIZE);
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
class SphericalHarmonicsBwd : public OpDef {
public:
    explicit SphericalHarmonicsBwd(const char *name) : OpDef(name)
    {
        this->Input("dirs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("coeffs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v_colors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v_dirs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v_coeffs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("degree").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::Tiling4SphericalHarmonicsBwd);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SphericalHarmonicsBwd);
} // namespace ops
