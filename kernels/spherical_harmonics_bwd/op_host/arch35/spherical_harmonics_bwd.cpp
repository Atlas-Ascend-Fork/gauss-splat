/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include "spherical_harmonics_bwd_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
static constexpr int32_t RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;
// Keep room for the Register Base VF stack and runtime-reserved UB space.
static const int64_t RESERVED_UB_SIZE = 40 * 1024;
static const int64_t DIRS_PTR_INDEX = 0;
static const int64_t COEFFS_PTR_INDEX = 1;
static const int64_t VCOLORS_PTR_INDEX = 2;
static const int64_t VDIRS_PTR_INDEX = 0;
static const int64_t VCOEFFS_PTR_INDEX = 1;
static const int64_t BUFFER_LEN_D0 = 5;
static const int64_t BUFFER_LEN_D1 = 5;
static const int64_t BUFFER_LEN_D2 = 13;
static const int64_t BUFFER_LEN_D3 = 21;
static const int64_t BUFFER_LEN_D4 = 27;
static const int64_t FLOAT_SIZE = 4;
static const int64_t FLOATS_PER_VL = 64;
static const int64_t MAX_CHANNEL_BUFFER_NUM = 2;
// dirs: 3 rows; accumulated v_dirs plus norm: 4 rows.
static const int64_t SHARED_BUFFER_ROWS = 7;
static const int64_t DIRS_DIM = 3;
static const int64_t COEFFS_DIM = 4;
static const int64_t VCOLORS_DIM = 3;
static const int64_t DEGREE_ZERO = 0;
static const int64_t DEGREE_ONE = 1;
static const int64_t DEGREE_TWO = 2;
static const int64_t DEGREE_THREE = 3;
static const int64_t DEGREE_FOUR = 4;
static const int64_t MIN_CHANNEL_NUM = 1;
static const int64_t MAX_CHANNEL_NUM = 4;
// Degree-0 small-workload grid: derive needCoreNum from a per-core minimum work
// quantum so the per-launch/per-core preamble is not paid on all maxCoreNum
// cores. Calibrated by a bounded grid sweep (round2 grid-probe); gaussNum >=
// GRID_MAX_N keeps the round-2 grid exactly.
static const int64_t MIN_PER_CORE = 1024;
static const int64_t GRID_MAX_N = 57344;
// Degree-0 channel-batched DMA predicate: activates only in the low-MBU region
// (gaussNum <= BATCH_MAX_N) so the medium/large cases keep the per-channel path.
static const int64_t BATCH_CHANNEL_NUM = 3;
static const int64_t BATCH_MAX_N = 131072;
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
    int64_t channelNum = coeffsShape->GetStorageShape().GetDim(2);

    if (channelNum < MIN_CHANNEL_NUM || channelNum > MAX_CHANNEL_NUM ||
        dirsShape->GetStorageShape().GetDim(1) != DIRS_DIM ||
        coeffsShape->GetStorageShape().GetDim(0) != batchNum ||
        coeffsShape->GetStorageShape().GetDim(3) != gaussNum ||
        vColorsShape->GetStorageShape().GetDim(0) != batchNum ||
        vColorsShape->GetStorageShape().GetDim(1) != channelNum ||
        vColorsShape->GetStorageShape().GetDim(2) != gaussNum) {
            return ge::GRAPH_FAILED;
        }

    tiling.set_batchNum(batchNum);
    tiling.set_gaussNum(gaussNum);
    tiling.set_channelNum(channelNum);

    if (maxCoreNum <= 0 || gaussNum <= 0 || batchNum <= 0) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrPointer<int32_t>(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t degree = *(attrs->GetAttrPointer<int32_t>(0));
    if (degree < DEGREE_ZERO || degree > DEGREE_FOUR || K != (degree + 1) * (degree + 1)) {
        return ge::GRAPH_FAILED;
    }

    // Degree-0 work quantum: for small gaussNum dispatch needCoreNum derived from
    // MIN_PER_CORE instead of all maxCoreNum cores. blockLength/lastcoreNum and the
    // kernel perCoreN_/curCoreN_ mapping keep their exact existing semantics.
    int64_t needCoreNum = gaussNum < maxCoreNum ? gaussNum : maxCoreNum;
    if (degree == DEGREE_ZERO && gaussNum <= GRID_MAX_N) {
        int64_t gridFromWork = (gaussNum + MIN_PER_CORE - 1) / MIN_PER_CORE;
        if (gridFromWork < 1) {
            gridFromWork = 1;
        }
        if (gridFromWork > maxCoreNum) {
            gridFromWork = maxCoreNum;
        }
        needCoreNum = gridFromWork;
    }
    int64_t blockLength = gaussNum / needCoreNum;

    int64_t lastblockLength = gaussNum - blockLength * (needCoreNum - 1);
    tiling.set_blockLength(blockLength);
    tiling.set_lastcoreNum(lastblockLength);
    tiling.set_degree(degree);

    // C > 1 uses two channel buffers: coeffs input (2*K), v_colors
    // input (2), and v_coeffs output (2*K). C == 1 has no adjacent channel
    // to overlap, so one buffer is sufficient and the tile can be larger.
    int64_t usableUbSize = static_cast<int64_t>(ubSize) - RESERVED_UB_SIZE;
    int64_t channelBufferNum = channelNum > 1 ? MAX_CHANNEL_BUFFER_NUM : 1;
    int64_t bufferRows = SHARED_BUFFER_ROWS + channelBufferNum * (2 * K + 1);
    if (degree == DEGREE_ZERO && channelNum == BATCH_CHANNEL_NUM && gaussNum <= BATCH_MAX_N) {
        // Degree-0 batched footprint: shared dirs queue + v_colors + v_coeffs + v_dirs
        // rows (one buffer each), not SHARED_BUFFER_ROWS + channelBufferNum*(2*K+1).
        bufferRows = DIRS_DIM + VCOLORS_DIM + K * channelNum + DIRS_DIM;
    }
    if (usableUbSize <= 0 || bufferRows <= 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t perloopNum = usableUbSize / (bufferRows * FLOAT_SIZE);
    perloopNum = perloopNum / FLOATS_PER_VL * FLOATS_PER_VL;
    if (perloopNum < FLOATS_PER_VL) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_needCoreNum(needCoreNum);
    tiling.set_K(K);
    int64_t bufferLen = 0;
    if (degree == DEGREE_ZERO) {
        bufferLen = BUFFER_LEN_D0;
    } else if (degree == DEGREE_ONE) {
        bufferLen = BUFFER_LEN_D1;
    } else if (degree == DEGREE_TWO) {
        bufferLen = BUFFER_LEN_D2;
    } else if (degree == DEGREE_THREE) {
        bufferLen = BUFFER_LEN_D3;
    } else {
        bufferLen = BUFFER_LEN_D4;
    }
    tiling.set_perloopNum(perloopNum);
    tiling.set_bufferLen(bufferLen);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(RESERVED_WORKSPACE_SIZE);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    context->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *dirsShape = context->GetInputShape(DIRS_PTR_INDEX);
    const gert::Shape *coeffsShape = context->GetInputShape(COEFFS_PTR_INDEX);
    gert::Shape *vDirsShape = context->GetOutputShape(VDIRS_PTR_INDEX);
    gert::Shape *vCoeffsShape = context->GetOutputShape(VCOEFFS_PTR_INDEX);
    if (dirsShape == nullptr || coeffsShape == nullptr || vDirsShape == nullptr || vCoeffsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *vDirsShape = *dirsShape;
    *vCoeffsShape = *coeffsShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(VDIRS_PTR_INDEX, context->GetInputDataType(DIRS_PTR_INDEX));
    context->SetOutputDataType(VCOEFFS_PTR_INDEX, context->GetInputDataType(COEFFS_PTR_INDEX));
    return ge::GRAPH_SUCCESS;
}
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
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(SphericalHarmonicsBwd);
} // namespace ops
