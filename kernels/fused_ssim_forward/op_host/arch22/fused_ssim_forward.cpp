/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_ssim_forward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
static constexpr int32_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const int64_t MAX_CORE_NUM = 48;
static const float UB_RATIO = 0.85;

static const int64_t IMG1_PTR_INDEX = 0;

static const int64_t DIM_B = 0;
static const int64_t DIM_C = 1;
static const int64_t DIM_H = 2;
static const int64_t DIM_W = 3;

static const int64_t WINDOW_SIZE_ATTR_INDEX = 0;
} // namespace

namespace optiling {
static ge::graphStatus Tiling4FusedSsimForward(gert::TilingContext *context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t aivCoreNum = platformInfo.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    if (aivCoreNum > MAX_CORE_NUM) {
        aivCoreNum = MAX_CORE_NUM;
    }

    const gert::StorageShape *img1Shape = context->GetInputShape(IMG1_PTR_INDEX);
    if (img1Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t b = img1Shape->GetStorageShape().GetDim(DIM_B);
    int64_t c = img1Shape->GetStorageShape().GetDim(DIM_C);
    int64_t height = img1Shape->GetStorageShape().GetDim(DIM_H);
    int64_t width = img1Shape->GetStorageShape().GetDim(DIM_W);
    int64_t bcNum = b * c;
    if (bcNum == 0 || height == 0 || width == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    int64_t windowSize = *(attrs->GetAttrPointer<int32_t>(WINDOW_SIZE_ATTR_INDEX));
    int64_t radius = windowSize / 2;

    int64_t coresPerImage = aivCoreNum / bcNum;
    if (coresPerImage < 1) {
        coresPerImage = 1;
    }
    if (coresPerImage > height) {
        coresPerImage = height;
    }
    int64_t rowsPerCore = (height + coresPerImage - 1) / coresPerImage;
    coresPerImage = (height + rowsPerCore - 1) / rowsPerCore;
    int64_t needCoreNum = coresPerImage * bcNum;

    const int64_t HALO_LEFT = 8;
    const int64_t ALIGN = 8;
    const int64_t haloAlign = 8;

    uint64_t ubSizeBytesU64 = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizeBytesU64);
    int64_t ubSizeBytes = static_cast<int64_t>(ubSizeBytesU64 * UB_RATIO);
    int64_t chunkW = width;
    {
        int64_t lo = ALIGN;
        int64_t hi = ((width + ALIGN - 1) / ALIGN) * ALIGN;
        if (hi < lo) { hi = lo; }
        while (lo <= hi) {
            int64_t mid = ((lo + hi) / 2 / ALIGN) * ALIGN;
            if (mid < lo) { mid = lo; }
            int64_t effW = mid + 2 * haloAlign;
            int64_t wA = ((effW + ALIGN - 1) / ALIGN) * ALIGN;
            int64_t pLen = ((HALO_LEFT + effW + radius + ALIGN - 1) / ALIGN) * ALIGN;
            int64_t br = 2 * radius + 1;
            int64_t ubNeed = 64 + (br + br + 5 + 5 + 1) * wA * 4 + 5 * pLen * 4;
            if (ubNeed <= ubSizeBytes) {
                chunkW = mid;
                lo = mid + ALIGN;
            } else {
                hi = mid - ALIGN;
            }
        }
    }
    if (chunkW > width) { chunkW = width; }
    int64_t numWChunks = (width + chunkW - 1) / chunkW;

    int64_t padLen = ((HALO_LEFT + chunkW + 2 * haloAlign + radius) + ALIGN - 1) / ALIGN * ALIGN;
    int64_t perCoreWsFloats = 5 * padLen;
    int64_t userWsBytes = needCoreNum * perCoreWsFloats * static_cast<int64_t>(sizeof(float));
    size_t sysWorkspaceByteSize = static_cast<size_t>(platformInfo.GetLibApiWorkSpaceSize());

    FusedSsimForwardTilingData tiling;
    tiling.set_needCoreNum(needCoreNum);
    tiling.set_bcNum(bcNum);
    tiling.set_height(height);
    tiling.set_width(width);
    tiling.set_windowSize(windowSize);
    tiling.set_radius(radius);
    tiling.set_coresPerImage(coresPerImage);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_sysWsBytes(static_cast<int64_t>(sysWorkspaceByteSize));
    tiling.set_perCoreWsFloats(perCoreWsFloats);
    tiling.set_chunkW(chunkW);
    tiling.set_numWChunks(numWChunks);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceByteSize + static_cast<size_t>(userWsBytes) + RESERVED_WORKSPACE_SIZE;

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
    const gert::Shape *img1Shape = context->GetInputShape(0);
    if (img1Shape == nullptr) {
        return GRAPH_FAILED;
    }
    for (int32_t i = 0; i < 6; i++) {
        gert::Shape *outShape = context->GetOutputShape(i);
        if (outShape != nullptr) {
            *outShape = *img1Shape;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    for (int32_t i = 0; i < 6; i++) {
        context->SetOutputDataType(i, context->GetInputDataType(0));
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FusedSsimForward : public OpDef {
public:
    explicit FusedSsimForward(const char *name) : OpDef(name)
    {
        this->Input("img1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("img2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("ssim_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mu1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("mu2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("S11")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("S22")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("S12")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("window_size").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::Tiling4FusedSsimForward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FusedSsimForward);
} // namespace ops
