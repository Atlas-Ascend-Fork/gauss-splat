/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_ssim_backward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
static constexpr int32_t RESERVED_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const int64_t MAX_CORE_NUM = 48;
static const float UB_RATIO = 0.85;

// Input index map -- must match Op::Input(...) order below.
static const int64_t GRAD_SSIM_MAP_PTR_INDEX = 0;

static const int64_t DIM_B = 0;
static const int64_t DIM_C = 1;
static const int64_t DIM_H = 2;
static const int64_t DIM_W = 3;

static const int64_t WINDOW_SIZE_ATTR_INDEX = 0;
static const int64_t NEED_GRAD_IMG2_ATTR_INDEX = 1;
} // namespace

namespace optiling {
static ge::graphStatus Tiling4FusedSsimBackward(gert::TilingContext *context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t aivCoreNum = platformInfo.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    if (aivCoreNum > MAX_CORE_NUM) {
        aivCoreNum = MAX_CORE_NUM;
    }

    // Shape comes from grad_ssim_map (input 0). All other [B,C,H,W] inputs share it;
    // weight is 1D length-16.
    const gert::StorageShape *gShape = context->GetInputShape(GRAD_SSIM_MAP_PTR_INDEX);
    if (gShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t b = gShape->GetStorageShape().GetDim(DIM_B);
    int64_t c = gShape->GetStorageShape().GetDim(DIM_C);
    int64_t height = gShape->GetStorageShape().GetDim(DIM_H);
    int64_t width = gShape->GetStorageShape().GetDim(DIM_W);
    int64_t bcNum = b * c;
    if (bcNum == 0 || height == 0 || width == 0) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    // CANN 8.5: Int() attr stored in 8-byte slot but only low 32 bits hold value;
    // upper 32 bits carry a type-tag-ish 0x10007 mask -- read via int32_t ptr to avoid garbage.
    int64_t windowSize = static_cast<int64_t>(*(attrs->GetAttrPointer<int32_t>(WINDOW_SIZE_ATTR_INDEX)));
    int64_t radius = windowSize / 2;
    int64_t needGradImg2 = static_cast<int64_t>(*(attrs->GetAttrPointer<int32_t>(NEED_GRAD_IMG2_ATTR_INDEX)));

    // Same row-band tiling as forward: split image's H rows across coresPerImage
    // cores so each core sees its 7-row halo within a single image (no bc cross).
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

    // Per-core GM scratch for separable conv intermediate: K_a / K_b / K_g vertical
    // conv accumulators with halo prefix + tail pad. Same padLen layout as forward.
    // For 1-sided: 3 K maps; 2-sided (need_g2=1): 4 K maps (add K_a2).
    const int64_t HALO_LEFT = 8;
    const int64_t ALIGN = 8;
    const int64_t haloAlign = 8;

    int64_t numKMaps = (needGradImg2 == 1) ? 4 : 3;
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
            int64_t ubNeed = 64 + (6 + 5 + numKMaps + 1) * wA * 4 + numKMaps * pLen * 4;
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
    int64_t perCoreWsFloats = numKMaps * padLen;
    int64_t userWsBytes = needCoreNum * perCoreWsFloats * static_cast<int64_t>(sizeof(float));
    size_t sysWorkspaceByteSize = static_cast<size_t>(platformInfo.GetLibApiWorkSpaceSize());

    FusedSsimBackwardTilingData tiling;
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
    tiling.set_needGradImg2(needGradImg2);
    tiling.set_chunkW(chunkW);
    tiling.set_numWChunks(numWChunks);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceByteSize + static_cast<size_t>(userWsBytes) + RESERVED_WORKSPACE_SIZE;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // Tiling key: 1 = 1-sided (grad_img1 only path), 2 = 2-sided (grad_img1 + grad_img2).
    // Kernel uses this to dispatch to the right Process() and avoid the K_a2 compute on hot path.
    context->SetTilingKey(needGradImg2 == 1 ? 2 : 1);
    context->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // grad_img1 (output 0) and grad_img2 (output 1) both share grad_ssim_map's [B,C,H,W].
    const gert::Shape *gShape = context->GetInputShape(0);
    if (gShape == nullptr) {
        return GRAPH_FAILED;
    }
    for (int32_t i = 0; i < 2; i++) {
        gert::Shape *outShape = context->GetOutputShape(i);
        if (outShape != nullptr) {
            *outShape = *gShape;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    for (int32_t i = 0; i < 2; i++) {
        context->SetOutputDataType(i, context->GetInputDataType(0));
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class FusedSsimBackward : public OpDef {
public:
    explicit FusedSsimBackward(const char *name) : OpDef(name)
    {
        // 9 inputs: grad_ssim_map, img1, img2, weight, mu1, mu2, S11, S22, S12
        this->Input("grad_ssim_map")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
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
        this->Input("mu1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("mu2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("S11")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("S22")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("S12")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // 2 outputs: grad_img1 (hot path), grad_img2 (only filled when need_grad_img2=true,
        // else kernel skips its compute and the buffer holds zeros from the workspace).
        this->Output("grad_img1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("grad_img2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // Attrs: window_size matches forward; need_grad_img2 controls 1-sided vs 2-sided
        // tiling key. Both are required (the Python wrapper sets need_grad_img2 from
        // ctx.needs_input_grad[1]).
        this->Attr("window_size").AttrType(REQUIRED).Int();
        // need_grad_img2 用 Int (0/1) 而非 Bool: CANN 8.5 Bool attr 编码不稳, fused_adam 同坑.
        this->Attr("need_grad_img2").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::Tiling4FusedSsimBackward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(FusedSsimBackward);
} // namespace ops
