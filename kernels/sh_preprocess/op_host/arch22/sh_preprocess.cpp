/**
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "register/op_def_registry.h"
#include "sh_preprocess_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t CAMTOWORLDS_PTR_IDX = 0;
const uint32_t MEANS_PTR_IDX = 1;
const uint32_t RAYSD_OUT_IDX = 0;

const uint32_t MEANS_N_DIM = 1;

const uint32_t AXIS_NUM = 3;
const uint32_t ALIGN = 8;
const int32_t FLOAT_SZ = 4;
const float UB_RATIO = 0.8;
}

namespace optiling {
static ge::graphStatus TilingForShPreprocess(gert::TilingContext *context)
{
    ShPreprocessTilingData tiling;
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto meansShape = context->GetInputShape(MEANS_PTR_IDX);
    if (meansShape == nullptr) return ge::GRAPH_FAILED;
    int64_t gaussianNum = meansShape->GetStorageShape().GetDim(MEANS_N_DIM);

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) return ge::GRAPH_FAILED;
    auto plat = platform_ascendc::PlatformAscendC(platform);
    uint64_t ubTotalSize;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotalSize);
    uint32_t blockDim = plat.GetCoreNumAiv();
    if (blockDim == 0) return ge::GRAPH_FAILED;

    uint32_t taskNumPerCore = (gaussianNum + blockDim - 1) / blockDim;

    // means, directions and norm occupy seven float arrays in UB.
    uint64_t bufPerLoop = (AXIS_NUM * 2 + 1) * FLOAT_SZ;
    uint32_t taskNumPerLoop = static_cast<uint32_t>((ubTotalSize * UB_RATIO) / bufPerLoop);
    taskNumPerLoop = (taskNumPerLoop / ALIGN) * ALIGN;
    if (taskNumPerLoop == 0) taskNumPerLoop = ALIGN;
    uint32_t taskLoop = (taskNumPerCore + taskNumPerLoop - 1) / taskNumPerLoop;
    uint32_t tailNum = 0;

    tiling.set_gaussianNum(gaussianNum);
    tiling.set_blockDim(blockDim);
    tiling.set_taskNumPerLoop(taskNumPerLoop);
    tiling.set_taskStartIndex(0);
    tiling.set_taskNumPerCore(taskNumPerCore);
    tiling.set_taskLoop(taskLoop);
    tiling.set_tailNum(tailNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1);

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = plat.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;

    const gert::Shape *meansShape = context->GetInputShape(MEANS_PTR_IDX);
    if (meansShape == nullptr) return ge::GRAPH_FAILED;

    uint32_t N = meansShape->GetDim(MEANS_N_DIM);

    gert::Shape *raysdShape = context->GetOutputShape(RAYSD_OUT_IDX);
    raysdShape->SetDimNum(3);
    raysdShape->SetDim(0, 1);
    raysdShape->SetDim(1, AXIS_NUM);
    raysdShape->SetDim(2, N);


    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto dt = context->GetInputDataType(CAMTOWORLDS_PTR_IDX);
    context->SetOutputDataType(RAYSD_OUT_IDX, dt);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class ShPreprocess : public OpDef {
public:
    explicit ShPreprocess(const char *name) : OpDef(name)
    {
        this->Input("camtoworlds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("means")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("rays_d")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingForShPreprocess);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ShPreprocess);
}
