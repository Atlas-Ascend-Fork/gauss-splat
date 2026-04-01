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

#include "tiling/platform/platform_ascendc.h"
#include "calc_render_fwd_double_clip_gsids_tiling.h"
#include "register/op_def_registry.h"

constexpr int32_t NPIXEL_DIM = 2;


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    const gert::StorageShape* depthsShape = context->GetInputShape(0);
    const gert::StorageShape* tileCoordsShape = context->GetInputShape(1);
    
    CalcRenderFwdDoubleClipGsidsTilingData tiling;
    tiling.set_nPixel(tileCoordsShape->GetStorageShape().GetDim(NPIXEL_DIM));
    tiling.set_tileNum(tileCoordsShape->GetStorageShape().GetDim(0));
    tiling.set_nGauss(depthsShape->GetStorageShape().GetDim(0));

    context->SetBlockDim(ascendcPlatform.GetCoreNumAiv());
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    auto workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}
}


namespace ops {
class CalcRenderFwdDoubleClipGsids : public OpDef {
public:
    explicit CalcRenderFwdDoubleClipGsids(const char* name) : OpDef(name)
    {
        this->Input("gs").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tileCoords").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offsets").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gsIds").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("color").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("depth").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("lastCumsum").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("error").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("gsClipIndex").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("alphaClipIndex").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        this->SetInferShape(ge::InferShape);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(CalcRenderFwdDoubleClipGsids);
}