/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// flash_gaussian_build_mask_package/op_host/flash_gaussian_build_mask_def.cpp
// =============================================================================
//
// ROLE: Operator definition (OpDef) for FlashGaussianBuildMask.
//   The OpDef below is kept item-by-item consistent with the interface truth
//   recorded in docs/flash_gaussian_build_mask/develop/proto.md:
//   - 8 OPTIONAL inputs  : means2d / opacity / conics / covars2d / depths /
//                          cnt / tile_grid / gather_mask
//   - 3 REQUIRED outputs : tile_sum / tile_depths / gauss_index
//   - 3 required attrs   : image_width (Float) / image_height (Float) /
//                          tile_size (Int)
//   - InferShape / InferDataType / Tiling are registered through the OpDef
//     (SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType),
//     AICore().SetTiling(optiling::TilingForFlashGaussianBuildMask)).
//
// CONTENTS:
//   - forward declarations of ge::InferShape / ge::InferDataType /
//     optiling::TilingForFlashGaussianBuildMask (stub implementations live in
//     op_host/flash_gaussian_build_mask_infershape.cpp,
//     op_graph/flash_gaussian_build_mask_graph_infer.cpp and
//     op_host/arch35/flash_gaussian_build_mask_tiling_arch35.cpp)
//   - class FlashGaussianBuildMask : public OpDef — operator definition
//   - OP_ADD(FlashGaussianBuildMask) — registration macro (kept)
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — class name, OP_ADD argument
//   snake_case   : flash_gaussian_build_mask  — filename
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — (not directly used)
// =============================================================================

#include "register/op_def_registry.h"  // OpDef base class, OP_ADD macro

// ---------------------------------------------------------------------------
// Forward declarations of the host functions referenced by the OpDef below.
// All three are interface stubs at this stage (Task 21): their signatures
// follow proto.md, their bodies contain no computation yet.
// ---------------------------------------------------------------------------
namespace ge {
ge::graphStatus InferShape(gert::InferShapeContext* context);
ge::graphStatus InferDataType(gert::InferDataTypeContext* context);
} // namespace ge

namespace optiling {
ge::graphStatus TilingForFlashGaussianBuildMask(gert::TilingContext* context);
} // namespace optiling

namespace ops {

// ---------------------------------------------------------------------------
// class FlashGaussianBuildMask : public OpDef
//
// Operator definition class. The constructor chain-calls methods to define
// inputs, outputs, and attributes using a builder pattern. Every
// Input / Output / Attr entry (name, order, ParamType, DataType, Format)
// matches docs/flash_gaussian_build_mask/develop/proto.md §1 exactly.
// ---------------------------------------------------------------------------
class FlashGaussianBuildMask : public OpDef {
public:
    explicit FlashGaussianBuildMask(const char* name) : OpDef(name)
    {
        this->Input("means2d").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("opacity").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("conics").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("covars2d").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depths").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cnt").ParamType(OPTIONAL).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tile_grid").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("gather_mask").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("tile_sum").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("tile_depths").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("gauss_index").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("image_width").Float();
        this->Attr("image_height").Float();
        this->Attr("tile_size").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingForFlashGaussianBuildMask);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};

// OP_ADD(FlashGaussianBuildMask):
//   Registers the FlashGaussianBuildMask OpDef class with the CANN operator
//   definition registry (kept from the original project; under OP_TILING_LIB
//   it also registers the tiling function set via SetTiling above).
OP_ADD(FlashGaussianBuildMask);
} // namespace ops
