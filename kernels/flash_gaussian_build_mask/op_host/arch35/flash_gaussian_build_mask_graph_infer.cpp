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
// flash_gaussian_build_mask_package/op_graph/flash_gaussian_build_mask_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference entry for the FlashGaussianBuildMask
//   operator. The OpDef (op_host/flash_gaussian_build_mask_def.cpp) registers
//   this function via SetInferDataType(ge::InferDataType), so its signature
//   follows docs/flash_gaussian_build_mask/develop/proto.md §1:
//     ge::graphStatus ge::InferDataType(gert::InferDataTypeContext*)
//
//   Task 21 (prototype-consistent interface stub): the function body keeps
//   only a compilable stub (null-check + GRAPH_SUCCESS). The real dtype
//   derivation (tile_sum → int32, tile_depths / gauss_index → float32,
//   see proto.md §1 / §2) is delivered by the later GEIR implementation task.
//
// CONTENTS:
//   - ge::InferDataType(gert::InferDataTypeContext*) — interface stub
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — (referenced by the OpDef)
//   snake_case   : flash_gaussian_build_mask  — filename
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — (not used directly)
// =============================================================================

#include "exe_graph/runtime/infer_datatype_context.h"  // InferDataTypeContext

// ---------------------------------------------------------------------------
// ge::InferDataType(context) — data type inference interface stub
//
// Signature matches the OpDef registration
// SetInferDataType(ge::InferDataType) in flash_gaussian_build_mask_def.cpp
// (docs/flash_gaussian_build_mask/develop/proto.md §1).
//
// Returns:
//   ge::GRAPH_SUCCESS on success (stub: no output dtype is derived yet)
// ---------------------------------------------------------------------------
namespace ge {

// [REF_SAMPLE] 原样例（scale_custom 谱系）推导逻辑——输出 dtype 取输入0 dtype，
// 经 IMPL_OP 注册。本算子 8 输入 / 3 输出原型与其不符，整段注释保留以备追溯：
// ---------------------------------------------------------------------------
// static ge::graphStatus InferDataTypeForFlashGaussianBuildMask(gert::InferDataTypeContext *context)
// {
//     // GetInputDataType(0): read the dtype of the first input (index 0 = x)
//     const ge::DataType xDataType = context->GetInputDataType(0);
//     // SetOutputDataType(0, dtype): set output y's dtype to match input x
//     context->SetOutputDataType(0, xDataType);
//     return ge::GRAPH_SUCCESS;
// }
//
// IMPL_OP(FlashGaussianBuildMask).InferDataType(InferDataTypeForFlashGaussianBuildMask);
// ---------------------------------------------------------------------------
// [REF_SAMPLE] END

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 接口桩：本阶段不做 dtype 推导（输出 dtype 契约见 proto.md §1/§2，由后续任务实现）
    return ge::GRAPH_SUCCESS;
}

} // namespace ge
