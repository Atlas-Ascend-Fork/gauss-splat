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
// flash_gaussian_build_mask_package/op_host/flash_gaussian_build_mask_infershape.cpp
// =============================================================================
//
// ROLE: Shape inference entry for the FlashGaussianBuildMask operator.
//   The OpDef (op_host/flash_gaussian_build_mask_def.cpp) registers this
//   function via SetInferShape(ge::InferShape), so its signature follows
//   docs/flash_gaussian_build_mask/develop/proto.md §1:
//     ge::graphStatus ge::InferShape(gert::InferShapeContext*)
//
//   Task 21 (prototype-consistent interface stub): the function body keeps
//   only a compilable stub (null-check + GRAPH_SUCCESS). The real shape
//   derivation (proto.md §4 Shape contract) is delivered by the later
//   InferShape/GEIR implementation tasks.
//
// CONTENTS:
//   - ge::InferShape(gert::InferShapeContext*) — interface stub
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — (referenced by the OpDef)
//   snake_case   : flash_gaussian_build_mask  — filename
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — (not used directly)
// =============================================================================

#include "exe_graph/runtime/infer_shape_context.h"      // InferShapeContext, gert::Shape

// ---------------------------------------------------------------------------
// ge::InferShape(context) — shape inference interface stub
//
// Signature matches the OpDef registration
// SetInferShape(ge::InferShape) in flash_gaussian_build_mask_def.cpp
// (docs/flash_gaussian_build_mask/develop/proto.md §1).
//
// Returns:
//   ge::GRAPH_SUCCESS on success (stub: no output shape is derived yet)
// ---------------------------------------------------------------------------
namespace ge {

// [REF_SAMPLE] 原样例（scale_custom 谱系）推导逻辑——输出 shape 直接拷贝输入0 shape，
// 经 IMPL_OP_INFERSHAPE 注册。本算子 8 输入 / 3 输出原型与其不符，整段注释保留以备追溯：
// ---------------------------------------------------------------------------
// static ge::graphStatus InferShape4FlashGaussianBuildMask(gert::InferShapeContext* context)
// {
//     // GetInputShape(0): reads the shape of the first input (index 0 = x)
//     const gert::Shape* input_shape = context->GetInputShape(0);
//     OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
//
//     // GetOutputShape(0): gets a mutable reference to output y's shape descriptor
//     gert::Shape* output_shape = context->GetOutputShape(0);
//     OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
//
//     // Copy: output shape = input shape (element-wise, scale/bias are broadcast)
//     *output_shape = *input_shape;
//
//     return ge::GRAPH_SUCCESS;
// }
//
// IMPL_OP_INFERSHAPE(FlashGaussianBuildMask).InferShape(InferShape4FlashGaussianBuildMask);
// ---------------------------------------------------------------------------
// [REF_SAMPLE] END

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 接口桩：本阶段不做 shape 推导（输出 shape 契约见 proto.md §4，由后续任务实现）
    return ge::GRAPH_SUCCESS;
}

} // namespace ge
