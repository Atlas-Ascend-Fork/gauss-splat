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
// kernels/calc_render_simt_bwd/op_host/arch35/calc_render_simt_bwd_infershape.cpp
// =============================================================================
//
// ROLE: Shape inference for the CalcRenderSimtBwd operator.
//   Aligned with docs/calc_render_simt_bwd/develop/proto.md §4 (Shape 契约):
//     vGs.shape = gs.shape = (N, 10), vGs.dtype = gs.dtype.
//   The output shape is copied from input "gs" (index 4).
//
// CONTENTS:
//   - InferShape4CalcRenderSimtBwd() — copies gs shape to vGs shape
//   - IMPL_OP_INFERSHAPE(CalcRenderSimtBwd).InferShape(...) — registration macro
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : CalcRenderSimtBwd   — IMPL_OP_INFERSHAPE argument, function name suffix
//   snake_case   : calc_render_simt_bwd  — filename
//   UPPER_SNAKE  : CALC_RENDER_SIMT_BWD  — (not used directly)
//
// =============================================================================

#include "register/op_impl_registry.h"                  // IMPL_OP_INFERSHAPE macro
#include "exe_graph/runtime/infer_shape_context.h"      // InferShapeContext, gert::Shape
#include "op_common/log/log.h"                          // OP_CHECK_NULL_WITH_CONTEXT macro

using namespace ge;

namespace ops {

// ---------------------------------------------------------------------------
// InferShape4CalcRenderSimtBwd(context) — shape inference function
//
// Copies the shape of input[4] (gs) to output[0] (vGs), per proto.md §4:
//   vGs.shape = gs.shape = (N, 10)
//
// Call-chain grid-consistency validation (Interface.md「Broadcast 规则」/spec
// boundary「像素网格维冲突 shape_mismatch」: the conflict is rejected BEFORE the
// kernel — negative scenarios are unreachable by any tilingKey; per
// DESIGN-BRANCH-0.md §9 this is verified at the call-chain layer, i.e. here):
//   vColor [3,T,P] / vDepth [1,T,P] / lastCumsum [T,P] / error [T,P] /
//   tileCoords [T,2,P] / lastIds [T,P] must share the same pixel grid (T,P)
//   and gs must be [N,10]. Any mismatch => GRAPH_FAILED (BND-03).
//
// Parameters:
//   context — InferShapeContext providing access to input shapes and
//             allowing setting of output shapes
//
// Returns:
//   ge::graphStatus — ge::GRAPH_SUCCESS on success
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4CalcRenderSimtBwd(gert::InferShapeContext* context)
{
    // GetInputShape(4): reads the shape of the "gs" input (index 4)
    const gert::Shape* gsShape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, gsShape);

    // GetOutputShape(0): gets a mutable reference to output vGs's shape descriptor
    gert::Shape* vGsShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, vGsShape);

    // Copy: output shape = gs shape (vGs.shape = gs.shape, proto.md §4)
    *vGsShape = *gsShape;

    // ---- pixel-grid consistency (call-chain layer, see function comment) ----
    // Reference grid (T, P) and per-input expected dims; short-circuit on the
    // first violation. Rank-checked before GetDim to stay within bounds.
    const gert::Shape* coordsShape = context->GetInputShape(5);   // tileCoords [T,2,P]
    OP_CHECK_NULL_WITH_CONTEXT(context, coordsShape);
    if (coordsShape->GetDimNum() != 3U) {
        OP_LOGE(context->GetNodeName(), "tileCoords rank %zu != 3", coordsShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    const int64_t tRef = coordsShape->GetDim(0);
    const int64_t pRef = coordsShape->GetDim(2);

    if (coordsShape->GetDim(1) != 2 || pRef == 0 || pRef > 1024) {
        OP_LOGE(context->GetNodeName(), "tileCoords requires [T,2,P], 1<=P<=1024");
        return ge::GRAPH_FAILED;
    }

    auto gridOk = [tRef, pRef](const gert::Shape* s, size_t rank, int64_t dimT, int64_t dimP) {
        return (s != nullptr) && (s->GetDimNum() == rank)
            && ((dimT < 0) || (s->GetDim(static_cast<size_t>(dimT)) == tRef))
            && ((dimP < 0) || (s->GetDim(static_cast<size_t>(dimP)) == pRef));
    };

    const gert::Shape* vColorShape = context->GetInputShape(0);   // vColor [3,T,P]
    const gert::Shape* vDepthShape = context->GetInputShape(1);   // vDepth [1,T,P]
    const gert::Shape* lastCumsumShape = context->GetInputShape(2); // lastCumsum [T,P]
    const gert::Shape* errorShape = context->GetInputShape(3);    // error [T,P]
    const gert::Shape* lastIdsShape = context->GetInputShape(8);  // lastIds [T,P]
    OP_CHECK_NULL_WITH_CONTEXT(context, vColorShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vDepthShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, lastCumsumShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, errorShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, lastIdsShape);
    if (!gridOk(vColorShape, 3U, 1, 2) || (vColorShape->GetDim(0) != 3)) {
        OP_LOGE(context->GetNodeName(), "vColor shape mismatch vs pixel grid (shape_mismatch)");
        return ge::GRAPH_FAILED;
    }
    if (!gridOk(vDepthShape, 3U, 1, 2) || (vDepthShape->GetDim(0) != 1)) {
        OP_LOGE(context->GetNodeName(), "vDepth shape mismatch vs pixel grid (shape_mismatch)");
        return ge::GRAPH_FAILED;
    }
    if (!gridOk(lastCumsumShape, 2U, 0, 1)) {
        OP_LOGE(context->GetNodeName(), "lastCumsum shape mismatch vs pixel grid (shape_mismatch)");
        return ge::GRAPH_FAILED;
    }
    if (!gridOk(errorShape, 2U, 0, 1)) {
        OP_LOGE(context->GetNodeName(), "error shape mismatch vs pixel grid (shape_mismatch)");
        return ge::GRAPH_FAILED;
    }
    if (!gridOk(lastIdsShape, 2U, 0, 1)) {
        OP_LOGE(context->GetNodeName(), "lastIds shape mismatch vs pixel grid (shape_mismatch)");
        return ge::GRAPH_FAILED;
    }
    if ((gsShape->GetDimNum() != 2U) || (gsShape->GetDim(1) != 10)) {
        OP_LOGE(context->GetNodeName(), "gs shape must be [N,10] (shape_mismatch)");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// [REF_SAMPLE] 原样例（y = x * scale + bias）推理逻辑：输出形状 = 输入 x 的形状。
// 本算子原型下 vGs 与 vColor 形状不同（(N,10) vs (3,T,P)），该逻辑不适用，仅作参考保留。
// // GetInputShape(0): reads the shape of the first input (index 0 = x)
// const gert::Shape* input_shape = context->GetInputShape(0);
// OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
// // GetOutputShape(0): gets a mutable reference to output y's shape descriptor
// gert::Shape* output_shape = context->GetOutputShape(0);
// OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
// // Copy: output shape = input shape (element-wise, scale/bias are broadcast)
// *output_shape = *input_shape;
// return ge::GRAPH_SUCCESS;

// IMPL_OP_INFERSHAPE(CalcRenderSimtBwd).InferShape(func):
//   Registers InferShape4CalcRenderSimtBwd as the shape inference function
//   for the CalcRenderSimtBwd operator type.
IMPL_OP_INFERSHAPE(CalcRenderSimtBwd).InferShape(InferShape4CalcRenderSimtBwd);

} // namespace ops
