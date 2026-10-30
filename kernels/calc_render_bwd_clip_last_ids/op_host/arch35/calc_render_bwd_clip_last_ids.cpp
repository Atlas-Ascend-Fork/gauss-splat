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
// kernels/calc_render_bwd_clip_last_ids/op_host/arch35/calc_render_bwd_clip_last_ids.cpp
// =============================================================================
//
// ROLE: Operator definition (OpDef) for CalcRenderBwdClipLastids.
//   The OpDef prototype is aligned item-by-item with
//   docs/calc_render_bwd_clip_last_ids/develop/proto.md §1 (OpDef):
//   - 9 REQUIRED inputs (vColor/vDepth/lastCumsum/error/gs/tileCoords float32,
//     offsets/gsIds int64, lastIds int32), all FORMAT_ND
//   - 1 REQUIRED output (vGs float32, FORMAT_ND; host 预清零后传入，kernel 仅原子累加)
//   - No attributes
//   - AICore config: ascend950 only; SetTiling/SetInferShape are NOT inlined
//   (tiling registered in op_host/arch35/calc_render_bwd_clip_last_ids_tiling.cpp,
//      infer-shape registered in op_host/arch35/calc_render_bwd_clip_last_ids_infershape.cpp)
//
//   The OpAICoreConfig block below is build plumbing (kernel file link via
//   opFile.value + dynamic-shape dispatch flags) required by the op_build /
//   npu_op_kernel_sources pipeline; it is not part of the interface prototype.
//
// CONTENTS:
//   - class CalcRenderBwdClipLastids : public OpDef — operator definition
//   - OP_ADD(CalcRenderBwdClipLastids) — registration macro
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : CalcRenderBwdClipLastids   — class name, OP_ADD argument
//   snake_case   : calc_render_bwd_clip_last_ids  — filename
//   UPPER_SNAKE  : calc_render_bwd_clip_last_ids  — (not directly used)
//
// =============================================================================

#include "register/op_def_registry.h"  // OpDef base class, OP_ADD macro

namespace ops {

// ---------------------------------------------------------------------------
// class CalcRenderBwdClipLastids : public OpDef
//
// Operator definition class (proto.md §1). The constructor chain-calls methods
// to define inputs and outputs using a builder pattern.
//
// KEY METHODS USED:
//   this->Input("name")   — begins defining a named input tensor
//     .ParamType(REQUIRED/OPTIONAL)  — whether the input must be provided
//     .DataType({...})              — single supported data type per input
//     .Format({...})                — FORMAT_ND (N-dimensional) layout
//     .UnknownShapeFormat({...})    — format for dynamic (unknown) shapes
//
//   this->Output("name")  — begins defining a named output tensor (same API)
//
//   OpAICoreConfig        — build plumbing configuration for AICore kernels
//     .ExtendCfgInfo(key, value)        — kernel file link (opFile.value)
//
//   this->AICore().AddConfig("chipname", config) — associate AICore config with a chip
// ---------------------------------------------------------------------------
class CalcRenderBwdClipLastids : public OpDef {
public:
    explicit CalcRenderBwdClipLastids(const char* name) : OpDef(name)
    {
        // --- Input: vColor (float32, [3,T,P]) ---
        // Forward-rendered per-channel color gradients (3 channels).
        this->Input("vColor")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: vDepth (float32, [1,T,P]) ---
        // Depth channel gradient (size-1 channel dim, broadcast to RGBD).
        this->Input("vDepth")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: lastCumsum (float32, [T,P]) ---
        // Reverse-scan cumulative sum seed per pixel.
        this->Input("lastCumsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: error (float32, [T,P]) ---
        // Per-pixel reconstruction error (dL/d rendered pixel).
        this->Input("error")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: gs (float32, [N,10]) ---
        // Per-gaussian 10-field packed attributes.
        this->Input("gs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: tileCoords (float32, [T,2,P]) ---
        // Per-tile pixel grid coordinates.
        this->Input("tileCoords")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: offsets (int64, [S]) ---
        // Three-segment offset layout (S = Vec_num + 2*T).
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: gsIds (int64, [G]) ---
        // Gaussian id scan list (values in [0, N)).
        this->Input("gsIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Input: lastIds (int32, [T,P]) ---
        // Reverse-traversal termination index per pixel (values in [0, G)).
        this->Input("lastIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- Output: vGs (float32, [N,10]) ---
        // Per-gaussian gradients; kernel 入口协作清零后仅原子累加（非 inplace）。
        // vGs.shape = gs.shape, vGs.dtype = gs.dtype (proto.md §4)。
        // 不声明 InitValue：kernel 侧 ZeroVGs+SyncAll 序言统一清零（Task 30——
        // 运行时 atomic clean 实测按陈旧尺寸 memset、越过更小的当前输出缓冲
        // 时间歇性越界写 aivec err95；自清零与 spec 的 vGs = 0 + Σ贡献及
        // Interface.md「调用方预清零」契约幂等兼容）。
        this->Output("vGs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // --- AICore Configuration for Ascend 950 (build plumbing) ---
        // opFile.value links the kernel implementation file
        // calc_render_bwd_clip_last_ids_apt.cpp (kernel entry name derives from it).
        // Dynamic-shape/rank flags keep the binary shape-agnostic; they do not
        // alter the operator interface prototype.
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)          // JIT compile for static shapes
            .DynamicFormatFlag(false)                        // Fixed format (no dynamic selection)
            .DynamicRankSupportFlag(true)                    // Support variable ranks at runtime
            .DynamicShapeSupportFlag(true)                   // Support variable shapes at runtime
            .NeedCheckSupportFlag(false)                     // Skip chip support check
            .PrecisionReduceFlag(true)                       // Allow precision reduction where legal
            .ExtendCfgInfo("opFile.value", "calc_render_bwd_clip_last_ids_apt");  // Links to kernel file calc_render_bwd_clip_last_ids_apt.cpp
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

// OP_ADD(CalcRenderBwdClipLastids):
//   Registers the CalcRenderBwdClipLastids OpDef class with the CANN operator definition registry.
//   This makes the operator visible to op_build and the rest of the CANN toolchain.
OP_ADD(CalcRenderBwdClipLastids);
} // namespace ops
