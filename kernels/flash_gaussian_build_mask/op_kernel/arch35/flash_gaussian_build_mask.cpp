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
// gauss-splat kernels/flash_gaussian_build_mask/op_kernel/arch35/flash_gaussian_build_mask.cpp
// （仓内适配：入口按仓内 arch35 算子命名约定 op_kernel/arch35/<op>.cpp 自交付包
//   op_kernel/flash_gaussian_build_mask_apt.cpp 重命名，arch35/ 前缀 include 改为
//   同目录引用——仓内构建将 arch35 内核源码扁平化到同一编译目录）
// =============================================================================
//
// ROLE: Ascend C kernel entry point for FlashGaussianBuildMask (arch35 /
//   Ascend 950 / A5, AIV-only).
//
//   按 docs/flash_gaussian_build_mask/design/Kernel.md §1（reduction 范式入口规范）
//   与 docs/flash_gaussian_build_mask/design/TilingKey.md 落地 TPL 双模板入口：
//     - 入口统一 REGISTER_NONE_TILING，模板参数与 TilingKey 字段一一对应、顺序一致
//       （单 BOOL 字段 isEmptyTensor，arch35/flash_gaussian_build_mask_struct.h 的
//       ASCENDC_TPL_ARGS_DECL(FlashGaussianBuildMask, ASCENDC_TPL_BOOL_DECL(...))）；
//       asc_opc 按 ASCENDC_TPL_SEL 两块实例化 sub-kernel …_0_mix_aiv（normal）与
//       …_1_mix_aiv（empty），runtime 由 host SetTilingKey 选符号（禁止
//       TilingKeyIS 数值分支）；禁止加 typename D_T（dtype 走 DTYPE_MEANS2D
//       编译期实例化，dtype 不进 TilingKey）。
//     - AscendC::TPipe pipe 统一在入口申请，以指针方式传入各模板 Init；
//     - if constexpr 分发：isEmptyTensor → FlashGaussianBuildMaskEmptyKernel
//       （空判提前返回 + tile_sum 零填充直出 + gauss_index/tile_depths 空列表
//       无写回，Task 28.2.4 / DESIGN-BRANCH-1.md §5）；
//       否则 → FlashGaussianBuildMaskKernel（三级过滤 + GatherMask 有序压缩，
//       Task 28.1.4 / DESIGN-BRANCH-0.md §3–§5）。
//     - 各分叉分别 GET_TILING_DATA_WITH_STRUCT 获取对应 TilingData（2 份 struct：
//       normal 用 FlashGaussianBuildMaskTilingData，empty 用
//       FlashGaussianBuildMaskEmptyTilingData，互不复用）。
//
//   签名与 docs/flash_gaussian_build_mask/develop/proto.md §3 一致：8 输入 + 3 输出
//   + workspace + tiling，全 GM_ADDR；第 8 参数保留参考源字面名 gathermask
//   （OpDef 输入名为 gather_mask，历史拼写差异按 proto.md 原样保留）。
//
// CONTENTS:
//   - template <bool isEmptyTensor> __global__ __aicore__ void
//     flash_gaussian_build_mask(...)
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — FlashGaussianBuildMaskTilingData 等
//   snake_case   : flash_gaussian_build_mask  — filename, kernel entry name
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — (not directly used)
//
// [REF_SAMPLE] 原样例（scale_custom 谱系）extern "C" 非模板入口（RANK 4/8 双
// tilingKey 分发）与样例 kernel 类（NDDMA 广播乘加）已分别由本 TPL 模板入口与
// arch35/flash_gaussian_build_mask_kernel.h 的本算子实现替换（Task 28.1.4，
// Kernel.md §1 / TilingKey.md），留此注释备溯。
// =============================================================================

#include "kernel_operator.h"                                        // Ascend C kernel framework (AscendC:: namespace)
#include "flash_gaussian_build_mask_tiling_struct.h"                // FlashGaussianBuildMaskTilingData / FlashGaussianBuildMaskEmptyTilingData
#include "flash_gaussian_build_mask_struct.h"                       // ASCENDC_TPL BOOL isEmptyTensor 声明（模板实例化依据，asc_opc 检测）
#include "flash_gaussian_build_mask_kernel.h"                       // FlashGaussianBuildMaskKernel<T>（normal 主模板，tilingKey=0）
#include "flash_gaussian_build_mask_empty.h"                        // FlashGaussianBuildMaskEmptyKernel<T>（empty 短路模板，tilingKey=1）

// ===========================================================================
// template <bool isEmptyTensor> __global__ __aicore__ void flash_gaussian_build_mask
//
// Key 传递: TilingKey.md（isEmptyTensor）→ flash_gaussian_build_mask_struct.h
//           （ASCENDC_TPL_ARGS_DECL/SEL）→ 本文件模板参数；asc_opc 按 TPL_SEL 块
//           实例化 …_0_mix_aiv（normal）/ …_1_mix_aiv（empty）。
//
// Parameters (all GM_ADDR, proto.md §3 order):
//   means2d    — input, [B, C, 2, N] float32（投影后 2D 高斯中心，x/y 通道在前）
//   opacity    — input, [B, C, 1, N] float32
//   conics     — input, [B, C, 3, N] float32（conic 逆协方差 c00/c01/c11）
//   covars2d   — input, [B, C, 3, N] float32（2D 协方差 cov00/cov01/cov11）
//   depths     — input, [B, C, 1, N] float32
//   cnt        — input, [B, C] or [B, C, 1] int32（每视图有效高斯数，g < cnt 截断）
//   tile_grid  — input, [NumTile, 2] float32（[t,0]=ty、[t,1]=tx）
//   gathermask — input, [BLOCK_NUM, 20, N] float32（gather_mask 核间/级间中转区，
//                语义上不参与输出数学）
//   tile_sum     — output, [B, C, NumTile, 1] int32
//   tile_depths  — output, [B, C, NumTile, N] float32
//   gauss_index  — output, [B, C, NumTile, N] float32
//   workspace  — workspace buffer（仅框架系统段，kernel 不感知）
//   tiling     — tiling data buffer（normal: FlashGaussianBuildMaskTilingData /
//                empty: FlashGaussianBuildMaskEmptyTilingData）
// ===========================================================================
template <bool isEmptyTensor>
__global__ __aicore__ void flash_gaussian_build_mask(GM_ADDR means2d, GM_ADDR opacity, GM_ADDR conics,
                                                     GM_ADDR covars2d, GM_ADDR depths, GM_ADDR cnt,
                                                     GM_ADDR tile_grid, GM_ADDR gathermask, GM_ADDR tile_sum,
                                                     GM_ADDR tile_depths, GM_ADDR gauss_index,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    // REGISTER_NONE_TILING: tiling data 由 GET_TILING_DATA_WITH_STRUCT 手动读取，
    // 非框架自动 tiling。
    REGISTER_NONE_TILING;

    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY): 本算子为纯 Vector 算子，
    // 仅在 AIV（AI Vector）核上运行。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;

    if constexpr (isEmptyTensor) {
        // empty 短路模板（tilingKey=1，…_1_mix_aiv）：tile_sum 零填充直出
        // （Duplicate 0 + DataCopyPad 分块），gauss_index/tile_depths 为 0 元素
        // 空列表无写回，fillTotal=0 唯一在环核空转早退
        // （DESIGN-BRANCH-1.md §5，Task 28.2.4）。
        GET_TILING_DATA_WITH_STRUCT(FlashGaussianBuildMaskEmptyTilingData, tilingData, tiling);
        FlashGaussianBuildMaskEmptyKernel<DTYPE_MEANS2D> op;
        op.Init(tile_sum, &tilingData, &pipe);
        op.Process();
    } else {
        // normal 主模板（tilingKey=0，…_0_mix_aiv）：Y 预过滤 → 包围盒粗过滤 →
        // 中心/椭圆精过滤 → GatherMask 有序压缩（DESIGN-BRANCH-0.md §3–§5）。
        GET_TILING_DATA_WITH_STRUCT(FlashGaussianBuildMaskTilingData, tilingData, tiling);
        FlashGaussianBuildMaskKernel<DTYPE_MEANS2D> op;
        op.Init(means2d, opacity, conics, covars2d, depths, cnt, tile_grid, gathermask, tile_sum, tile_depths,
                gauss_index, &tilingData, &pipe);
        op.Process();
    }
}
