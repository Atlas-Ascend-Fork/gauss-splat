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
// flash_gaussian_build_mask_package/op_kernel/arch35/flash_gaussian_build_mask_tiling_struct.h
// =============================================================================
//
// ROLE: TilingData structs shared between host-side tiling and device-side
//   kernel for the FlashGaussianBuildMask operator on arch35 (Ascend 950).
//   Per docs/flash_gaussian_build_mask/design/TilingData.md this operator has
//   TWO independent TilingData structs (Reduction-paradigm hard constraint:
//   the empty short-circuit template must NOT reuse the main struct):
//   - FlashGaussianBuildMaskTilingData       — normal main template (tilingKey=0):
//       flat (b,c)×tile task-space multicore split + UB gaussian batching
//   - FlashGaussianBuildMaskEmptyTilingData  — empty short-circuit template
//       (tilingKey=1): tile_sum zero-fill split
//   plus the gather_mask slot-structure constant GATHER_NUM_SLOTS.
//
//   The structs are populated by the host-side tiling function
//   (op_host/arch35/flash_gaussian_build_mask_tiling_arch35.cpp) and read by
//   the device-side kernel templates. Both sides see the same struct layout.
//
//   Memory layout note: the structs must stay trivially copyable (scalar
//   fields only, no pointers / virtual methods) because they are serialised
//   into a raw byte buffer passed from host to device via global memory
//   (GET_TILING_DATA_WITH_STRUCT / GetTilingData<T>).
//
// CONTENTS:
//   - GATHER_NUM_SLOTS = 20  (gather_mask 2nd-dim slot-structure constant)
//   - FlashGaussianBuildMaskTilingData        (normal, 15 fields)
//   - FlashGaussianBuildMaskEmptyTilingData   (empty, 6 fields)
//
// FIELD UNITS:
//   - Split/scale fields (taskTotal, taskNumPerScore, ..., gaussNumPerLoop,
//     fillTotal, ...) are ELEMENT COUNTS, not bytes and not bits.
//   - numScore / numLcore / blockDim are CORE COUNTS.
//   - imageWidth / imageHeight / tileSize are pixel-domain float attributes.
//   - ubTotalSize is BYTES (platform GetCoreMemSize(UB)).
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — struct name prefix
//   snake_case   : flash_gaussian_build_mask  — filename
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — (not directly used here)
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   FlashGaussianBuildMaskTilingData → FooBarTilingData
//   FlashGaussianBuildMaskEmptyTilingData → FooBarEmptyTilingData
//   GATHER_NUM_SLOTS → rename only if the output slot layout changes
//   flash_gaussian_build_mask_tiling_struct → foo_bar_tiling_struct (filename)
//   arch35 → appropriate arch directory for target hardware
//
// =============================================================================

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// GATHER_NUM_SLOTS — gather_mask second-dimension slot-structure constant
//   (interface [BLOCK_NUM, 20, N]): slots 0–11 hold the Y-filtered compact
//   data (means2d×2 / opacity / conics×3 / rect×4 / depths / gaussIndex —
//   12 lanes), slots 12–19 hold the bound-filtered compact data (8 lanes).
//   Slot base addresses are computed by the kernel as
//   `blockIndex · gaussNum · 20`; this is a compile-time structure constant,
//   NOT a TilingData field (TilingData.md §3).
// ---------------------------------------------------------------------------
constexpr uint32_t GATHER_NUM_SLOTS = 20;

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskTilingData — normal main template (tilingKey=0)
//
// Flat (b,c)×tile task-space multicore split + UB gaussian batching.
// Filled by the host tiling function when the empty-tensor condition does
// NOT hold (B > 0 && C > 0 && N > 0, BranchRoute.md §2), then dispatched to
// the normal kernel template via SetTilingKey(GET_TPL_TILING_KEY(0)).
//
// Fields (units in comments; element counts unless stated otherwise):
//   taskTotal        — flat task total = batchSize·cameraNum·numTile
//   taskNumPerScore  — short-core task count = taskTotal / blockDim (floor)
//   taskNumPerLcore  — long-core task count = taskNumPerScore + 1
//   numScore         — short-core count = blockDim − (taskTotal % blockDim)
//   numLcore         — long-core count = taskTotal % blockDim (remainder
//                      cores take the long-core size)
//   blockDim         — actual participating core count (shrinks when
//                      taskTotal < AIV core count, ≤ GetCoreNumAiv())
//   batchSize        — B (means2d.shape[0])
//   cameraNum        — C (means2d.shape[1])
//   numTile          — tile total (tile_grid.shape[0]); task-decode divisor:
//                      bcIdx = task / numTile, tIdx = task % numTile
//   gaussNum         — N: gaussian-axis length (consistent across inputs;
//                      gather_mask row length = gaussNum)
//   gaussNumPerLoop  — gaussians per core per batch = UB budget / per-gaussian
//                      slot byte cost, aligned up to 64
//   imageWidth       — W (pixels, Float attribute, passed through as fp32)
//   imageHeight      — H (pixels, Float attribute, passed through as fp32)
//   tileSize         — ts (Int attribute read then cast to fp32 for the
//                      floating-point boundary arithmetic)
//   ubTotalSize      — platform GetCoreMemSize(UB) in BYTES (A5 ≈ 248KB
//                      magnitude; kernel uses it for checks/logging only)
// ---------------------------------------------------------------------------
struct FlashGaussianBuildMaskTilingData {
    // ─── flat task-space multicore split (taskTotal = B·C·numTile,
    //      view-major continuous dispatch + long/short core balancing) ───
    uint32_t taskTotal;         // 扁平任务总数 = batchSize·cameraNum·numTile（元素数/任务数）
    uint32_t taskNumPerScore;   // 短核任务数 = taskTotal / blockDim（floor）
    uint32_t taskNumPerLcore;   // 长核任务数 = taskNumPerScore + 1
    uint32_t numScore;          // 短核数（核数）= blockDim − (taskTotal % blockDim)
    uint32_t numLcore;          // 长核数（核数）= taskTotal % blockDim（余数核取长核）
    uint32_t blockDim;          // 实际参与核数（核数；taskTotal < AIV 核数时收缩，≤ GetCoreNumAiv()）
    // ─── scale parameters (GM layout & task decode; element counts) ───
    uint32_t batchSize;         // B（means2d.shape[0]）
    uint32_t cameraNum;         // C（means2d.shape[1]）
    uint32_t numTile;           // tile 总数（tile_grid.shape[0]）；任务解码除数：bcIdx = task / numTile、tIdx = task % numTile
    uint32_t gaussNum;          // N：高斯轴长度（跨输入末维一致；gather_mask 行长 = gaussNum）
    // ─── UB batching (element count) ───
    uint32_t gaussNumPerLoop;   // 单核单批处理的高斯数（元素数）= UB 预算 / 单高斯槽位字节开销，向上对齐 64
    // ─── attributes (fp32 pass-through, pixel domain) ───
    float imageWidth;           // W（像素，Float 属性）
    float imageHeight;          // H（像素，Float 属性）
    float tileSize;             // ts（Int 属性读取后转 fp32 参与浮点边界运算）
    // ─── platform (check/log only) ───
    uint64_t ubTotalSize;       // platform GetCoreMemSize(UB) 字节（A5 = 248KB 量级，kernel 不参与计算）
};

// ---------------------------------------------------------------------------
// FlashGaussianBuildMaskEmptyTilingData — empty short-circuit template
// (tilingKey=1)
//
// tile_sum zero-fill task split. Filled by the host tiling function when the
// empty-tensor condition holds (B == 0 || C == 0 || N == 0, BranchRoute.md
// §2), then dispatched to the empty kernel template via
// SetTilingKey(GET_TPL_TILING_KEY(1)). Reduction-paradigm hard constraint:
// independent struct, never aliases the normal struct.
//
// Fields (element counts unless stated otherwise):
//   fillTotal       — tile_sum element total = batchSize·cameraNum·numTile
//                     (int32 elements; fillTotal = 0 when B·C = 0 → all
//                     cores early-exit)
//   fillNumPerScore — short-core fill element count = fillTotal / blockDim
//   fillNumPerLcore — long-core fill element count = fillNumPerScore + 1
//   numScore        — short-core count (cores)
//   numLcore        — long-core count (cores)
//   blockDim        — actual participating core count (cores; set to 1 when
//                     fillTotal = 0, the single in-ring core early-exits)
// ---------------------------------------------------------------------------
struct FlashGaussianBuildMaskEmptyTilingData {
    // ─── zero-fill task split (fillTotal = B·C·numTile int32 elements;
    //      B·C = 0 → fillTotal = 0, all cores early-exit) ───
    uint32_t fillTotal;         // tile_sum 元素总数 = batchSize·cameraNum·numTile（元素数）
    uint32_t fillNumPerScore;   // 短核填充元素数（元素数）= fillTotal / blockDim（floor）
    uint32_t fillNumPerLcore;   // 长核填充元素数（元素数）= fillNumPerScore + 1
    uint32_t numScore;          // 短核数（核数）
    uint32_t numLcore;          // 长核数（核数）
    uint32_t blockDim;          // 实际参与核数（核数；fillTotal=0 时置 1，唯一在环核空转早退）
};
