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
// flash_gaussian_build_mask_package/op_kernel/arch35/flash_gaussian_build_mask_struct.h
// =============================================================================
//
// ROLE: Ascend C template parameter (TPL) declarations for FlashGaussianBuildMask.
//   Per docs/flash_gaussian_build_mask/design/TilingKey.md §1 the tiling key
//   carries a SINGLE BOOL template parameter `isEmptyTensor` (1 bit):
//     - isEmptyTensor=0 → normal main template (tilingKey=0, sub-kernel …_0_mix_aiv):
//       Y pre-filter → bounding-box coarse filter → center/ellipse fine filter
//       → GatherMask ordered compression
//     - isEmptyTensor=1 → empty short-circuit template (tilingKey=1, sub-kernel
//       …_1_mix_aiv): tile_sum zero-fill + empty-list direct output
//
//   Host-side tiling (op_host/arch35/flash_gaussian_build_mask_tiling_arch35.cpp)
//   includes this header and produces the branch key via
//   GET_TPL_TILING_KEY(0)/(1) — BOOL params occupy 1 bit by value, so the
//   encoded keys are exactly 0 (normal) / 1 (empty), matching
//   BranchRoute.md §1 routing table and BranchCatalog.md.
//
//   The ASCENDC_TPL system works as follows:
//   1. ASCENDC_TPL_ARGS_DECL — declares the template argument list: one BOOL
//      parameter `isEmptyTensor` selectable from {0, 1}
//   2. ASCENDC_TPL_SEL — enumerates the template specialization dispatch: one
//      ARGS_SEL block per BOOL value (2 blocks cover the full domain, no gaps)
//   3. GET_TPL_TILING_KEY — used by host-side tiling to encode the key that
//      tells the runtime which kernel binary (sub-kernel instance) to select
//
//   NOTE (Task 27): this file previously carried the scale_custom-lineage
//   RANK UINT declaration {RANK_4, RANK_8} from the interface-stub phase; it
//   is replaced by the designed single-BOOL declaration above. No compiled
//   translation unit consumed the old RANK declaration (op_kernel/
//   flash_gaussian_build_mask_apt.cpp no longer includes this header), so the
//   replacement is build-neutral for the kernel chain.
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — first argument to ASCENDC_TPL_ARGS_DECL
//   snake_case   : flash_gaussian_build_mask  — filename
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — header guard
//
// NAME REPLACEMENT RULES (to create FooBar operator):
//   FlashGaussianBuildMask → FooBar                (ASCENDC_TPL_ARGS_DECL argument)
//   FLASH_GAUSSIAN_BUILD_MASK → FOO_BAR             (header guard)
//   flash_gaussian_build_mask_struct → foo_bar_struct (filename)
//   isEmptyTensor: keep (Reduction-paradigm BOOL template parameter name)
//   ASCENDC_TPL macros: keep (CANN framework macros)
//
// =============================================================================

#ifndef FLASH_GAUSSIAN_BUILD_MASK_STRUCT_H_
#define FLASH_GAUSSIAN_BUILD_MASK_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"  // ASCENDC_TPL macros

// ---------------------------------------------------------------------------
// ASCENDC_TPL_ARGS_DECL — declares the compile-time template argument list
//   (docs/flash_gaussian_build_mask/design/TilingKey.md §1)
//
// Arguments:
//   FlashGaussianBuildMask  → operator type name
//   ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1)
//     → declares a BOOL template parameter "isEmptyTensor" occupying 1 bit
//     → allowed values: 0 (normal main template) and 1 (empty short-circuit)
//
// To create FooBar:
//   ASCENDC_TPL_ARGS_DECL(FooBar,
//       ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1)
//   );
// ---------------------------------------------------------------------------
ASCENDC_TPL_ARGS_DECL(FlashGaussianBuildMask,
    ASCENDC_TPL_BOOL_DECL(isEmptyTensor, 0, 1)
);

// ---------------------------------------------------------------------------
// ASCENDC_TPL_SEL — template specialization selectors
//
// Each ASCENDC_TPL_ARGS_SEL block creates a specialization for one combination
// of template parameter values; the kernel compiler generates a separate
// sub-kernel binary per block (…_0_mix_aiv / …_1_mix_aiv).
//
// Block 1: ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0) → normal main template
// Block 2: ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1) → empty short-circuit template
//
// The two blocks cover the full BOOL domain {0, 1} — no empty slots.
//
// To create FooBar:
//   ASCENDC_TPL_SEL(
//       ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
//       ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1))
//   );
// ---------------------------------------------------------------------------
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(isEmptyTensor, 1))
);

#endif
