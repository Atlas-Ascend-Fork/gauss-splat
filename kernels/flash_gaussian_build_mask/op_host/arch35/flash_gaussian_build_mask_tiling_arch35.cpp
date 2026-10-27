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
// flash_gaussian_build_mask_package/op_host/arch35/flash_gaussian_build_mask_tiling_arch35.cpp
// =============================================================================
//
// ROLE: Host-side tiling implementation for FlashGaussianBuildMask on arch35
//   (Ascend 950 / DAV_3510). Implements the common tiling pipeline of
//   docs/flash_gaussian_build_mask/design/HostTiling.md §3–§7 with the
//   dual-path routing of docs/flash_gaussian_build_mask/design/BranchRoute.md:
//
//     TilingForFlashGaussianBuildMask
//       ① context / 8-input null checks (null_input)
//       ② platform params via GetPlatformInfo → PlatformAscendC:
//          aivCoreNum = GetCoreNumAiv(), ubTotalSize = GetCoreMemSize(UB),
//          systemWorkspaceSize = GetLibApiWorkSpaceSize()  (non-zero checks)
//       ③ one metadata pass (rank/dtype/format), combined shape checks
//          and attribute/range validation (any failure → GRAPH_FAILED,
//          no kernel launch)
//       ④ empty-tensor routing (shape-driven only):
//          B==0 || C==0 || N==0 → empty short-circuit template:
//              tilingKey = GET_TPL_TILING_KEY(1) = 1, fill
//              FlashGaussianBuildMaskEmptyTilingData (tile_sum zero-fill
//              long/short core split, fillTotal==0 → blockDim=1, never 0)
//          otherwise → normal main template:
//              tilingKey = GET_TPL_TILING_KEY(0) = 0, fill
//              FlashGaussianBuildMaskTilingData (flat (b,c)×tile task-space
//              long/short core split + UB gaussian batching)
//       ⑤ SetBlockDim + SetTilingKey, workspace ws[0] = system workspace only
//
//   Registration: OpDef-class form (HostTiling.md §8) — the tiling entry is
//   hooked via AICore().SetTiling(optiling::TilingForFlashGaussianBuildMask)
//   in op_host/flash_gaussian_build_mask_def.cpp; under OP_TILING_LIB the
//   OP_ADD registration publishes it into the gert implementation registry.
//   IMPL_OP_OPTILING / TilingParse are N/A for this operator (no compile-info
//   cache fields; platform params are queried live in step ②).
//
// FIELD UNITS (TilingData.md): split/scale fields are ELEMENT COUNTS;
// numScore/numLcore/blockDim are CORE COUNTS; ubTotalSize is BYTES;
// imageWidth/imageHeight/tileSize are pixel-domain floats.
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : FlashGaussianBuildMask   — function suffix, struct prefix
//   snake_case   : flash_gaussian_build_mask  — filename, namespace
//   UPPER_SNAKE  : FLASH_GAUSSIAN_BUILD_MASK  — GET_TPL_TILING_KEY macro
//
// =============================================================================

#include <cmath>
#include <cstdint>

#include "register/op_def_registry.h"                              // OP_ADD / OpDef registry
#include "op_common/log/log.h"                                     // OP_LOGE, OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/platform_util.h"                  // PlatformAscendC, fe::PlatFormInfos, gert::TilingContext
#include "../../op_kernel/arch35/flash_gaussian_build_mask_tiling_struct.h"     // FlashGaussianBuildMaskTilingData / FlashGaussianBuildMaskEmptyTilingData, GATHER_NUM_SLOTS
#include "../../op_kernel/arch35/flash_gaussian_build_mask_struct.h"             // ASCENDC_TPL BOOL isEmptyTensor declaration (GET_TPL_TILING_KEY)
#include "flash_gaussian_build_mask_tiling_arch35.h"                            // FlashGaussianBuildMaskCompileInfo

namespace optiling {

namespace {

// ---------------------------------------------------------------------------
// Constants — hand-copied from the design docs (do not "fix" by memory).
// ---------------------------------------------------------------------------

// HostTiling.md §4 (UB batching, normal branch):
constexpr float UB_RATIO = 0.85F;          // UB budget ratio: absorbs the 5 uint8 compare lanes
constexpr uint32_t TASK_BUFFER_NUM = 20;   // P value: fp32-equivalent lanes per gaussian (§1.2)
constexpr int32_t FLOAT_SIZE = 4;          // fp32 bytes
constexpr uint32_t ALIGN_VALUE = 64;       // 64 elements = 256B (8×32B datablock) copy alignment

// Interface.md §82-83 / §86-87 (rank / channel / value-range / attr locks):
constexpr int64_t GAUSS_NUM_MAX = 16777215;      // N ≤ 2^24−1 (gauss_index fp32 encoding bound)
constexpr int64_t GAUSS_NUM_MIN = 0;             // N ≥ 0 (empty N=0 routes to the empty branch)
constexpr int64_t NUM_TILE_MIN = 1;              // NumTile ∈ [1, 65536] (0/empty tile grid illegal)
constexpr int64_t NUM_TILE_MAX = 65536;
constexpr int64_t VIEW_DIM_MIN = 0;              // B, C ≥ 0 (B=0/C=0 route to the empty branch)
constexpr double IMAGE_SIZE_MIN = 1.0;           // image_width / image_height ∈ [1.0, 4294967295.0]
constexpr double IMAGE_SIZE_MAX = 4294967295.0;
constexpr int64_t TILE_SIZE_MIN = 1;             // tile_size ∈ [1, 2^63−1] (positive integer)

// Interface.md §82 input table (OpDef registration order). 8 inputs / 3
// outputs; the tiling function validates the input side only (see ② below).
constexpr size_t INPUT_NUM = 8U;

// Input indices (OpDef order): means2d, opacity, conics, covars2d, depths,
// cnt, tile_grid, gather_mask.
constexpr size_t IDX_MEANS2D = 0U;
constexpr size_t IDX_OPACITY = 1U;
constexpr size_t IDX_CONICS = 2U;
constexpr size_t IDX_COVARS2D = 3U;
constexpr size_t IDX_DEPTHS = 4U;
constexpr size_t IDX_CNT = 5U;
constexpr size_t IDX_TILE_GRID = 6U;
constexpr size_t IDX_GATHER_MASK = 7U;

// Rank locks: means2d/opacity/conics/covars2d/depths = 4, cnt = 2 or 3 (trailing singleton),
// tile_grid = 2, gather_mask = 3 (Interface.md §82; interface layout locked).
constexpr size_t INPUT_RANKS[INPUT_NUM] = {4U, 4U, 4U, 4U, 4U, 2U, 2U, 3U};

// ① dtype support set — single combination (Interface.md §算子规格):
// 7 float inputs + int32 cnt.
const ge::DataType INPUT_DTYPES[INPUT_NUM] = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                              ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT};

// Input names (log anchors, Interface.md §82 order).
const char *const INPUT_NAMES[INPUT_NUM] = {"means2d", "opacity",   "conics",  "covars2d",
                                            "depths",   "cnt",       "tile_grid", "gather_mask"};

// TilingKey.md §1: single BOOL template parameter isEmptyTensor occupies
// 1 bit by value → GET_TPL_TILING_KEY(0) = 0 (normal main template),
// GET_TPL_TILING_KEY(1) = 1 (empty short-circuit template). These are the
// TPL selector VALUES (not pre-encoded keys) passed to GET_TPL_TILING_KEY.
constexpr uint64_t TPL_SEL_NORMAL = 0U;   // isEmptyTensor = 0
constexpr uint64_t TPL_SEL_EMPTY = 1U;    // isEmptyTensor = 1

// ---------------------------------------------------------------------------
// CeilAttr(a, b) — ceil(a / b) for a Float attr over a positive Int attr
// (HostTiling.md §3.3 ⑤: numTileByAttr = ceil(H/ts) * ceil(W/ts)).
// Only called after the ④ attr-range check has ruled out ts < 1, so b > 0.
// ---------------------------------------------------------------------------
inline int64_t CeilAttr(const float a, const int64_t b)
{
    return static_cast<int64_t>(std::ceil(static_cast<double>(a) / static_cast<double>(b)));
}

// ---------------------------------------------------------------------------
// ComputeGaussNumPerLoop — UB gaussian batching (HostTiling.md §4).
//
//   gaussNumPerLoop = AlignUp(⌊ubTotalSize × 0.85 / (20 × 4B)⌋, 64)
//
// Byte-based dimensional analysis (single-gaussian slot cost =
// TASK_BUFFER_NUM × FLOAT_SIZE = 80 B). Float arithmetic exactly as the
// design snippet: (uint64 × float) / int → float, truncate, align up 64.
// A5 248KB magnitude: ⌊253952×0.85/80⌋ = 2698 → 2752;
// A2 192KB magnitude: ⌊196608×0.85/80⌋ = 2088 → 2112.
// Depends only on the UB budget (pure platform constant), never on gaussNum.
// ---------------------------------------------------------------------------
inline uint32_t ComputeGaussNumPerLoop(const uint64_t ubTotalSize)
{
    uint32_t gaussNumPerLoop =
        static_cast<uint32_t>((ubTotalSize * UB_RATIO) / (TASK_BUFFER_NUM * FLOAT_SIZE));
    gaussNumPerLoop = (gaussNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE;
    return gaussNumPerLoop;
}

// ---------------------------------------------------------------------------
// ComputeTaskSplit — normal flat (b,c)×tile task-space multicore split
// (HostTiling.md §5.1): view-major continuous dispatch + long/short core
// balancing + core-count shrink.
//
// Kernel consumption (TilingData.md): blockIdx < numLcore is a long core with
// segment base blockIdx×taskNumPerLcore; short-core base =
// numLcore×taskNumPerLcore + (blockIdx−numLcore)×taskNumPerScore; task decode
// (view-major): bcIdx = task / numTile, tIdx = task % numTile.
// ---------------------------------------------------------------------------
inline void ComputeTaskSplit(const uint32_t batchSize, const uint32_t cameraNum, const uint32_t numTile,
                              const uint32_t availCores, FlashGaussianBuildMaskTilingData *td)
{
    // uint64 intermediate against overflow (B·C·numTile stays within uint32
    // for allocatable outputs, TilingData.md field-width note).
    td->taskTotal = static_cast<uint32_t>(static_cast<uint64_t>(batchSize) * cameraNum * numTile);
    // availCores = min(AIV 核数, gather_mask BLOCK_NUM)（调用侧工作区行数上界，
    // 见 ⑤ gather 行配给）；taskTotal < availCores 时收缩。
    uint32_t blockDim = availCores;                       // dynamic core count, never hardcoded
    td->taskNumPerScore = td->taskTotal / blockDim;    // short-core task count (floor)
    td->taskNumPerLcore = td->taskNumPerScore + 1;     // long-core task count
    td->numLcore = td->taskTotal % blockDim;           // long-core count (remainder cores)
    if (td->taskNumPerScore == 0) {                    // taskTotal < core count: shrink
        blockDim = td->numLcore;                       // all in-ring cores are long cores
    }
    td->numScore = blockDim - td->numLcore;            // short-core count (0 after shrink)
    td->blockDim = blockDim;                           // numScore + numLcore == blockDim always
}

// ---------------------------------------------------------------------------
// ComputeEmptyFillSplit — empty branch tile_sum zero-fill split
// (HostTiling.md §5.2): fillTotal = B·C·numTile int32 elements, element-level
// long/short core balance; fillTotal == 0 (B=0 or C=0: all three outputs have
// 0 elements) → all split fields 0 and blockDim = 1 (NEVER 0; the single
// in-ring core early-exits).
// ---------------------------------------------------------------------------
inline void ComputeEmptyFillSplit(const uint32_t batchSize, const uint32_t cameraNum, const uint32_t numTile,
                                  const uint32_t aivCoreNum, FlashGaussianBuildMaskEmptyTilingData *td)
{
    td->fillTotal = static_cast<uint32_t>(static_cast<uint64_t>(batchSize) * cameraNum * numTile);
    if (td->fillTotal == 0) {
        // B=0 or C=0: three outputs all have 0 elements.
        td->fillNumPerScore = 0;
        td->fillNumPerLcore = 0;
        td->numLcore = 0;
        td->numScore = 0;
        td->blockDim = 1;                              // ⛔ SetBlockDim(0) forbidden
        return;
    }
    uint32_t blockDim = aivCoreNum;
    td->fillNumPerScore = td->fillTotal / blockDim;    // short-core fill count (floor)
    td->fillNumPerLcore = td->fillNumPerScore + 1;     // long-core fill count
    td->numLcore = td->fillTotal % blockDim;           // long-core count (remainder cores)
    if (td->fillNumPerScore == 0) {                    // fillTotal < core count: shrink
        blockDim = td->numLcore;                       // same shrink semantics as §5.1
    }
    td->numScore = blockDim - td->numLcore;            // short-core count (0 after shrink)
    td->blockDim = blockDim;
}

} // namespace

// ---------------------------------------------------------------------------
// TilingForFlashGaussianBuildMask(context) — tiling entry point
//
// Registered through the OpDef (op_host/flash_gaussian_build_mask_def.cpp,
// AICore().SetTiling(optiling::TilingForFlashGaussianBuildMask)); OP_ADD
// under OP_TILING_LIB publishes it into the gert implementation registry —
// no IMPL_OP_OPTILING block (HostTiling.md §8: OpDef-class registration
// form; no TilingParse compile-info cache for this operator).
//
// Pipeline (HostTiling.md §9): null checks → platform → validation
// (dtype → format → dims → attr → shape) → empty routing → branch fill +
// SetBlockDim/SetTilingKey → workspace.
//
// Returns: ge::GRAPH_SUCCESS, or ge::GRAPH_FAILED on any null input / null or
// zero platform param / validation failure (no kernel launch on failure).
// ---------------------------------------------------------------------------
ge::graphStatus TilingForFlashGaussianBuildMask(gert::TilingContext *context)
{
    // ===== ① context + 8-input null checks (null_input) =====
    if (context == nullptr) {
        OP_LOGE("FlashGaussianBuildMask", "context is nullptr (null_input)");
        return ge::GRAPH_FAILED;
    }
    // One metadata pass; no tensor data access or successful-launch logging.
    const gert::StorageShape *inShapes[INPUT_NUM] = {};
    for (size_t i = 0; i < INPUT_NUM; ++i) {
        inShapes[i] = context->GetInputShape(i);
        const auto *desc = context->GetInputDesc(i);
        if (inShapes[i] == nullptr || desc == nullptr) {
            OP_LOGE("FlashGaussianBuildMask", "input %s metadata is null", INPUT_NAMES[i]);
            return ge::GRAPH_FAILED;
        }
        const size_t rank = inShapes[i]->GetStorageShape().GetDimNum();
        const bool validRank = i == IDX_CNT ? (rank == 2U || rank == 3U) : rank == INPUT_RANKS[i];
        if (!validRank || desc->GetDataType() != INPUT_DTYPES[i] ||
            desc->GetStorageFormat() != ge::FORMAT_ND) {
            OP_LOGE("FlashGaussianBuildMask", "input %s rank/dtype/format mismatch", INPUT_NAMES[i]);
            return ge::GRAPH_FAILED;
        }
        if (i == IDX_CNT && rank == 3U && inShapes[i]->GetStorageShape().GetDim(2) != 1) {
            OP_LOGE("FlashGaussianBuildMask", "rank3 cnt must have shape [B,C,1]");
            return ge::GRAPH_FAILED;
        }
    }

    // ===== ② platform params (non-zero checks must not be skipped) =====
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubTotalSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotalSize);   // A5 ≈ 248KB magnitude
    const uint32_t aivCoreNum = platform.GetCoreNumAiv();                     // A5: 56/64, never hardcoded
    if (aivCoreNum == 0 || ubTotalSize == 0) {
        OP_LOGE("FlashGaussianBuildMask", "invalid platform params: aivCoreNum=%u ubTotalSize=%lu",
                aivCoreNum, static_cast<unsigned long>(ubTotalSize));
        return ge::GRAPH_FAILED;
    }
    // Workspace: system workspace only (§6, both branches identical).
    const size_t systemWorkspaceSize = platform.GetLibApiWorkSpaceSize();

    // Scale dims parsed from opacity (HostTiling.md §9: B, C = opacity.shape[0..1],
    // N = opacity.shape[3]; numTile = tile_grid.shape[0]) — safe after rank locks.
    const gert::Shape &opacityShape = inShapes[IDX_OPACITY]->GetStorageShape();
    const int64_t batch = opacityShape.GetDim(0);
    const int64_t camera = opacityShape.GetDim(1);
    const int64_t gauss = opacityShape.GetDim(3);
    const int64_t numTile = inShapes[IDX_TILE_GRID]->GetStorageShape().GetDim(0);
    const int64_t gatherDim0 = inShapes[IDX_GATHER_MASK]->GetStorageShape().GetDim(0);

    constexpr int64_t channels[] = {2, 1, 3, 3, 1};
    for (size_t i = IDX_MEANS2D; i <= IDX_DEPTHS; ++i) {
        const auto &shape = inShapes[i]->GetStorageShape();
        if (shape.GetDim(0) != batch || shape.GetDim(1) != camera ||
            shape.GetDim(2) != channels[i] || shape.GetDim(3) != gauss) {
            OP_LOGE("FlashGaussianBuildMask", "input %s shape mismatch", INPUT_NAMES[i]);
            return ge::GRAPH_FAILED;
        }
    }
    const auto &cntShape = inShapes[IDX_CNT]->GetStorageShape();
    const auto &gatherShape = inShapes[IDX_GATHER_MASK]->GetStorageShape();
    if (cntShape.GetDim(0) != batch || cntShape.GetDim(1) != camera ||
        inShapes[IDX_TILE_GRID]->GetStorageShape().GetDim(1) != 2 ||
        gatherShape.GetDim(1) != GATHER_NUM_SLOTS || gatherShape.GetDim(2) != gauss) {
        OP_LOGE("FlashGaussianBuildMask", "cnt/tile_grid/gather_mask shape mismatch");
        return ge::GRAPH_FAILED;
    }

    // ③ single-axis value ranges: N ∈ [0, 2^24−1], NumTile ∈ [1, 65536],
    // B/C ≥ 0 (Interface.md §83).
    if (gauss < GAUSS_NUM_MIN || gauss > GAUSS_NUM_MAX || numTile < NUM_TILE_MIN || numTile > NUM_TILE_MAX ||
        batch < VIEW_DIM_MIN || camera < VIEW_DIM_MIN) {
        OP_LOGE("FlashGaussianBuildMask",
                "axis value out of range: B=%ld C=%ld N=%ld NumTile=%ld (shape_mismatch)", batch, camera, gauss,
                numTile);
        return ge::GRAPH_FAILED;
    }

    if (batch > UINT32_MAX || camera > UINT32_MAX ||
        (camera != 0 && numTile != 0 && static_cast<uint64_t>(batch) > UINT32_MAX / static_cast<uint64_t>(camera) / numTile)) {
        OP_LOGE("FlashGaussianBuildMask", "task count exceeds uint32 range");
        return ge::GRAPH_FAILED;
    }

    // ④ attr value ranges (read first, checked before the ceil division):
    // image_width / image_height Float attrs via GetAttrPointer<float>,
    // tile_size Int attr via GetAttrPointer<int32_t> (Nnopbase 投递的 Int 为 32 位，
    // HostTiling.md §3.3 ④).
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE("FlashGaussianBuildMask", "attrs unavailable (null_input)");
        return ge::GRAPH_FAILED;
    }
    const float *imageWidthPtr = attrs->GetAttrPointer<float>(0);    // image_width (Float)
    const float *imageHeightPtr = attrs->GetAttrPointer<float>(1);   // image_height (Float)
    const int32_t *tileSizePtr = attrs->GetAttrPointer<int32_t>(2);  // tile_size (Int)
    if (imageWidthPtr == nullptr || imageHeightPtr == nullptr || tileSizePtr == nullptr) {
        OP_LOGE("FlashGaussianBuildMask", "attr image_width/image_height/tile_size missing (null_input)");
        return ge::GRAPH_FAILED;
    }
    const float imageWidth = *imageWidthPtr;
    const float imageHeight = *imageHeightPtr;
    const int64_t tileSize = static_cast<int64_t>(*tileSizePtr);
    if (!std::isfinite(imageWidth) || !std::isfinite(imageHeight) ||
        static_cast<double>(imageWidth) < IMAGE_SIZE_MIN || static_cast<double>(imageWidth) > IMAGE_SIZE_MAX ||
        static_cast<double>(imageHeight) < IMAGE_SIZE_MIN || static_cast<double>(imageHeight) > IMAGE_SIZE_MAX ||
        tileSize < TILE_SIZE_MIN) {
        OP_LOGE("FlashGaussianBuildMask",
                "attr out of range: image_width=%f image_height=%f tile_size=%ld (attribute_value_out_of_range)",
                imageWidth, imageHeight, tileSize);
        return ge::GRAPH_FAILED;
    }

    const int64_t rows = CeilAttr(imageHeight, tileSize);
    const int64_t cols = CeilAttr(imageWidth, tileSize);
    const int64_t numTileByAttr = rows <= NUM_TILE_MAX && cols <= NUM_TILE_MAX ? rows * cols : -1;
    if (numTile != numTileByAttr) {
        OP_LOGE("FlashGaussianBuildMask", "tile_grid rows %ld != ceil(H/ts)*ceil(W/ts) = %ld (shape_mismatch)",
                numTile, numTileByAttr);
        return ge::GRAPH_FAILED;
    }
    // ⑤ gather_mask 行配给（branch-aware，DESIGN-BRANCH-0.md §8「每核独占一行」）：
    //   - empty 分支（B/C/N 任一为 0）：empty kernel 仅 tile_sum 零填充、不绑定
    //     gather_mask，无 dim0 约束（校验置于路由之前的历史实现误拒空分支用例）；
    //   - normal 分支：kernel 仅寻址行 [0, blockDim)（blockIndex 独占行），blockDim
    //     上界收缩至 min(aivCoreNum, gatherDim0) 即满足安全性——spec notes「BLOCK_NUM
    //     为调用侧按平台适配的工作区行数（A2=48，A5=56/64）」，A2 尺寸 48 行的
    //     gather_mask 在 A5 56 核上合法（收缩后每核仍独占一行）；gatherDim0 == 0
    //     时无行可配（blockDim 不可为 0）→ 拒绝。
    const bool isEmptyTensor = (batch == 0) || (camera == 0) || (gauss == 0);
    uint32_t maxCores = aivCoreNum;
    if (!isEmptyTensor) {
        if (gatherDim0 < 1) {
            OP_LOGE("FlashGaussianBuildMask", "gather_mask dim0 %ld < 1 with non-empty task space (shape_mismatch)",
                    gatherDim0);
            return ge::GRAPH_FAILED;
        }
        maxCores = (static_cast<int64_t>(aivCoreNum) < gatherDim0)
                       ? aivCoreNum
                       : static_cast<uint32_t>(gatherDim0);
    }

    // ===== ④ empty-tensor routing (shape-driven only; empty first, normal
    // fallback — complementary and exclusive, BranchRoute.md §2) =====
    if (isEmptyTensor) {
        // empty short-circuit template: tile_sum zero-fill split (§5.2).
        auto *tiling = context->GetTilingData<FlashGaussianBuildMaskEmptyTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        ComputeEmptyFillSplit(static_cast<uint32_t>(batch), static_cast<uint32_t>(camera),
                              static_cast<uint32_t>(numTile), aivCoreNum, tiling);
        context->SetBlockDim(tiling->blockDim);              // fillTotal=0 → 1, never 0
        context->SetTilingKey(GET_TPL_TILING_KEY(TPL_SEL_EMPTY));   // = 1 → …_1_mix_aiv
    } else {
        // normal main template: UB batching (§4) + flat task split (§5.1).
        auto *tiling = context->GetTilingData<FlashGaussianBuildMaskTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        ComputeTaskSplit(static_cast<uint32_t>(batch), static_cast<uint32_t>(camera),
                         static_cast<uint32_t>(numTile), maxCores, tiling);
        tiling->batchSize = static_cast<uint32_t>(batch);
        tiling->cameraNum = static_cast<uint32_t>(camera);
        tiling->numTile = static_cast<uint32_t>(numTile);
        tiling->gaussNum = static_cast<uint32_t>(gauss);
        tiling->gaussNumPerLoop = ComputeGaussNumPerLoop(ubTotalSize);
        tiling->imageWidth = imageWidth;                     // Float attr passthrough (fp32)
        tiling->imageHeight = imageHeight;                   // Float attr passthrough (fp32)
        tiling->tileSize = static_cast<float>(tileSize);     // Int attr → fp32
        tiling->ubTotalSize = ubTotalSize;
        context->SetBlockDim(tiling->blockDim);              // shrunk dynamic core count
        context->SetTilingKey(GET_TPL_TILING_KEY(TPL_SEL_NORMAL)); // = 0 → …_0_mix_aiv
    }

    // ===== ⑤ workspace (both branches): ws[0] = system workspace size only —
    // no user workspace; must be set explicitly even though it is system-only
    // (HostTiling.md §6 / §7.5.4). =====
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
