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
// kernels/calc_render_simt_bwd/op_host/arch35/calc_render_simt_bwd_tiling.cpp
// =============================================================================
//
// ROLE: Host-side TilingFunc implementation for CalcRenderSimtBwd on arch35
//   (Ascend 950).
//
//   Task 27 (公共 Tiling 实现) replaces the Task 25 stub with the real
//   formulas from docs/calc_render_simt_bwd/design/HostTiling.md §3-§9:
//     §3  shape 提取（tileCoords.shape[0]/[2]、gs.shape[0]，动态读取）+
//         判空防御（shape 指针空 / GetCoreMemSize(UB)==0 → GRAPH_FAILED）
//     §4  UB 预算（ubSize > DCACHE_SIZE=64KB 时 SetLocalMemorySize(ubSize−64KB)；
//         ubSize 字段仅记录 GetCoreMemSize(UB) 原始值，kernel 不消费）
//     §5  BlockDim 决策（aivNum = GetCoreNumAiv()，返回 0 防御性置 1；
//         tileNum==0 || nGauss==0（空 Tensor）→ BlockDim=1 空转兜底；
//         否则 = aivNum——核间任务分配数据驱动于 offsets 调度表，host 不重算）
//     §7  TilingData 6 字段赋值（TilingData.md §1 契约：前 3 字段 kernel 按偏移
//         0/1/2 × int64 裸解析，顺序不可调整）+ useTwoPass 恒 0
//     §9  整体结构（调用顺序 = 平台查询 → UB 预算 → shape 提取 → BlockDim →
//         填 TilingData → SetTilingKey(0)/SetBlockDim/workspace(0) → 序列化）
//
//   Branch decision (BranchRoute.md §1/§2): tilingKey 集 = {0}（单模板单分支），
//   无条件 SetTilingKey(0)；负向场景（ubSize==0 / shape 指针空）不路由、
//   直接 GRAPH_FAILED；空 Tensor 短路仅改 BlockDim 不改 key。
//
//   Registration per HostTiling.md §8:
//     IMPL_OP_OPTILING(CalcRenderSimtBwd).Tiling(TilingFunc) — static entry,
//     no TilingParse (本算子不注册 TilingParse：平台参数在 Tiling 阶段经
//     GetCoreMemSize/GetCoreNumAiv 直取，无 CompileInfo 缓存需求).
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : CalcRenderSimtBwd   — IMPL_OP_OPTILING, function suffixes
//   snake_case   : calc_render_simt_bwd — filename
// =============================================================================

#include "register/op_def_registry.h"                              // IMPL_OP_OPTILING
#include "exe_graph/runtime/tiling_context.h"                      // gert::TilingContext
#include "op_common/log/log.h"                                     // OP_LOGI / OP_CHECK_NULL_WITH_CONTEXT
#include "tiling/platform/platform_ascendc.h"                      // platform_ascendc::PlatformAscendC
#include "../../op_kernel/arch35/calc_render_simt_bwd_tiling_struct.h"     // CalcRenderSimtBwdTilingData（TilingData.md §1 契约）

namespace optiling {

// ---------------------------------------------------------------------------
// Constants（HostTiling.md §3/§4/§5）
// ---------------------------------------------------------------------------

// 输入索引（OpDef 声明序，proto.md §1：vColor/vDepth/lastCumsum/error/gs/
// tileCoords/offsets/gsIds/lastIds —— TilingFunc 读这 4 个 shape）.
constexpr size_t SHAPE_IDX_GS = 4U;          // gs         [N, 10]
constexpr size_t SHAPE_IDX_TILE_COORDS = 5U; // tileCoords [T, 2, P]
constexpr size_t SHAPE_IDX_OFFSETS = 6U;     // offsets    [S]
constexpr size_t SHAPE_IDX_GSIDS = 7U;       // gsIds      [G]

// DCache 预留（字节）：ubSize > DCACHE_SIZE 时可用 UB = ubSize − 64KB
// （Ascend950 实测 248KB → 184KB；禁止硬编码 192/248KB 兜底，HostTiling.md §4）.
constexpr uint64_t DCACHE_SIZE = 64ULL * 1024ULL;

// ---------------------------------------------------------------------------
// TilingFuncCalcRenderSimtBwd(context) — tiling entry point
//
// HostTiling.md §9 整体结构（顺序即实现顺序）：
//   平台动态查询（GetCoreMemSize(UB)==0 → 防御性 GRAPH_FAILED；
//   GetCoreNumAiv()==0 → 防御性置 1）
//   → UB 预算（SetLocalMemorySize(ubSize − 64KB)，仅 ubSize > 64KB 时调用）
//   → shape 提取（tileCoords/gs 指针判空 → nPixel/tileNum/nGauss 动态读取）
//   → BlockDim 决策（空 Tensor 短路 =1，否则 = aivNum）
//   → 填 TilingData 6 字段 + SetTilingKey(0)/SetBlockDim/workspace(0)
//   → SaveToBuffer 序列化 → GRAPH_SUCCESS.
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncCalcRenderSimtBwd(gert::TilingContext* context)
{
    // ===== 平台动态查询（禁止硬编码，HostTiling.md §3/§4）=====
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0U) {
        // 平台信息异常：禁止硬编码 192/248KB 兜底（HostTiling.md §3 第 2 步）
        OP_LOGE(context->GetNodeName(), "GetCoreMemSize(UB) returned 0, platform info abnormal");
        return ge::GRAPH_FAILED;
    }

    // GetCoreNumAiv：AIV 核数动态查询；返回 0 时防御性置 1（HostTiling.md §5）
    uint32_t aivNum = platform.GetCoreNumAiv();
    if (aivNum == 0U) {
        aivNum = 1U;
    }

    // ===== UB 预算（HostTiling.md §4：ubSize > 64KB 时下发可用 UB）=====
    if (ubSize > DCACHE_SIZE) {
        context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE));
    }

    // ===== shape 提取（HostTiling.md §3：动态读取、禁止假定常量）=====
    // tileCoords / gs / offsets / gsIds 是 tiling 全部 shape 事实来源，判空防御（null_input）
    const gert::StorageShape* coordsShape = context->GetInputShape(SHAPE_IDX_TILE_COORDS);
    const gert::StorageShape* gsShape = context->GetInputShape(SHAPE_IDX_GS);
    const gert::StorageShape* offsetsShape = context->GetInputShape(SHAPE_IDX_OFFSETS);
    const gert::StorageShape* gsIdsShape = context->GetInputShape(SHAPE_IDX_GSIDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, coordsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetsShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gsIdsShape);

    // Validate ranks before indexing dimensions (GetDim is not a rank check).
    if (coordsShape->GetStorageShape().GetDimNum() != 3U
        || gsShape->GetStorageShape().GetDimNum() != 2U) {
        OP_LOGE(context->GetNodeName(), "tileCoords/gs rank must be 3/2");
        return ge::GRAPH_FAILED;
    }
    const int64_t nPixel = coordsShape->GetStorageShape().GetDim(2); // P：每 tile 像素数（256/1024 两类）
    const int64_t tileNum = coordsShape->GetStorageShape().GetDim(0); // T：tile 数
    const int64_t nGauss = gsShape->GetStorageShape().GetDim(0);      // N：高斯数
    // S = offsets.shape[0] / G = gsIds.shape[0]：spec formula 几何语义（VN = max(S-2T,0)、
    // posC = clip(pos,0,G-1)）的 shape 事实，经 TilingData 下发 kernel（裸解析偏移 40/48）
    const int64_t offsetsLen = offsetsShape->GetStorageShape().GetDimNum() > 0
        ? offsetsShape->GetStorageShape().GetDim(0) : 0;
    const int64_t gsIdsLen = gsIdsShape->GetStorageShape().GetDimNum() > 0
        ? gsIdsShape->GetStorageShape().GetDim(0) : 0;

    // ===== 接口契约级负向拦截（Interface.md「约束与限制」/spec boundary）=====
    // (1) tileCoords 坐标轴维（dim1）恒为 2（[T,2,P] 像素 (x,y) 坐标契约）——
    //     EX-S7：dim1 != 2 属 shape_mismatch，不路由任何 tilingKey（BranchRoute.md §1）。
    if (coordsShape->GetStorageShape().GetDimNum() != 3U
        || coordsShape->GetStorageShape().GetDim(1) != 2) {
        OP_LOGE(context->GetNodeName(),
                 "tileCoords must be [T,2,P]: got dim1=%ld (rank %u)",
                 static_cast<long>(coordsShape->GetStorageShape().GetDimNum() > 1
                     ? coordsShape->GetStorageShape().GetDim(1) : -1),
                 static_cast<unsigned>(coordsShape->GetStorageShape().GetDimNum()));
        return ge::GRAPH_FAILED;
    }

    // (2) offsets/gsIds rank 恒为 1（proto.md §4：(S,)/(G,) 契约；spec boundary
    //     "offsets.shape == (S,)（rank 1）"、"gsIds.shape == (G,)，rank 恒为 1"）——
    //     EX-S8/EX-S9：rank != 1 属 shape_mismatch（ACL_ERROR_INVALID_DIMENSION），
    //     不路由任何 tilingKey。Task 30 r2：L2_exc_constraint_041（offsets [4,70]）
    //     曾被静默接受（GetDim(0)=4 当 S 用），FAIL 仅由 golden 崩溃掩盖。
    if (offsetsShape->GetStorageShape().GetDimNum() != 1U
        || gsIdsShape->GetStorageShape().GetDimNum() != 1U) {
        OP_LOGE(context->GetNodeName(),
                 "offsets/gsIds must be rank-1 [(S,)/(G,)]: got rank %u/%u",
                 static_cast<unsigned>(offsetsShape->GetStorageShape().GetDimNum()),
                 static_cast<unsigned>(gsIdsShape->GetStorageShape().GetDimNum()));
        return ge::GRAPH_FAILED;
    }

    // ===== 像素网格一致性（调用链层负向拦截，Interface.md「Broadcast 规则」/spec boundary
    // 「像素网格维冲突 shape_mismatch」）=====
    // 像素网格 (T,P) 跨 vColor/vDepth/lastCumsum/error/tileCoords/lastIds 一致为调用链契约；
    // 单算子 aclnn 链路中 InferShape 不被调度，负向冲突（BND-03）必须在本层（进入 kernel 前）
    // 拒绝——shape_mismatch 不路由任何 tilingKey（BranchRoute.md §1）。
    {
        auto gridMismatch = [tileNum, nPixel](const gert::StorageShape* s, bool rank3,
                                              int64_t dimT, int64_t dimP) {
            const auto& sh = s->GetStorageShape();
            if (rank3) {
                return sh.GetDimNum() != 3U || sh.GetDim(static_cast<size_t>(dimT)) != tileNum
                    || sh.GetDim(static_cast<size_t>(dimP)) != nPixel;
            }
            return sh.GetDimNum() != 2U || sh.GetDim(static_cast<size_t>(dimT)) != tileNum
                || sh.GetDim(static_cast<size_t>(dimP)) != nPixel;
        };
        constexpr size_t SHAPE_IDX_VCOLOR = 0U;      // vColor     [3,T,P]
        constexpr size_t SHAPE_IDX_VDEPTH = 1U;      // vDepth     [1,T,P]
        constexpr size_t SHAPE_IDX_LASTCUMSUM = 2U;  // lastCumsum [T,P]
        constexpr size_t SHAPE_IDX_ERROR = 3U;       // error      [T,P]
        constexpr size_t SHAPE_IDX_LASTIDS = 8U;     // lastIds    [T,P]
        const gert::StorageShape* vColorShape = context->GetInputShape(SHAPE_IDX_VCOLOR);
        const gert::StorageShape* vDepthShape = context->GetInputShape(SHAPE_IDX_VDEPTH);
        const gert::StorageShape* lastCumsumShape = context->GetInputShape(SHAPE_IDX_LASTCUMSUM);
        const gert::StorageShape* errorShape = context->GetInputShape(SHAPE_IDX_ERROR);
        const gert::StorageShape* lastIdsShape = context->GetInputShape(SHAPE_IDX_LASTIDS);
        OP_CHECK_NULL_WITH_CONTEXT(context, vColorShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, vDepthShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, lastCumsumShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, errorShape);
        OP_CHECK_NULL_WITH_CONTEXT(context, lastIdsShape);
        if (gridMismatch(vColorShape, true, 1, 2) || vColorShape->GetStorageShape().GetDim(0) != 3
            || gridMismatch(vDepthShape, true, 1, 2) || vDepthShape->GetStorageShape().GetDim(0) != 1
            || gridMismatch(lastCumsumShape, false, 0, 1)
            || gridMismatch(errorShape, false, 0, 1)
            || gridMismatch(lastIdsShape, false, 0, 1)
            || gsShape->GetStorageShape().GetDimNum() != 2U
            || gsShape->GetStorageShape().GetDim(1) != 10) {
            OP_LOGE(context->GetNodeName(),
                     "pixel-grid shape_mismatch: tileCoords [%ld,%ld] inconsistent with grid inputs",
                     static_cast<long>(tileNum), static_cast<long>(nPixel));
            return ge::GRAPH_FAILED;
        }
    }

    // ===== 输出 vGs host 预分配契约（InferShapeDtype.md：vGs.shape = gs.shape）=====
    // EX-S10：调用方传入的输出 shape 与 gs 不一致（host 预分配契约）→ SHAPE_NOT_MATCH，
    // 必须在进入 kernel 前拒绝（kernel 对 vGs 全量原子累加，shape 不一致即越界写）。
    {
        const gert::StorageShape* vGsShape = context->GetOutputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(context, vGsShape);
        if (vGsShape->GetStorageShape().GetDimNum() != 2U
            || vGsShape->GetStorageShape().GetDim(0) != nGauss
            || vGsShape->GetStorageShape().GetDim(1) != 10) {
            OP_LOGE(context->GetNodeName(),
                     "vGs shape must equal gs shape [%ld,10]: output shape mismatch",
                     static_cast<long>(nGauss));
            return ge::GRAPH_FAILED;
        }
    }

    // One VF owns at most 1024 pixels. Reject unsupported shapes rather
    // than silently omitting pixels beyond the UB capacity. A nonempty grid
    // requires at least one schedule bin; values follow the documented ABI.
    if (nPixel <= 0 || nPixel > 1024 || tileNum < 0 || nGauss < 0
        || (tileNum > 0 && offsetsLen <= 2 * tileNum)) {
        OP_LOGE(context->GetNodeName(), "invalid pixel count or schedule layout");
        return ge::GRAPH_FAILED;
    }

    // ===== 多核切分（HostTiling.md §5：仅核数决策）=====
    // 空 Tensor（T=0 或 N=0）→ BlockDim=1 空转兜底；正常路径 = aivNum
    // （核间任务分配数据驱动于 offsets 调度表，与 tileNum 无关）.
    const uint32_t blockDim = (tileNum == 0 || nGauss == 0) ? 1U : aivNum;

    // ===== 填 TilingData（HostTiling.md §7；字段序 = TilingData.md §1 契约）=====
    CalcRenderSimtBwdTilingData td;
    td.set_nPixel(nPixel);        // tileCoords.shape[2]（元素数）
    td.set_tileNum(tileNum);      // tileCoords.shape[0]（个数）
    td.set_nGauss(nGauss);        // gs.shape[0]（行数）
    td.set_ubSize(ubSize);        // GetCoreMemSize(UB) 原始值（字节；仅记录备查）
    td.set_blockDim(blockDim);    // 与 SetBlockDim 同值（仅记录）
    td.set_useTwoPass(0U);        // 预留恒 0：单遍结构，无 two-pass 变体
    td.set_offsetsLen(offsetsLen); // offsets.shape[0]：kernel 几何定位三段布局 VN = max(S-2T,0)
    td.set_gsIdsLen(gsIdsLen);     // gsIds.shape[0]：kernel posC = clip(pos, 0, G-1) 裁剪上界

    OP_LOGI(context->GetNodeName(),
            "tiling: nPixel=%ld, tileNum=%ld, nGauss=%ld, offsetsLen=%ld, gsIdsLen=%ld, ubSize=%lu, blockDim=%u",
            static_cast<long>(nPixel), static_cast<long>(tileNum), static_cast<long>(nGauss),
            static_cast<long>(offsetsLen), static_cast<long>(gsIdsLen),
            static_cast<unsigned long>(ubSize), blockDim);

    // ===== 下发框架调度参数（HostTiling.md §7）=====
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);     // 显式单值 key=0（BranchRoute.md §1：单分支）
    // TOPK-13（Task 36 F5）：kernel 入口使用全核同步 SyncAll<true>（vGs 协作清零
    // 序言，calc_render_simt_bwd_apt.cpp），多流场景下 tiling 必须下发
    // BATCH_MODE_SCHEDULE 调度模式，否则 SyncAll 可能卡死。
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    // 输出清零：vGs 由 kernel 入口的协作清零序言统一清零（ZeroVGs + SyncAll，
    // DESIGN-BRANCH-0 §5.3）——不声明 SetNeedAtomic/Output.InitValue 运行时
    // atomic clean：实测 aclnn 运行时的 clean 以陈旧尺寸（前一执行输出大小）
    // memset，越过当前更小的输出缓冲区时产生越界写（间歇 aivec err95 MTE
    // 非法 GM 地址，Task 30）；kernel 自清零语义上等价 spec 的
    // vGs = 0 + Σ贡献（golden np.zeros + add.at），并与 Interface.md 的
    // 调用方预清零契约兼容（幂等）。
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = 0;        // 无 workspace（HostTiling.md §6 N/A 论证）

    // ===== 序列化下发（SaveToBuffer 按注册结构体布局写入 raw buffer）=====
    // SaveToBuffer 只写裸字节，不回写 RawTilingData 的数据长度；必须显式
    // SetDataSize 发布长度，否则框架侧（GE/TTK）按 GetDataSize()==0 捕获到
    // 空 tiling 数据，kernel 拿到空 tiling buffer（基线 calc_render_bwd_var_
    // clip_gsids 及本仓全部同构算子的既有两行式，HostTiling.md §7 序列化契约）。
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData->GetData());
    td.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(td.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(CalcRenderSimtBwd) — register tiling function with CANN
//   (HostTiling.md §8: no .TilingParse — 平台参数 Tiling 阶段直取)
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(CalcRenderSimtBwd).Tiling(TilingFuncCalcRenderSimtBwd);

} // namespace optiling
