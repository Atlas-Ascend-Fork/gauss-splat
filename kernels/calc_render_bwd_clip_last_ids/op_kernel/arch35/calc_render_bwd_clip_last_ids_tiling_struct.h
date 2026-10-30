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
// kernels/calc_render_bwd_clip_last_ids/op_kernel/arch35/calc_render_bwd_clip_last_ids_tiling_struct.h
// =============================================================================
//
// ROLE: TilingData struct shared between host-side tiling
//   (op_host/arch35/calc_render_bwd_clip_last_ids_tiling.cpp) and device-side
//   kernel (op_kernel/arch35/calc_render_bwd_clip_last_ids.cpp) for CalcRenderBwdClipLastids on
//   arch35 (Ascend 950).
//
//   Source of truth: docs/calc_render_bwd_clip_last_ids/design/TilingData.md §1
//   (fields and their order are part of the contract — 不增不删、顺序不可调整:
//    kernel 入口按偏移 0/1/2 × int64 裸解析前 3 字段 nPixel/tileNum/nGauss,
//    按偏移 24/32 × int64 裸解析 offsetsLen/gsIdsLen（spec formula 几何语义
//    VN = max(SN-2*TN,0) 与 posC = clip(pos,0,G-1) 所需的 shape 事实，置于
//    前 40 字节可靠区）；ubSize/blockDim/useTwoPass 随 SaveToBuffer 一并
//    序列化但 kernel 不读取).
//
//   Single template / single branch: tilingKey 唯一 key=0
//   (docs/calc_render_bwd_clip_last_ids/design/TilingKey.md), no ASCENDC_TPL args.
//
// HOST/KERNEL DUAL DEFINITION (same name, same namespace, same serialized layout):
//   - host (default): BEGIN_TILING_DATA_DEF 宏结构体 + REGISTER_TILING_DATA_CLASS
//     (register/tilingdata_base.h, 与基线同式; 序列化布局 = 字段声明序 + 自然对齐).
//     注意宏类对象内存布局 ≠ 序列化布局（data_ptr_ 打包序列化），宿主侧须经
//     set_xxx / SaveToBuffer 读写。
//   - device (__CCE__/__CCE_AICORE__, bisheng -x cce): 纯 POD 镜像结构体。
//     设备编译不能包含 register/tilingdata_base.h（其 graph/*.h 的 DT_* 枚举与
//     kernel 侧 basic_api/kernel_type.h 的 DT_* 宏冲突），kernel 经
//     GET_TILING_DATA_WITH_STRUCT 按序列化布局裸拷贝 sizeof(struct) 字节。
//
// Field units: nPixel/tileNum/nGauss = 元素数/个数, ubSize = 字节,
//   blockDim = 核数, useTwoPass = 布尔语义 0/1.
// =============================================================================

#pragma once

#if defined(__CCE_AICORE__) || defined(__CCE__)

// ---------------------------------------------------------------------------
// device-side mirror（序列化布局镜像；与 host 宏结构体字段顺序/类型逐字段一致）
// ---------------------------------------------------------------------------
#include <cstdint>
namespace optiling {
#pragma pack(push, 8)
struct CalcRenderBwdClipLastidsTilingData {
    int64_t  nPixel;     // 每 tile 像素数 P = tileCoords.shape[2]（元素数；实测 256/1024 两类）
    int64_t  tileNum;    // tile 数 T = tileCoords.shape[0]（个数）
    int64_t  nGauss;     // 高斯数 N = gs.shape[0]（行数）
    int64_t  offsetsLen; // S = offsets.shape[0]（元素数）——kernel 几何定位三段布局 VN = max(S-2T,0)
                         // 与 gsIds 位置裁剪的 shape 事实来源（spec formula SN/gsIds.shape[0]）
    int64_t  gsIdsLen;   // G = gsIds.shape[0]（元素数）——kernel posC = clip(pos, 0, G-1) 裁剪上界
    uint64_t ubSize;     // GetCoreMemSize(UB) 原始查询值（字节；仅记录备查，kernel 不消费）
    uint32_t blockDim;   // BlockDim = AIV 核数（空 Tensor 时 =1；仅记录，经 SetBlockDim 下发）
    uint32_t useTwoPass; // 预留（恒 0：单遍结构，无 two-pass 变体）
};
#pragma pack(pop)
// 序列化布局契约：8+8+8+8+8+8+4+4 = 56 字节（自然对齐无填充，与 host 侧一致）。
// kernel 入口按偏移 0..4 × int64 裸解析 nPixel/tileNum/nGauss/offsetsLen/gsIdsLen ——
// offsetsLen/gsIdsLen 放在前 40 字节（基线版 tiling 布局的既有可靠区）是有意的：
// aclnn 运行时向 kernel 递交 tiling buffer 时，实测前 40 字节（旧布局全量）恒可靠，
// 而越过旧布局末尾的追加字段偶发读到未初始化/陈旧内容（间歇 aivec err95 MTE 非法
// GM 地址——追加字段被读成垃圾值后 gsIds 快路守卫失效所致，Task 30 实测）。
static_assert(sizeof(CalcRenderBwdClipLastidsTilingData) == 56, "tiling struct layout drift vs host macro struct");
} // namespace optiling

#else // host compile

#include "register/tilingdata_base.h"

namespace optiling {

// 本算子 TilingData — 单模板（tilingKey=0），kernel 按偏移 0/1/2 × int64 裸解析前 3 字段（与基线一致），
// 按偏移 24/32 × int64 裸解析 offsetsLen/gsIdsLen（前 40 字节可靠区，spec formula 几何语义所需 shape 事实）
BEGIN_TILING_DATA_DEF(CalcRenderBwdClipLastidsTilingData)
    TILING_DATA_FIELD_DEF(int64_t,  nPixel);     // 每 tile 像素数 P = tileCoords.shape[2]（元素数；实测 256/1024 两类）
    TILING_DATA_FIELD_DEF(int64_t,  tileNum);    // tile 数 T = tileCoords.shape[0]（个数）
    TILING_DATA_FIELD_DEF(int64_t,  nGauss);     // 高斯数 N = gs.shape[0]（行数）
    TILING_DATA_FIELD_DEF(int64_t,  offsetsLen); // S = offsets.shape[0]（元素数）——kernel 几何定位三段布局
    TILING_DATA_FIELD_DEF(int64_t,  gsIdsLen);   // G = gsIds.shape[0]（元素数）——kernel posC 裁剪上界
    TILING_DATA_FIELD_DEF(uint64_t, ubSize);     // GetCoreMemSize(UB) 原始查询值（字节；仅记录备查，kernel 不消费）
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);   // BlockDim = AIV 核数（空 Tensor 时 =1；仅记录，经 SetBlockDim 下发）
    TILING_DATA_FIELD_DEF(uint32_t, useTwoPass); // 预留（恒 0：单遍结构，无 two-pass 变体）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CalcRenderBwdClipLastids, CalcRenderBwdClipLastidsTilingData)

} // namespace optiling

#endif // __CCE_AICORE__ / __CCE__
