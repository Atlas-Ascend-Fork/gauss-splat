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
// operators/calc_render_fwd_clip_last_ids_package/op_host/arch35/calc_render_fwd_clip_last_ids_tiling.h
// =============================================================================
//
// ROLE: CalcRenderFwdClipLastids 公共 TilingData 汇总结构体 —— 全局唯一定义
//   （设计事实源 docs/calc_render_fwd_clip_last_ids/design/TilingData.md，字段/类型/含义
//   逐字段一致，不增不删；单分支 tilingKey=0，见 TilingKey.md，无分支私有字段
//   并入问题；workspace = 0，不设 workspace 字段）。
//
// 共享包含（host / kernel 同一定义，不重复定义）：
//   - host：op_host/arch35/calc_render_fwd_clip_last_ids_tiling.cpp（TilingFunc 经
//     GetRawTilingData 裸填 6 字段并 SetDataSize(sizeof)）。
//   - device：op_kernel/arch35（经 calc_render_fwd_clip_last_ids_tiling_data.h 转发包含本
//     文件；kernel 入口 REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT 消费，
//     见 DevView.md「Kernel 侧代码组织」）。
//
// 注册方式按工程现有管线适配（TilingData.md 约定的等价实现）：kernel 侧
// REGISTER_TILING_DEFAULT 生成 tiling 结构元数据（48 字节 ABI），host 侧经
// IMPL_OP_OPTILING 注册的 TilingFunc 用 GetRawTilingData 填充；不使用
// BEGIN_TILING_DATA_DEF / REGISTER_TILING_DATA_CLASS 宏（register/tilingdata_base.h
// 仅 host 编译器可用，且宏结构体内存布局 ≠ 序列化布局，无法与本工程 host/device
// 双端共享同一 POD 的既有通路兼容）。
//
// 同头文件共享常量：DCACHE_RESERVED = 128KB —— host
// SetLocalMemorySize(ubSize − DCACHE_RESERVED) 与 kernel 侧容量校验同源
// （SIMT 范式 DCache 预留 ≥ 32KB 最低要求；TilingData.md / DevView.md）。
// =============================================================================

// host 侧包含入口：CalcRenderFwdClipLastidsTilingData / DCACHE_RESERVED 唯一定义位于
// kernel 侧 op_kernel/arch35/calc_render_fwd_clip_last_ids_tiling_data.h，host 经本文件
// 相对路径转发取用（与 flash_gaussian_build_mask / calc_render_simt_bwd 约定一致）。
// 定义放 kernel 侧可同时满足二进制编译的平铺源码目录（见 kernel 侧头文件说明）。
#ifndef CALC_RENDER_FWD_CLIP_LAST_IDS_TILING_H
#define CALC_RENDER_FWD_CLIP_LAST_IDS_TILING_H

#include "../../op_kernel/arch35/calc_render_fwd_clip_last_ids_tiling_data.h"

#endif // CALC_RENDER_FWD_CLIP_LAST_IDS_TILING_H
