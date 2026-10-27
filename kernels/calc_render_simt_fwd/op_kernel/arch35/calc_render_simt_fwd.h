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
// operators/calc_render_simt_fwd_package/op_kernel/arch35/calc_render_simt_fwd.h
// =============================================================================
//
// calc_render_simt_fwd kernel 侧公共包含入口（arch35 / Ascend950，设计事实源
// docs/calc_render_simt_fwd/design/DevView.md「Kernel 侧代码组织」）：
//   - CalcRenderSimtFwdTilingData 唯一定义于 op_kernel/arch35/calc_render_simt_fwd_tiling_data.h
//     （host 侧 op_host/arch35/calc_render_simt_fwd_tiling.h 相对转发同一文件，双端共享，
//     kernel 入口 REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT 消费）；
//   - SIMD 侧编排 API（kernel_operator.h）+ SIMT 范式 include 纪律（T6：禁
//     device_functions.h / asc_simt.h，防 PipeBarrier 模板实例化错误）；
//   - SIMT VF 线程体与全部编译期常量（fwd_simt_vf.h）。
// 编排类 CalcRenderSimtFwdKernel 内联于入口 TU（单分支无多模板，DevView.md
// 「模板范式文件映射」），本头文件不承载类声明。
// =============================================================================

#ifndef CALC_RENDER_SIMT_FWD_H
#define CALC_RENDER_SIMT_FWD_H

#include "calc_render_simt_fwd_tiling_data.h"
#include "kernel_operator.h"
#include "simt_api/math_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/common_functions.h"   // asc_vf_call（arch35 SIMT VF 启动，cce::async_invoke 同体）
#include "fwd_simt_vf.h"

#endif // CALC_RENDER_SIMT_FWD_H
