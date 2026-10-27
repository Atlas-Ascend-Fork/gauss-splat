/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "tensorutils.h"
#include "_zero_lane0_vf.h"
#include "_prefix_scan_offset_total_vf.h"
#include "_phase1_tile_vf.h"
#include "_packed_filter_float_to_u8_vf.h"
#include "_phase2_tile_compact_vf_half0_vf.h"
#include "_phase2_tile_compact_vf_half1_vf.h"
#include "_phase2_tile_compact_k4_vf_half0_vf.h"
#include "_phase2_tile_compact_k4_vf_half1_vf.h"
#include "_phase2_tile_compact_k5_vf_half0_vf.h"
#include "_phase2_tile_compact_k5_vf_half1_vf.h"
#include "_phase2_tile_compact_k6_vf_half0_vf.h"
#include "_phase2_tile_compact_k6_vf_half1_vf.h"
#include "_phase2_tile_compact_k7_vf_half0_vf.h"
#include "_phase2_tile_compact_k7_vf_half1_vf.h"

__aicore__ inline void gaussian_filter_cube(GM_ADDR means_, GM_ADDR colors_, GM_ADDR quats_, GM_ADDR scales_, GM_ADDR det_, GM_ADDR depths_, GM_ADDR means2d_, GM_ADDR radius_in_, GM_ADDR conics_, GM_ADDR covars2d_, GM_ADDR opacities_, GM_ADDR compensations_, GM_ADDR vi_, GM_ADDR sigma_, GM_ADDR means_culling_, GM_ADDR colors_culling_, GM_ADDR quats_culling_, GM_ADDR scales_culling_, GM_ADDR means2d_culling_, GM_ADDR depths_culling_, GM_ADDR radius_culling_, GM_ADDR covars2d_culling_, GM_ADDR conics_culling_, GM_ADDR opacities_culling_, GM_ADDR vi_culling_, GM_ADDR compensations_culling_, GM_ADDR filter_float_, GM_ADDR cnt_float_, GM_ADDR dense_indices_, GM_ADDR proj_filter_packed_, GM_ADDR workspace, int B, int C, int N, float width, float height, float near_plane, float far_plane, float if_depth_correct, int color_dims, int sigma_len, int M) {
    TPipe pipe;
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    // pipe_ptr->Reset();
    // OccupyMMTE1Events();
    // END: Auto inserted buffers/events/GMTensors
    // ----- User's codes start from here -----
    LocalTensor<float> pc_cnt_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 8);
    LocalTensor<float> offset_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 8);
    LocalTensor<float> total_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 8);
    LocalTensor<float> ws_all_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 64);
    LocalTensor<float> powers_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 64);
    LocalTensor<float> chunk_cnts_ub = AllocateLocalTensor<TPosition::VECCALC, float>(384 * 1);
    DBuff<float, TPosition::VECCALC> det_buf; det_buf.Init(1 * 384);
    DBuff<float, TPosition::VECCALC> dep_buf; dep_buf.Init(1 * 384);
    DBuff<float, TPosition::VECCALC> m2d_buf; m2d_buf.Init(2 * 384);
    DBuff<float, TPosition::VECCALC> rad_buf; rad_buf.Init(2 * 384);
    DBuff<float, TPosition::VECCALC> vi_buf; vi_buf.Init(1 * 384);
    DBuff<float, TPosition::VECCALC> filter_buf; filter_buf.Init(1 * 384);
    DBuff<float, TPosition::VECCALC> packed_filter_buf; packed_filter_buf.Init(6 * 64);
    DBuff<half, TPosition::VECCALC> packed_filter_half_buf; packed_filter_half_buf.Init(6 * 128);
    DBuff<uint8_t, TPosition::VECCALC> packed_filter_u8_buf; packed_filter_u8_buf.Init(6 * 64);
    DBuff<float, TPosition::VECCALC> means_buf; means_buf.Init(3 * 384);
    DBuff<float, TPosition::VECCALC> colors_buf; colors_buf.Init(7 * 384);
    DBuff<float, TPosition::VECCALC> quats_buf; quats_buf.Init(4 * 384);
    DBuff<float, TPosition::VECCALC> scales_buf; scales_buf.Init(3 * 384);
    DBuff<float, TPosition::VECCALC> conics_buf; conics_buf.Init(3 * 384);
    DBuff<float, TPosition::VECCALC> covars2d_buf; covars2d_buf.Init(3 * 384);
    DBuff<float, TPosition::VECCALC> op_buf; op_buf.Init(1 * 384);
    DBuff<float, TPosition::VECCALC> comp_buf; comp_buf.Init(1 * 384);
    DBuff<float, TPosition::VECCALC> combined_buf; combined_buf.Init(31 * 384);
    DBuff<int, TPosition::VECCALC> dense_idx_buf; dense_idx_buf.Init(1 * 384);
    auto tmp_cnt_per_core_ws_ptr = shiftAddr<float>(workspace, 64, _offset);
    GlobalTensor<float> cnt_per_core_ws;
    cnt_per_core_ws.SetGlobalBuffer((__gm__ float*) tmp_cnt_per_core_ws_ptr);
    int total_tiles = CeilDiv(N, 384);

    // Auto generated code. Readability is not guaranteed.
}
