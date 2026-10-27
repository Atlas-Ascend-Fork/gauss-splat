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
#include "_phase2_tile_compact_vf_half1_vf.h"
#include "_phase2_tile_compact_k5_vf_half0_vf.h"
#include "_phase1_tile_vf.h"
#include "_phase2_tile_compact_k6_vf_half1_vf.h"
#include "_phase2_tile_compact_k7_vf_half0_vf.h"
#include "_phase2_tile_compact_k5_vf_half1_vf.h"
#include "_phase2_tile_compact_k4_vf_half1_vf.h"
#include "_packed_filter_float_to_u8_vf.h"
#include "_phase2_tile_compact_k4_vf_half0_vf.h"
#include "_phase2_tile_compact_vf_half0_vf.h"
#include "_phase2_tile_compact_k7_vf_half1_vf.h"
#include "_phase2_tile_compact_k6_vf_half0_vf.h"

__aicore__ inline void gaussian_filter_vec(GM_ADDR means_, GM_ADDR colors_, GM_ADDR quats_, GM_ADDR scales_, GM_ADDR det_, GM_ADDR depths_, GM_ADDR means2d_, GM_ADDR radius_in_, GM_ADDR conics_, GM_ADDR covars2d_, GM_ADDR opacities_, GM_ADDR compensations_, GM_ADDR vi_, GM_ADDR sigma_, GM_ADDR means_culling_, GM_ADDR colors_culling_, GM_ADDR quats_culling_, GM_ADDR scales_culling_, GM_ADDR means2d_culling_, GM_ADDR depths_culling_, GM_ADDR radius_culling_, GM_ADDR covars2d_culling_, GM_ADDR conics_culling_, GM_ADDR opacities_culling_, GM_ADDR vi_culling_, GM_ADDR compensations_culling_, GM_ADDR filter_float_, GM_ADDR cnt_float_, GM_ADDR dense_indices_, GM_ADDR proj_filter_packed_, GM_ADDR workspace, int B, int C, int N, float width, float height, float near_plane, float far_plane, float if_depth_correct, int color_dims, int sigma_len, int M) {
    TPipe pipe;
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    SEvent<PIPE_V, PIPE_MTE2, true> _tmp_sevent_valid_ubin_0;
    DEvent<PIPE_MTE2, PIPE_V, false> _tmp_devent_ready_ubin_2;
    DEvent<PIPE_V, PIPE_MTE2, true> _tmp_devent_valid_ubin_1;
    DEvent<PIPE_MTE2, PIPE_V, false> _tmp_devent_ready_ubin_1;
    DEvent<PIPE_V, PIPE_MTE2, true> _tmp_devent_valid_ubin_2;
    SEvent<PIPE_MTE2, PIPE_V, false> _tmp_sevent_ready_ubin_0;
    DEvent<PIPE_V, PIPE_MTE3, false> _tmp_devent_ready_ubout_2;
    DEvent<PIPE_MTE3, PIPE_V, true> _tmp_devent_valid_ubout_2;
    SEvent<PIPE_MTE3, PIPE_V, true> _tmp_sevent_valid_ubout_0;
    SEvent<PIPE_V, PIPE_MTE3, false> _tmp_sevent_ready_ubout_0;
    DEvent<PIPE_MTE3, PIPE_V, true> _tmp_devent_valid_ubout_1;
    DEvent<PIPE_V, PIPE_MTE3, false> _tmp_devent_ready_ubout_1;
    GlobalTensor<float> means; means.SetGlobalBuffer((__gm__ float*) means_);
    GlobalTensor<float> colors; colors.SetGlobalBuffer((__gm__ float*) colors_);
    GlobalTensor<float> quats; quats.SetGlobalBuffer((__gm__ float*) quats_);
    GlobalTensor<float> scales; scales.SetGlobalBuffer((__gm__ float*) scales_);
    GlobalTensor<float> det; det.SetGlobalBuffer((__gm__ float*) det_);
    GlobalTensor<float> depths; depths.SetGlobalBuffer((__gm__ float*) depths_);
    GlobalTensor<float> means2d; means2d.SetGlobalBuffer((__gm__ float*) means2d_);
    GlobalTensor<float> radius_in; radius_in.SetGlobalBuffer((__gm__ float*) radius_in_);
    GlobalTensor<float> conics; conics.SetGlobalBuffer((__gm__ float*) conics_);
    GlobalTensor<float> covars2d; covars2d.SetGlobalBuffer((__gm__ float*) covars2d_);
    GlobalTensor<float> opacities; opacities.SetGlobalBuffer((__gm__ float*) opacities_);
    GlobalTensor<float> compensations; compensations.SetGlobalBuffer((__gm__ float*) compensations_);
    GlobalTensor<float> vi; vi.SetGlobalBuffer((__gm__ float*) vi_);
    GlobalTensor<float> sigma; sigma.SetGlobalBuffer((__gm__ float*) sigma_);
    GlobalTensor<float> means_culling; means_culling.SetGlobalBuffer((__gm__ float*) means_culling_);
    GlobalTensor<float> colors_culling; colors_culling.SetGlobalBuffer((__gm__ float*) colors_culling_);
    GlobalTensor<float> quats_culling; quats_culling.SetGlobalBuffer((__gm__ float*) quats_culling_);
    GlobalTensor<float> scales_culling; scales_culling.SetGlobalBuffer((__gm__ float*) scales_culling_);
    GlobalTensor<float> means2d_culling; means2d_culling.SetGlobalBuffer((__gm__ float*) means2d_culling_);
    GlobalTensor<float> depths_culling; depths_culling.SetGlobalBuffer((__gm__ float*) depths_culling_);
    GlobalTensor<float> radius_culling; radius_culling.SetGlobalBuffer((__gm__ float*) radius_culling_);
    GlobalTensor<float> covars2d_culling; covars2d_culling.SetGlobalBuffer((__gm__ float*) covars2d_culling_);
    GlobalTensor<float> conics_culling; conics_culling.SetGlobalBuffer((__gm__ float*) conics_culling_);
    GlobalTensor<float> opacities_culling; opacities_culling.SetGlobalBuffer((__gm__ float*) opacities_culling_);
    GlobalTensor<float> vi_culling; vi_culling.SetGlobalBuffer((__gm__ float*) vi_culling_);
    GlobalTensor<float> compensations_culling; compensations_culling.SetGlobalBuffer((__gm__ float*) compensations_culling_);
    GlobalTensor<float> filter_float; filter_float.SetGlobalBuffer((__gm__ float*) filter_float_);
    GlobalTensor<float> cnt_float; cnt_float.SetGlobalBuffer((__gm__ float*) cnt_float_);
    GlobalTensor<int> dense_indices; dense_indices.SetGlobalBuffer((__gm__ int*) dense_indices_);
    GlobalTensor<uint8_t> proj_filter_packed; proj_filter_packed.SetGlobalBuffer((__gm__ uint8_t*) proj_filter_packed_);
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
    float my_vec_idx_f = GetBlockIdx();
    float active_vec_f = 2*GetBlockNum();
    auto tmp_cnt_per_core_ws_ptr = shiftAddr<float>(workspace, 64, _offset);
    GlobalTensor<float> cnt_per_core_ws;
    cnt_per_core_ws.SetGlobalBuffer((__gm__ float*) tmp_cnt_per_core_ws_ptr);
    float sigma_v = (float) sigma.GetValue(0);
    float depth_scale = 0.0f;
    float half_f = 0.5f;
    if (if_depth_correct > half_f) {
        depth_scale = (float) (1.0f / sigma_v);  // inv_sigma for dc = 1 - depth/sigma
    } else {
        depth_scale = (float) 0.0f;
    }
    int total_tiles = CeilDiv(N, 384);
    int next_vec_idx = GetBlockIdx() + 1;
    int tile_cnt_db = 0;
    float _tmp_var_44 = 1.0f;
    powers_ub.SetValue(0, (float)(_tmp_var_44));
    float _tmp_var_46 = 2.0f;
    powers_ub[1].SetValue(0, (float)(_tmp_var_46));
    float _tmp_var_48 = 4.0f;
    powers_ub[2].SetValue(0, (float)(_tmp_var_48));
    float _tmp_var_50 = 8.0f;
    powers_ub[3].SetValue(0, (float)(_tmp_var_50));
    float _tmp_var_52 = 16.0f;
    powers_ub[4].SetValue(0, (float)(_tmp_var_52));
    float _tmp_var_54 = 32.0f;
    powers_ub[5].SetValue(0, (float)(_tmp_var_54));
    float _tmp_var_56 = 64.0f;
    powers_ub[6].SetValue(0, (float)(_tmp_var_56));
    float _tmp_var_58 = 128.0f;
    powers_ub[7].SetValue(0, (float)(_tmp_var_58));
    float _tmp_var_60 = 1.0f;
    powers_ub[8].SetValue(0, (float)(_tmp_var_60));
    float _tmp_var_62 = 2.0f;
    powers_ub[9].SetValue(0, (float)(_tmp_var_62));
    float _tmp_var_64 = 4.0f;
    powers_ub[10].SetValue(0, (float)(_tmp_var_64));
    float _tmp_var_66 = 8.0f;
    powers_ub[11].SetValue(0, (float)(_tmp_var_66));
    float _tmp_var_68 = 16.0f;
    powers_ub[12].SetValue(0, (float)(_tmp_var_68));
    float _tmp_var_70 = 32.0f;
    powers_ub[13].SetValue(0, (float)(_tmp_var_70));
    float _tmp_var_72 = 64.0f;
    powers_ub[14].SetValue(0, (float)(_tmp_var_72));
    float _tmp_var_74 = 128.0f;
    powers_ub[15].SetValue(0, (float)(_tmp_var_74));
    float _tmp_var_76 = 1.0f;
    powers_ub[16].SetValue(0, (float)(_tmp_var_76));
    float _tmp_var_78 = 2.0f;
    powers_ub[17].SetValue(0, (float)(_tmp_var_78));
    float _tmp_var_80 = 4.0f;
    powers_ub[18].SetValue(0, (float)(_tmp_var_80));
    float _tmp_var_82 = 8.0f;
    powers_ub[19].SetValue(0, (float)(_tmp_var_82));
    float _tmp_var_84 = 16.0f;
    powers_ub[20].SetValue(0, (float)(_tmp_var_84));
    float _tmp_var_86 = 32.0f;
    powers_ub[21].SetValue(0, (float)(_tmp_var_86));
    float _tmp_var_88 = 64.0f;
    powers_ub[22].SetValue(0, (float)(_tmp_var_88));
    float _tmp_var_90 = 128.0f;
    powers_ub[23].SetValue(0, (float)(_tmp_var_90));
    float _tmp_var_92 = 1.0f;
    powers_ub[24].SetValue(0, (float)(_tmp_var_92));
    float _tmp_var_94 = 2.0f;
    powers_ub[25].SetValue(0, (float)(_tmp_var_94));
    float _tmp_var_96 = 4.0f;
    powers_ub[26].SetValue(0, (float)(_tmp_var_96));
    float _tmp_var_98 = 8.0f;
    powers_ub[27].SetValue(0, (float)(_tmp_var_98));
    float _tmp_var_100 = 16.0f;
    powers_ub[28].SetValue(0, (float)(_tmp_var_100));
    float _tmp_var_102 = 32.0f;
    powers_ub[29].SetValue(0, (float)(_tmp_var_102));
    float _tmp_var_104 = 64.0f;
    powers_ub[30].SetValue(0, (float)(_tmp_var_104));
    float _tmp_var_106 = 128.0f;
    powers_ub[31].SetValue(0, (float)(_tmp_var_106));
    float _tmp_var_108 = 1.0f;
    powers_ub[32].SetValue(0, (float)(_tmp_var_108));
    float _tmp_var_110 = 2.0f;
    powers_ub[33].SetValue(0, (float)(_tmp_var_110));
    float _tmp_var_112 = 4.0f;
    powers_ub[34].SetValue(0, (float)(_tmp_var_112));
    float _tmp_var_114 = 8.0f;
    powers_ub[35].SetValue(0, (float)(_tmp_var_114));
    float _tmp_var_116 = 16.0f;
    powers_ub[36].SetValue(0, (float)(_tmp_var_116));
    float _tmp_var_118 = 32.0f;
    powers_ub[37].SetValue(0, (float)(_tmp_var_118));
    float _tmp_var_120 = 64.0f;
    powers_ub[38].SetValue(0, (float)(_tmp_var_120));
    float _tmp_var_122 = 128.0f;
    powers_ub[39].SetValue(0, (float)(_tmp_var_122));
    float _tmp_var_124 = 1.0f;
    powers_ub[40].SetValue(0, (float)(_tmp_var_124));
    float _tmp_var_126 = 2.0f;
    powers_ub[41].SetValue(0, (float)(_tmp_var_126));
    float _tmp_var_128 = 4.0f;
    powers_ub[42].SetValue(0, (float)(_tmp_var_128));
    float _tmp_var_130 = 8.0f;
    powers_ub[43].SetValue(0, (float)(_tmp_var_130));
    float _tmp_var_132 = 16.0f;
    powers_ub[44].SetValue(0, (float)(_tmp_var_132));
    float _tmp_var_134 = 32.0f;
    powers_ub[45].SetValue(0, (float)(_tmp_var_134));
    float _tmp_var_136 = 64.0f;
    powers_ub[46].SetValue(0, (float)(_tmp_var_136));
    float _tmp_var_138 = 128.0f;
    powers_ub[47].SetValue(0, (float)(_tmp_var_138));
    float _tmp_var_140 = 1.0f;
    powers_ub[48].SetValue(0, (float)(_tmp_var_140));
    float _tmp_var_142 = 2.0f;
    powers_ub[49].SetValue(0, (float)(_tmp_var_142));
    float _tmp_var_144 = 4.0f;
    powers_ub[50].SetValue(0, (float)(_tmp_var_144));
    float _tmp_var_146 = 8.0f;
    powers_ub[51].SetValue(0, (float)(_tmp_var_146));
    float _tmp_var_148 = 16.0f;
    powers_ub[52].SetValue(0, (float)(_tmp_var_148));
    float _tmp_var_150 = 32.0f;
    powers_ub[53].SetValue(0, (float)(_tmp_var_150));
    float _tmp_var_152 = 64.0f;
    powers_ub[54].SetValue(0, (float)(_tmp_var_152));
    float _tmp_var_154 = 128.0f;
    powers_ub[55].SetValue(0, (float)(_tmp_var_154));
    float _tmp_var_156 = 1.0f;
    powers_ub[56].SetValue(0, (float)(_tmp_var_156));
    float _tmp_var_158 = 2.0f;
    powers_ub[57].SetValue(0, (float)(_tmp_var_158));
    float _tmp_var_160 = 4.0f;
    powers_ub[58].SetValue(0, (float)(_tmp_var_160));
    float _tmp_var_162 = 8.0f;
    powers_ub[59].SetValue(0, (float)(_tmp_var_162));
    float _tmp_var_164 = 16.0f;
    powers_ub[60].SetValue(0, (float)(_tmp_var_164));
    float _tmp_var_166 = 32.0f;
    powers_ub[61].SetValue(0, (float)(_tmp_var_166));
    float _tmp_var_168 = 64.0f;
    powers_ub[62].SetValue(0, (float)(_tmp_var_168));
    float _tmp_var_170 = 128.0f;
    powers_ub[63].SetValue(0, (float)(_tmp_var_170));
    for (int b = 0; b < B; b += 1) {
        for (int c = 0; c < C; c += 1) {
            int global_base_int = 0;
            float global_base_f = 0.0f;
            float valid_n_f = 0;
            for (int window_tile_begin = 0; window_tile_begin < total_tiles; window_tile_begin += 3584) {
                int window_tile_end = Min(window_tile_begin + 3584, total_tiles);
                int window_tile_count = -window_tile_begin + window_tile_end;
                int tile_begin = window_tile_begin + ((window_tile_count*GetBlockIdx()) / (2*GetBlockNum()));
                int tile_end = window_tile_begin + ((next_vec_idx*window_tile_count) / (2*GetBlockNum()));
                _tmp_sevent_valid_ubout_0.wait();
                _zero_lane0_vf((__ubuf__ float*) (pc_cnt_ub).GetPhyAddr());
                int local_tile_idx = 0;
                _tmp_sevent_ready_ubout_0.set();
                _tmp_sevent_ready_ubout_0.wait();
                for (int tile_idx = tile_begin; tile_idx < tile_end; tile_idx += 1) {
                    int n0 = 384*tile_idx;
                    int valid_n = Min(384, N - n0);
                    valid_n_f = (float) valid_n;
                    _tmp_devent_valid_ubin_1.wait();
                    GM2UBPAD(det_buf.get(tile_cnt_db), det[b * C * N + c * N + n0], 1, 4*valid_n, 0, ((384 - valid_n) / (8)));
                    GM2UBPAD(dep_buf.get(tile_cnt_db), depths[b * C * N + c * N + n0], 1, 4*valid_n, 0, ((384 - valid_n) / (8)));
                    GM2UBPAD(m2d_buf.get(tile_cnt_db), means2d[b * C * 2 * N + c * 2 * N + n0], 2, 4*valid_n, 4*N - 4*valid_n, ((384 - valid_n) / (8)));
                    GM2UBPAD(rad_buf.get(tile_cnt_db), radius_in[b * C * 2 * N + c * 2 * N + n0], 2, 4*valid_n, 4*N - 4*valid_n, ((384 - valid_n) / (8)));
                    GM2UBPAD(vi_buf.get(tile_cnt_db), vi[b * C * N + c * N + n0], 1, 4*valid_n, 0, ((384 - valid_n) / (8)));
                    _tmp_devent_ready_ubin_1.set();
                    _tmp_devent_ready_ubin_1.wait();
                    _tmp_devent_valid_ubout_1.wait();
                    _phase1_tile_vf((__ubuf__ float*) (det_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (packed_filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (powers_ub).GetPhyAddr(), (__ubuf__ float*) (pc_cnt_ub).GetPhyAddr(), (__ubuf__ float*) (chunk_cnts_ub).GetPhyAddr(), local_tile_idx, sigma_v, width, height, near_plane, far_plane, if_depth_correct, valid_n_f);
                    _tmp_devent_ready_ubout_1.set();
                    _tmp_devent_ready_ubout_1.wait();
                    UB2GMPAD(filter_float[b * C * N + c * N + n0], filter_buf.get(tile_cnt_db), 1, 4*valid_n, ((384 - valid_n) / (8)), 0);
                    _tmp_devent_valid_ubout_1.set();
                    _tmp_devent_valid_ubout_1.wait();
                    _packed_filter_float_to_u8_vf((__ubuf__ float*) (packed_filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ half*) (packed_filter_half_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ uint8_t*) (packed_filter_u8_buf.get(tile_cnt_db)).GetPhyAddr());
                    int byte0 = 48*tile_idx;
                    _tmp_devent_ready_ubout_1.set();
                    _tmp_devent_ready_ubout_1.wait();
                    for (int pack_chunk_idx = 0; pack_chunk_idx < 6; pack_chunk_idx += 1) {
                        if (valid_n > 64*pack_chunk_idx) {
                            int chunk_valid_n = Min(64, -64*pack_chunk_idx + valid_n);
                            int bytes_to_write = CeilDiv(chunk_valid_n, 8);
                            int byte_off = byte0 + 8*pack_chunk_idx;
                            UB2GMPAD(proj_filter_packed[b * C * M + c * M + byte_off], packed_filter_u8_buf.get(tile_cnt_db)[pack_chunk_idx * 64], 1, bytes_to_write, ((64 - bytes_to_write) / (32)), 0);
                        }
                    }
                    tile_cnt_db = tile_cnt_db + 1;
                    local_tile_idx = local_tile_idx + 1;
                    _tmp_devent_valid_ubin_1.set();
                    _tmp_devent_valid_ubout_1.set();
                }
                UB2GMPAD(cnt_per_core_ws[GetBlockIdx()], pc_cnt_ub, 1, 4, 0, 0);
                ALLVEC_READY(0, PIPE_MTE3);
                ALLVEC_WAIT(0, PIPE_S);
                _tmp_sevent_valid_ubin_0.wait();
                GM2UBPAD(ws_all_ub, cnt_per_core_ws, 1, 256, 0, 0);
                _tmp_sevent_ready_ubin_0.set();
                _tmp_sevent_ready_ubin_0.wait();
                _tmp_sevent_valid_ubout_0.set();
                _tmp_sevent_valid_ubout_0.wait();
                _prefix_scan_offset_total_vf((__ubuf__ float*) (offset_ub).GetPhyAddr(), (__ubuf__ float*) (total_ub).GetPhyAddr(), (__ubuf__ float*) (ws_all_ub).GetPhyAddr(), my_vec_idx_f, active_vec_f);
                PipeBarrier<PIPE_ALL>();
                ALLVEC_READY(1, PIPE_MTE3);
                ALLVEC_WAIT(1, PIPE_S);
                PipeBarrier<PIPE_ALL>();
                int offset_int = (int) offset_ub.GetValue(0);
                int local_offset = 0;
                int local_tile_idx_1 = 0;
                _tmp_sevent_ready_ubout_0.set();
                _tmp_sevent_ready_ubout_0.wait();
                for (int tile_idx_1 = tile_begin; tile_idx_1 < tile_end; tile_idx_1 += 1) {
                    int n0_1 = 384*tile_idx_1;
                    int valid_n_1 = Min(384, N - n0_1);
                    valid_n_f = (float) valid_n_1;
                    int tile_chunk_cnt0 = 0;
                    int tile_chunk_slot0 = 6*local_tile_idx_1;
                    tile_chunk_cnt0 = (int) chunk_cnts_ub[tile_chunk_slot0 * 1].GetValue(0);
                    int tile_chunk_cnt1 = 0;
                    int tile_chunk_slot1 = 6*local_tile_idx_1 + 1;
                    tile_chunk_cnt1 = (int) chunk_cnts_ub[tile_chunk_slot1 * 1].GetValue(0);
                    int tile_chunk_cnt2 = 0;
                    int tile_chunk_slot2 = 6*local_tile_idx_1 + 2;
                    tile_chunk_cnt2 = (int) chunk_cnts_ub[tile_chunk_slot2 * 1].GetValue(0);
                    int tile_chunk_cnt3 = 0;
                    int tile_chunk_slot3 = 6*local_tile_idx_1 + 3;
                    tile_chunk_cnt3 = (int) chunk_cnts_ub[tile_chunk_slot3 * 1].GetValue(0);
                    int tile_chunk_cnt4 = 0;
                    int tile_chunk_slot4 = 6*local_tile_idx_1 + 4;
                    tile_chunk_cnt4 = (int) chunk_cnts_ub[tile_chunk_slot4 * 1].GetValue(0);
                    int tile_chunk_cnt5 = 0;
                    int tile_chunk_slot5 = 6*local_tile_idx_1 + 5;
                    tile_chunk_cnt5 = (int) chunk_cnts_ub[tile_chunk_slot5 * 1].GetValue(0);
                    _tmp_devent_valid_ubin_2.wait();
                    int tile_dst0 = 0;
                    GM2UBPAD(filter_buf.get(tile_cnt_db), filter_float[b * C * N + c * N + n0_1], 1, 4*valid_n_1, 0, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(dep_buf.get(tile_cnt_db), depths[b * C * N + c * N + n0_1], 1, 4*valid_n_1, 0, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(m2d_buf.get(tile_cnt_db), means2d[b * C * 2 * N + c * 2 * N + n0_1], 2, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(rad_buf.get(tile_cnt_db), radius_in[b * C * 2 * N + c * 2 * N + n0_1], 2, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(vi_buf.get(tile_cnt_db), vi[b * C * N + c * N + n0_1], 1, 4*valid_n_1, 0, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(means_buf.get(tile_cnt_db), means[b * color_dims * N + n0_1], 3, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    if (color_dims > 6) {
                        GM2UBPAD(colors_buf.get(tile_cnt_db), colors[b * color_dims * N + n0_1], 7, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    } else {
                        if (color_dims > 5) {
                            GM2UBPAD(colors_buf.get(tile_cnt_db), colors[b * color_dims * N + n0_1], 6, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                        } else {
                            if (color_dims > 4) {
                                GM2UBPAD(colors_buf.get(tile_cnt_db), colors[b * color_dims * N + n0_1], 5, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                            } else {
                                if (color_dims > 3) {
                                    GM2UBPAD(colors_buf.get(tile_cnt_db), colors[b * color_dims * N + n0_1], 4, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                                } else {
                                    GM2UBPAD(colors_buf.get(tile_cnt_db), colors[b * color_dims * N + n0_1], 3, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                                }
                            }
                        }
                    }
                    GM2UBPAD(quats_buf.get(tile_cnt_db), quats[b * 4 * N + n0_1], 4, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(scales_buf.get(tile_cnt_db), scales[b * color_dims * N + n0_1], 3, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(conics_buf.get(tile_cnt_db), conics[b * C * color_dims * N + c * color_dims * N + n0_1], 3, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(covars2d_buf.get(tile_cnt_db), covars2d[b * C * color_dims * N + c * color_dims * N + n0_1], 3, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(op_buf.get(tile_cnt_db), opacities[b * N + n0_1], 1, 4*valid_n_1, 4*N - 4*valid_n_1, ((384 - valid_n_1) / (8)));
                    GM2UBPAD(comp_buf.get(tile_cnt_db), compensations[b * C * N + c * N + n0_1], 1, 4*valid_n_1, 0, ((384 - valid_n_1) / (8)));
                    _tmp_devent_ready_ubin_2.set();
                    _tmp_devent_ready_ubin_2.wait();
                    _tmp_devent_valid_ubout_2.wait();
                    if (color_dims > 6) {
                        _phase2_tile_compact_k7_vf_half0_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_dst0, tile_chunk_cnt0 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_dst0, n0_1, depth_scale, valid_n_f);
                    } else {
                        if (color_dims > 5) {
                            _phase2_tile_compact_k6_vf_half0_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_dst0, tile_chunk_cnt0 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_dst0, n0_1, depth_scale, valid_n_f);
                        } else {
                            if (color_dims > 4) {
                                _phase2_tile_compact_k5_vf_half0_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_dst0, tile_chunk_cnt0 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_dst0, n0_1, depth_scale, valid_n_f);
                            } else {
                                if (color_dims > 3) {
                                    _phase2_tile_compact_k4_vf_half0_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_dst0, tile_chunk_cnt0 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_dst0, n0_1, depth_scale, valid_n_f);
                                } else {
                                    _phase2_tile_compact_vf_half0_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_dst0, tile_chunk_cnt0 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_dst0, n0_1, depth_scale, valid_n_f);
                                }
                            }
                        }
                    }
                    if (color_dims > 6) {
                        _phase2_tile_compact_k7_vf_half1_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_dst0, n0_1, depth_scale, valid_n_f);
                    } else {
                        if (color_dims > 5) {
                            _phase2_tile_compact_k6_vf_half1_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_dst0, n0_1, depth_scale, valid_n_f);
                        } else {
                            if (color_dims > 4) {
                                _phase2_tile_compact_k5_vf_half1_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_dst0, n0_1, depth_scale, valid_n_f);
                            } else {
                                if (color_dims > 3) {
                                    _phase2_tile_compact_k4_vf_half1_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_dst0, n0_1, depth_scale, valid_n_f);
                                } else {
                                    _phase2_tile_compact_vf_half1_vf((__ubuf__ float*) (filter_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (dep_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (m2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (rad_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (means_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (colors_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (quats_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (scales_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (conics_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (covars2d_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (op_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (comp_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (vi_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ float*) (combined_buf.get(tile_cnt_db)).GetPhyAddr(), (__ubuf__ int*) (dense_idx_buf.get(tile_cnt_db)).GetPhyAddr(), tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_dst0, tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_dst0, n0_1, depth_scale, valid_n_f);
                                }
                            }
                        }
                    }
                    _tmp_devent_ready_ubout_2.set();
                    _tmp_devent_ready_ubout_2.wait();
                    if ((tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_chunk_cnt5 + tile_dst0) > 0) {
                        UB2GMPAD(means_culling[b * C * color_dims * N + c * color_dims * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[tile_dst0], 3, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(colors_culling[b * C * color_dims * N + c * color_dims * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[3 * 384 + tile_dst0], color_dims, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(quats_culling[b * C * 4 * N + c * 4 * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[10 * 384 + tile_dst0], 4, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(scales_culling[b * C * color_dims * N + c * color_dims * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[14 * 384 + tile_dst0], 3, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(means2d_culling[b * C * 2 * N + c * 2 * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[17 * 384 + tile_dst0], 2, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(depths_culling[b * C * N + c * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[19 * 384 + tile_dst0], 1, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 0);
                        UB2GMPAD(radius_culling[b * C * 2 * N + c * 2 * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[20 * 384 + tile_dst0], 2, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(covars2d_culling[b * C * color_dims * N + c * color_dims * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[22 * 384 + tile_dst0], 3, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(conics_culling[b * C * color_dims * N + c * color_dims * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[25 * 384 + tile_dst0], 3, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 4*N - 4*tile_chunk_cnt0 - 4*tile_chunk_cnt1 - 4*tile_chunk_cnt2 - 4*tile_chunk_cnt3 - 4*tile_chunk_cnt4 - 4*tile_chunk_cnt5 - 4*tile_dst0);
                        UB2GMPAD(opacities_culling[b * C * N + c * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[28 * 384 + tile_dst0], 1, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 0);
                        UB2GMPAD(vi_culling[b * C * N + c * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[29 * 384 + tile_dst0], 1, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 0);
                        UB2GMPAD(compensations_culling[b * C * N + c * N + (global_base_int + local_offset + offset_int)], combined_buf.get(tile_cnt_db)[30 * 384 + tile_dst0], 1, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 0);
                        UB2GMPAD(dense_indices[b * C * N + c * N + (global_base_int + local_offset + offset_int)], dense_idx_buf.get(tile_cnt_db)[tile_dst0], 1, 4*tile_chunk_cnt0 + 4*tile_chunk_cnt1 + 4*tile_chunk_cnt2 + 4*tile_chunk_cnt3 + 4*tile_chunk_cnt4 + 4*tile_chunk_cnt5 + 4*tile_dst0, ((-tile_chunk_cnt0 - tile_chunk_cnt1 - tile_chunk_cnt2 - tile_chunk_cnt3 - tile_chunk_cnt4 - tile_chunk_cnt5 - tile_dst0 + 384) / (8)), 0);
                    }
                    local_offset = local_offset + (tile_chunk_cnt0 + tile_chunk_cnt1 + tile_chunk_cnt2 + tile_chunk_cnt3 + tile_chunk_cnt4 + tile_chunk_cnt5 + tile_dst0);
                    tile_cnt_db = tile_cnt_db + 1;
                    local_tile_idx_1 = local_tile_idx_1 + 1;
                    _tmp_devent_valid_ubin_2.set();
                    _tmp_devent_valid_ubout_2.set();
                }
                int window_total_int = (int) total_ub.GetValue(0);
                float window_total_f = (float) total_ub.GetValue(0);
                global_base_int = global_base_int + window_total_int;
                global_base_f = global_base_f + window_total_f;
                _tmp_sevent_valid_ubin_0.set();
                _tmp_sevent_valid_ubout_0.set();
            }
            if (GetBlockIdx() == (2*GetBlockNum() - 1)) {
                cnt_float[b * C + c].SetValue(0, (float)(global_base_f));
            }
        }
    }

    // Auto generated code. Readability is not guaranteed.
}
