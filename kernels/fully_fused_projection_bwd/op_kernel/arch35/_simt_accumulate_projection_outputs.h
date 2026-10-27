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
#include "simt_api/device_sync_functions.h"

__simt_vf__ __launch_bounds__(256) inline void _simt_accumulate_projection_outputs(__ubuf__ int* dense_indices_ub, __ubuf__ float* color0_ub, __ubuf__ float* color1_ub, __ubuf__ float* color2_ub, __ubuf__ float* color3_ub, __ubuf__ float* color4_ub, __ubuf__ float* color5_ub, __ubuf__ float* color6_ub, __ubuf__ float* vpw0_ub, __ubuf__ float* vpw1_ub, __ubuf__ float* vpw2_ub, __ubuf__ float* opacity_ub, __ubuf__ float* vi_ub, __ubuf__ float* quat0_ub, __ubuf__ float* quat1_ub, __ubuf__ float* quat2_ub, __ubuf__ float* quat3_ub, __ubuf__ float* scale0_ub, __ubuf__ float* scale1_ub, __ubuf__ float* scale2_ub, __gm__ float* v_colors_gm, __gm__ float* v_p_w_gm, __gm__ float* v_opacities_gm, __gm__ float* dvi_gm, __gm__ float* dsigma_gm, __gm__ float* v_quats_gm, __gm__ float* v_scales_gm, int base1, int slice_base1, int color_base, int color_base_ch1, int color_base_ch2, int color_base_ch3, int color_base_ch4, int color_base_ch5, int color_base_ch6, int base3, int base3_ch1, int base3_ch2, int base4, int base4_ch1, int base4_ch2, int base4_ch3, int valid_cols, int color_dims, __ubuf__ float* depths_ub, __ubuf__ float* vi_in_ub, float sigma_v, float if_depth_v){
    float thr_ds_sigma = 0.0f;
    for (int lane = Simt::GetThreadIdx<0>(); lane < valid_cols; lane += Simt::GetThreadNum<0>()) {
        int dense_gidx = dense_indices_ub[lane];
        Simt::AtomicAdd(v_colors_gm + (color_base + dense_gidx), color0_ub[lane]);
        Simt::AtomicAdd(v_colors_gm + (color_base_ch1 + dense_gidx), color1_ub[lane]);
        Simt::AtomicAdd(v_colors_gm + (color_base_ch2 + dense_gidx), color2_ub[lane]);
        if (color_dims > 3) {
            Simt::AtomicAdd(v_colors_gm + (color_base_ch3 + dense_gidx), color3_ub[lane]);
        }
        if (color_dims > 4) {
            Simt::AtomicAdd(v_colors_gm + (color_base_ch4 + dense_gidx), color4_ub[lane]);
        }
        if (color_dims > 5) {
            Simt::AtomicAdd(v_colors_gm + (color_base_ch5 + dense_gidx), color5_ub[lane]);
        }
        if (color_dims > 6) {
            Simt::AtomicAdd(v_colors_gm + (color_base_ch6 + dense_gidx), color6_ub[lane]);
        }
        Simt::AtomicAdd(v_p_w_gm + (base3 + dense_gidx), vpw0_ub[lane]);
        Simt::AtomicAdd(v_p_w_gm + (base3_ch1 + dense_gidx), vpw1_ub[lane]);
        Simt::AtomicAdd(v_p_w_gm + (base3_ch2 + dense_gidx), vpw2_ub[lane]);
        Simt::AtomicAdd(v_opacities_gm + (base1 + dense_gidx), opacity_ub[lane]);
        float vvi = vi_ub[lane];
        float dvi_contrib = vvi;
        if (if_depth_v > 0.5f) {
            float inv_s = 1.0f / sigma_v;
            float dep = depths_ub[lane] * inv_s;
            float dc = max(0.0f, 1.0f - dep);
            float dc_safe = max(dc, 1e-9f);
            dvi_contrib = vvi * dc;
            thr_ds_sigma += (vvi * dep * inv_s) * (vi_in_ub[lane] / dc_safe);
        }
        Simt::AtomicAdd(dvi_gm + (slice_base1 + dense_gidx), dvi_contrib);
        Simt::AtomicAdd(v_quats_gm + (base4 + dense_gidx), quat0_ub[lane]);
        Simt::AtomicAdd(v_quats_gm + (base4_ch1 + dense_gidx), quat1_ub[lane]);
        Simt::AtomicAdd(v_quats_gm + (base4_ch2 + dense_gidx), quat2_ub[lane]);
        Simt::AtomicAdd(v_quats_gm + (base4_ch3 + dense_gidx), quat3_ub[lane]);
        Simt::AtomicAdd(v_scales_gm + (base3 + dense_gidx), scale0_ub[lane]);
        Simt::AtomicAdd(v_scales_gm + (base3_ch1 + dense_gidx), scale1_ub[lane]);
        Simt::AtomicAdd(v_scales_gm + (base3_ch2 + dense_gidx), scale2_ub[lane]);
    }
    if (if_depth_v > 0.5f) {
        float warp_ds = Simt::WarpReduceAddSync(thr_ds_sigma);
        int tid_s = Simt::GetThreadIdx<0>();
        if ((tid_s & 31) == 0) {
            Simt::AtomicAdd(dsigma_gm + 0, warp_ds);
        }
    }
}
