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
#include "_compute_quat_rot_scale_cache_vf.h"
#include "_compute_projection_covar_mean_intermediates_vf.h"
#include "_accumulate_covar_direct_no_vr_from_intermediates_vf.h"
#include "_accumulate_covariance_mean_no_vr_from_intermediates_vf.h"
#include "_finish_scale_quat_from_covar_cache_vf.h"
#include "_simt_accumulate_projection_outputs.h"

__aicore__ inline void fully_fused_projection_bwd_vec(GM_ADDR means_, GM_ADDR quats_, GM_ADDR scales_, GM_ADDR conics_, GM_ADDR viewmats_, GM_ADDR ks_, GM_ADDR v_means2d_culling_, GM_ADDR v_depths_culling_, GM_ADDR v_conics_culling_, GM_ADDR v_colors_culling_, GM_ADDR v_opacities_culling_, GM_ADDR v_vi_culling_, GM_ADDR opacities_in_culling_, GM_ADDR vi_culling_, GM_ADDR sigma_, GM_ADDR depths_culling_, GM_ADDR compensations_or_ones_, GM_ADDR dense_indices_, GM_ADDR cnt_flat_, GM_ADDR v_p_w_, GM_ADDR v_quats_, GM_ADDR v_scales_, GM_ADDR v_colors_, GM_ADDR v_opacities_, GM_ADDR dL_dvi_, GM_ADDR dL_dsigma_, GM_ADDR workspace, int dense_g, int slice_count, int batch_count, int camera_count, int max_compact_count, int chunk_stride, int total_tasks, int scalar_one, int feature2, int feature3, int feature4, int view_cols, int k_cols, int if_depth_correct, int image_width, int image_height, int color_dims) {
    TPipe pipe;
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
    SEvent<PIPE_MTE2, PIPE_V, false> _tmp_sevent_ready_ubin_1;
    SEvent<PIPE_V, PIPE_MTE2, true> _tmp_sevent_valid_ubin_2;
    SEvent<PIPE_MTE2, PIPE_V, false> _tmp_sevent_ready_ubin_0;
    SEvent<PIPE_V, PIPE_MTE2, true> _tmp_sevent_valid_ubin_1;
    SEvent<PIPE_MTE2, PIPE_V, false> _tmp_sevent_ready_ubin_2;
    SEvent<PIPE_V, PIPE_MTE2, true> _tmp_sevent_valid_ubin_0;
    GlobalTensor<float> means; means.SetGlobalBuffer((__gm__ float*) means_);
    GlobalTensor<float> quats; quats.SetGlobalBuffer((__gm__ float*) quats_);
    GlobalTensor<float> scales; scales.SetGlobalBuffer((__gm__ float*) scales_);
    GlobalTensor<float> conics; conics.SetGlobalBuffer((__gm__ float*) conics_);
    GlobalTensor<float> viewmats; viewmats.SetGlobalBuffer((__gm__ float*) viewmats_);
    GlobalTensor<float> ks; ks.SetGlobalBuffer((__gm__ float*) ks_);
    GlobalTensor<float> v_means2d_culling; v_means2d_culling.SetGlobalBuffer((__gm__ float*) v_means2d_culling_);
    GlobalTensor<float> v_depths_culling; v_depths_culling.SetGlobalBuffer((__gm__ float*) v_depths_culling_);
    GlobalTensor<float> v_conics_culling; v_conics_culling.SetGlobalBuffer((__gm__ float*) v_conics_culling_);
    GlobalTensor<float> v_colors_culling; v_colors_culling.SetGlobalBuffer((__gm__ float*) v_colors_culling_);
    GlobalTensor<float> v_opacities_culling; v_opacities_culling.SetGlobalBuffer((__gm__ float*) v_opacities_culling_);
    GlobalTensor<float> v_vi_culling; v_vi_culling.SetGlobalBuffer((__gm__ float*) v_vi_culling_);
    GlobalTensor<float> opacities_in_culling; opacities_in_culling.SetGlobalBuffer((__gm__ float*) opacities_in_culling_);
    GlobalTensor<float> vi_culling; vi_culling.SetGlobalBuffer((__gm__ float*) vi_culling_);
    GlobalTensor<float> sigma; sigma.SetGlobalBuffer((__gm__ float*) sigma_);
    GlobalTensor<float> depths_culling; depths_culling.SetGlobalBuffer((__gm__ float*) depths_culling_);
    GlobalTensor<float> compensations_or_ones; compensations_or_ones.SetGlobalBuffer((__gm__ float*) compensations_or_ones_);
    GlobalTensor<int> dense_indices; dense_indices.SetGlobalBuffer((__gm__ int*) dense_indices_);
    GlobalTensor<int> cnt_flat; cnt_flat.SetGlobalBuffer((__gm__ int*) cnt_flat_);
    GlobalTensor<float> v_p_w; v_p_w.SetGlobalBuffer((__gm__ float*) v_p_w_);
    GlobalTensor<float> v_quats; v_quats.SetGlobalBuffer((__gm__ float*) v_quats_);
    GlobalTensor<float> v_scales; v_scales.SetGlobalBuffer((__gm__ float*) v_scales_);
    GlobalTensor<float> v_colors; v_colors.SetGlobalBuffer((__gm__ float*) v_colors_);
    GlobalTensor<float> v_opacities; v_opacities.SetGlobalBuffer((__gm__ float*) v_opacities_);
    GlobalTensor<float> dL_dvi; dL_dvi.SetGlobalBuffer((__gm__ float*) dL_dvi_);
    GlobalTensor<float> dL_dsigma; dL_dsigma.SetGlobalBuffer((__gm__ float*) dL_dsigma_);
    // pipe_ptr->Reset();
    // OccupyMMTE1Events();
    // END: Auto inserted buffers/events/GMTensors
    // ----- User's codes start from here -----
    {
    LocalTensor<float> means_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 1024);
    LocalTensor<float> quats_ub = AllocateLocalTensor<TPosition::VECCALC, float>(4 * 1024);
    LocalTensor<float> scales_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 1024);
    LocalTensor<float> quat_cache_ub = AllocateLocalTensor<TPosition::VECCALC, float>(5 * 64);
    LocalTensor<float> rot_cache_ub = AllocateLocalTensor<TPosition::VECCALC, float>(9 * 64);
    LocalTensor<float> scale_sq_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 64);
    LocalTensor<float> conics_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 1024);
    LocalTensor<float> vconics_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 1024);
    LocalTensor<float> view_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 16);
    LocalTensor<float> k_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 16);
    LocalTensor<float> vmeans2d_ub = AllocateLocalTensor<TPosition::VECCALC, float>(2 * 1024);
    LocalTensor<float> vdepths_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> color_ub = AllocateLocalTensor<TPosition::VECCALC, float>(7 * 1024);
    LocalTensor<float> opacity_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> opacities_in_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> vi_in_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> vi_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> sigma_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1);
    LocalTensor<float> depths_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> compensation_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> accum_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 64);
    LocalTensor<float> proj_covar_ub = AllocateLocalTensor<TPosition::VECCALC, float>(13 * 64);
    LocalTensor<float> cam_covar_sym_ub = AllocateLocalTensor<TPosition::VECCALC, float>(6 * 64);
    LocalTensor<float> cam_covar_grad_ub = AllocateLocalTensor<TPosition::VECCALC, float>(6 * 64);
    LocalTensor<float> covar_accum_ub = AllocateLocalTensor<TPosition::VECCALC, float>(6 * 64);
    LocalTensor<float> vpw_stage_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 1024);
    LocalTensor<float> vquats_stage_ub = AllocateLocalTensor<TPosition::VECCALC, float>(4 * 1024);
    LocalTensor<float> vscales_stage_ub = AllocateLocalTensor<TPosition::VECCALC, float>(3 * 1024);
    LocalTensor<int> dense_indices_ub = AllocateLocalTensor<TPosition::VECCALC, int>(1 * 1024);
    int worker_count = 2*GetBlockNum();
    int worker_idx = GetBlockIdx();
    int tasks_per_worker = CeilDiv(total_tasks, worker_count);
    int task_begin = tasks_per_worker*worker_idx;
    int task_end = Min(task_begin + tasks_per_worker, total_tasks);
    int slice_begin = ((task_begin) / (chunk_stride));
    int slice_end = CeilDiv(task_end, chunk_stride);
    _tmp_sevent_valid_ubin_0.wait();
    GM2UBPAD(sigma_ub, sigma, 1, 4, 0, 0);
    PipeBarrier<PIPE_ALL>();
    _tmp_sevent_ready_ubin_0.set();
    _tmp_sevent_ready_ubin_0.wait();
    for (int slice_idx = slice_begin; slice_idx < slice_end; slice_idx += 1) {
        int slice_task_begin = chunk_stride*slice_idx;
        int chunk_end = Min(-slice_task_begin + task_end, chunk_stride);
        if ((Max(-slice_task_begin + task_begin, 0)) < chunk_end) {
            int batch_idx = ((slice_idx) / (camera_count));
            int base1 = batch_idx*dense_g;
            int slice_base1 = dense_g*slice_idx;
            int color_base = batch_idx*color_dims*dense_g;
            int color_base_ch1 = color_base + dense_g;
            int color_base_ch2 = color_base + 2*dense_g;
            int color_base_ch3 = color_base + 3*dense_g;
            int color_base_ch4 = color_base + 4*dense_g;
            int color_base_ch5 = color_base + 5*dense_g;
            int color_base_ch6 = color_base + 6*dense_g;
            int base3 = 3*batch_idx*dense_g;
            int base3_ch1 = base3 + dense_g;
            int base3_ch2 = base3 + 2*dense_g;
            int base4 = 4*batch_idx*dense_g;
            int base4_ch1 = base4 + dense_g;
            int base4_ch2 = base4 + 2*dense_g;
            int base4_ch3 = base4 + 3*dense_g;
            int compact_count = (int) cnt_flat[slice_idx].GetValue(0);
            _tmp_sevent_valid_ubin_1.wait();
            GM2UBPAD(view_ub, viewmats[slice_idx * view_cols], 1, 48, 0, 0);
            GM2UBPAD(k_ub, ks[slice_idx * k_cols], 1, 24, 0, 0);
            PipeBarrier<PIPE_ALL>();
            _tmp_sevent_ready_ubin_1.set();
            _tmp_sevent_ready_ubin_1.wait();
            for (int stage_chunk = Max(-slice_task_begin + task_begin, 0); stage_chunk < chunk_end; stage_chunk += 16) {
                int row0 = 64*stage_chunk;
                int owned_stage_cols = Min(1024, 64*chunk_end - 64*stage_chunk);
                int valid_cols = Min(owned_stage_cols, compact_count - row0);
                if (valid_cols > 0) {
                    _tmp_sevent_valid_ubin_2.wait();
                    GM2UBPAD(dense_indices_ub, dense_indices[slice_idx * max_compact_count + row0], 1, 4*valid_cols, 0, 0);
                    for (int feature = 0; feature < 3; feature += 1) {
                        GM2UBPAD(means_ub[feature * 1024], means[slice_idx * feature3 * dense_g + feature * dense_g + row0], 1, 4*valid_cols, 0, 0);
                        GM2UBPAD(scales_ub[feature * 1024], scales[slice_idx * feature3 * dense_g + feature * dense_g + row0], 1, 4*valid_cols, 0, 0);
                        GM2UBPAD(conics_ub[feature * 1024], conics[slice_idx * feature3 * dense_g + feature * dense_g + row0], 1, 4*valid_cols, 0, 0);
                        GM2UBPAD(vconics_ub[feature * 1024], v_conics_culling[slice_idx * feature3 * dense_g + feature * dense_g + row0], 1, 4*valid_cols, 0, 0);
                        GM2UBPAD(color_ub[feature * 1024], v_colors_culling[slice_idx * color_dims * dense_g + feature * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    }
                    for (int color_feature = 3; color_feature < 7; color_feature += 1) {
                        if (color_feature < color_dims) {
                            GM2UBPAD(color_ub[color_feature * 1024], v_colors_culling[slice_idx * color_dims * dense_g + color_feature * dense_g + row0], 1, 4*valid_cols, 0, 0);
                        }
                    }
                    for (int feature_1 = 0; feature_1 < 4; feature_1 += 1) {
                        GM2UBPAD(quats_ub[feature_1 * 1024], quats[slice_idx * feature4 * dense_g + feature_1 * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    }
                    for (int feature_2 = 0; feature_2 < 2; feature_2 += 1) {
                        GM2UBPAD(vmeans2d_ub[feature_2 * 1024], v_means2d_culling[slice_idx * feature2 * dense_g + feature_2 * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    }
                    GM2UBPAD(vdepths_ub, v_depths_culling[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    GM2UBPAD(opacity_ub, v_opacities_culling[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    GM2UBPAD(opacities_in_ub, opacities_in_culling[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    GM2UBPAD(vi_in_ub, vi_culling[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    GM2UBPAD(vi_ub, v_vi_culling[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    GM2UBPAD(depths_ub, depths_culling[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    GM2UBPAD(compensation_ub, compensations_or_ones[slice_idx * dense_g + row0], 1, 4*valid_cols, 0, 0);
                    int sub_valid_cols = Min(64, valid_cols);
                    _tmp_sevent_ready_ubin_2.set();
                    _tmp_sevent_ready_ubin_2.wait();
                    if (sub_valid_cols > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub).GetPhyAddr(), (__ubuf__ float*) (conics_ub).GetPhyAddr(), (__ubuf__ float*) (vconics_ub).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub).GetPhyAddr(), (__ubuf__ float*) (opacity_ub).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub).GetPhyAddr(), (__ubuf__ float*) (compensation_ub).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub).GetPhyAddr());
                    }
                    int sub_valid_cols_1 = Min(64, valid_cols - 64);
                    if (sub_valid_cols_1 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[64]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[64]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[64]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[64]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[64]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[64]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[64]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[64]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[64]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[64]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[64]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[64]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[64]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[64]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[64]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[64]).GetPhyAddr());
                    }
                    int sub_valid_cols_2 = Min(64, valid_cols - 128);
                    if (sub_valid_cols_2 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[128]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[128]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[128]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[128]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[128]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[128]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[128]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[128]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[128]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[128]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[128]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[128]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[128]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[128]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[128]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[128]).GetPhyAddr());
                    }
                    int sub_valid_cols_3 = Min(64, valid_cols - 192);
                    if (sub_valid_cols_3 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[192]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[192]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[192]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[192]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[192]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[192]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[192]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[192]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[192]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[192]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[192]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[192]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[192]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[192]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[192]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[192]).GetPhyAddr());
                    }
                    int sub_valid_cols_4 = Min(64, valid_cols - 256);
                    if (sub_valid_cols_4 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[256]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[256]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[256]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[256]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[256]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[256]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[256]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[256]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[256]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[256]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[256]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[256]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[256]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[256]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[256]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[256]).GetPhyAddr());
                    }
                    int sub_valid_cols_5 = Min(64, valid_cols - 320);
                    if (sub_valid_cols_5 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[320]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[320]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[320]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[320]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[320]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[320]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[320]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[320]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[320]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[320]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[320]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[320]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[320]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[320]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[320]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[320]).GetPhyAddr());
                    }
                    int sub_valid_cols_6 = Min(64, valid_cols - 384);
                    if (sub_valid_cols_6 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[384]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[384]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[384]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[384]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[384]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[384]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[384]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[384]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[384]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[384]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[384]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[384]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[384]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[384]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[384]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[384]).GetPhyAddr());
                    }
                    int sub_valid_cols_7 = Min(64, valid_cols - 448);
                    if (sub_valid_cols_7 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[448]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[448]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[448]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[448]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[448]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[448]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[448]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[448]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[448]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[448]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[448]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[448]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[448]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[448]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[448]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[448]).GetPhyAddr());
                    }
                    int sub_valid_cols_8 = Min(64, valid_cols - 512);
                    if (sub_valid_cols_8 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[512]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[512]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[512]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[512]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[512]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[512]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[512]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[512]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[512]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[512]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[512]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[512]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[512]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[512]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[512]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[512]).GetPhyAddr());
                    }
                    int sub_valid_cols_9 = Min(64, valid_cols - 576);
                    if (sub_valid_cols_9 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[576]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[576]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[576]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[576]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[576]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[576]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[576]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[576]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[576]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[576]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[576]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[576]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[576]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[576]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[576]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[576]).GetPhyAddr());
                    }
                    int sub_valid_cols_10 = Min(64, valid_cols - 640);
                    if (sub_valid_cols_10 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[640]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[640]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[640]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[640]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[640]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[640]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[640]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[640]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[640]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[640]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[640]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[640]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[640]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[640]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[640]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[640]).GetPhyAddr());
                    }
                    int sub_valid_cols_11 = Min(64, valid_cols - 704);
                    if (sub_valid_cols_11 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[704]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[704]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[704]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[704]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[704]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[704]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[704]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[704]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[704]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[704]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[704]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[704]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[704]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[704]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[704]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[704]).GetPhyAddr());
                    }
                    int sub_valid_cols_12 = Min(64, valid_cols - 768);
                    if (sub_valid_cols_12 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[768]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[768]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[768]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[768]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[768]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[768]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[768]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[768]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[768]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[768]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[768]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[768]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[768]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[768]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[768]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[768]).GetPhyAddr());
                    }
                    int sub_valid_cols_13 = Min(64, valid_cols - 832);
                    if (sub_valid_cols_13 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[832]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[832]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[832]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[832]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[832]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[832]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[832]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[832]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[832]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[832]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[832]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[832]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[832]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[832]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[832]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[832]).GetPhyAddr());
                    }
                    int sub_valid_cols_14 = Min(64, valid_cols - 896);
                    if (sub_valid_cols_14 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[896]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[896]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[896]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[896]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[896]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[896]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[896]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[896]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[896]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[896]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[896]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[896]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[896]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[896]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[896]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[896]).GetPhyAddr());
                    }
                    int sub_valid_cols_15 = Min(64, valid_cols - 960);
                    if (sub_valid_cols_15 > 0) {
                        _compute_quat_rot_scale_cache_vf((__ubuf__ float*) (quats_ub[960]).GetPhyAddr(), (__ubuf__ float*) (scales_ub[960]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr());
                        _compute_projection_covar_mean_intermediates_vf((__ubuf__ float*) (means_ub[960]).GetPhyAddr(), (__ubuf__ float*) (conics_ub[960]).GetPhyAddr(), (__ubuf__ float*) (vconics_ub[960]).GetPhyAddr(), (__ubuf__ float*) (vmeans2d_ub[960]).GetPhyAddr(), (__ubuf__ float*) (vdepths_ub[960]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub[960]).GetPhyAddr(), (__ubuf__ float*) (opacities_in_ub[960]).GetPhyAddr(), (__ubuf__ float*) (compensation_ub[960]).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub[960]).GetPhyAddr(), (__ubuf__ float*) (sigma_ub).GetPhyAddr(), (__ubuf__ float*) (depths_ub[960]).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), image_width, image_height, if_depth_correct);
                        PipeBarrier<PIPE_ALL>();
                        _accumulate_covar_direct_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_grad_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr());
                        PipeBarrier<PIPE_ALL>();
                        _finish_scale_quat_from_covar_cache_vf((__ubuf__ float*) (quat_cache_ub).GetPhyAddr(), (__ubuf__ float*) (rot_cache_ub).GetPhyAddr(), (__ubuf__ float*) (scales_ub[960]).GetPhyAddr(), (__ubuf__ float*) (scale_sq_ub).GetPhyAddr(), (__ubuf__ float*) (covar_accum_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[960]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[960]).GetPhyAddr());
                        _accumulate_covariance_mean_no_vr_from_intermediates_vf((__ubuf__ float*) (proj_covar_ub).GetPhyAddr(), (__ubuf__ float*) (cam_covar_sym_ub).GetPhyAddr(), (__ubuf__ float*) (view_ub).GetPhyAddr(), (__ubuf__ float*) (k_ub).GetPhyAddr(), (__ubuf__ float*) (accum_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[960]).GetPhyAddr());
                    }
                    PipeBarrier<PIPE_ALL>();
                    Simt::VF_CALL<_simt_accumulate_projection_outputs>(Simt::Dim3{256, 1, 1}, (__ubuf__ int*) (dense_indices_ub).GetPhyAddr(), (__ubuf__ float*) (color_ub).GetPhyAddr(), (__ubuf__ float*) (color_ub[1 * 1024]).GetPhyAddr(), (__ubuf__ float*) (color_ub[2 * 1024]).GetPhyAddr(), (__ubuf__ float*) (color_ub[3 * 1024]).GetPhyAddr(), (__ubuf__ float*) (color_ub[4 * 1024]).GetPhyAddr(), (__ubuf__ float*) (color_ub[5 * 1024]).GetPhyAddr(), (__ubuf__ float*) (color_ub[6 * 1024]).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[1 * 1024]).GetPhyAddr(), (__ubuf__ float*) (vpw_stage_ub[2 * 1024]).GetPhyAddr(), (__ubuf__ float*) (opacity_ub).GetPhyAddr(), (__ubuf__ float*) (vi_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[1 * 1024]).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[2 * 1024]).GetPhyAddr(), (__ubuf__ float*) (vquats_stage_ub[3 * 1024]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[1 * 1024]).GetPhyAddr(), (__ubuf__ float*) (vscales_stage_ub[2 * 1024]).GetPhyAddr(), (__gm__ float*) (v_colors).GetPhyAddr(), (__gm__ float*) (v_p_w).GetPhyAddr(), (__gm__ float*) (v_opacities).GetPhyAddr(), (__gm__ float*) (dL_dvi).GetPhyAddr(), (__gm__ float*) (dL_dsigma).GetPhyAddr(), (__gm__ float*) (v_quats).GetPhyAddr(), (__gm__ float*) (v_scales).GetPhyAddr(), base1, slice_base1, color_base, color_base_ch1, color_base_ch2, color_base_ch3, color_base_ch4, color_base_ch5, color_base_ch6, base3, base3_ch1, base3_ch2, base4, base4_ch1, base4_ch2, base4_ch3, valid_cols, color_dims, (__ubuf__ float*) (depths_ub).GetPhyAddr(), (__ubuf__ float*) (vi_in_ub).GetPhyAddr(), sigma.GetValue(0), if_depth_correct);
                    PipeBarrier<PIPE_ALL>();
                    _tmp_sevent_valid_ubin_2.set();
                }
            }
            _tmp_sevent_valid_ubin_1.set();
        }
    }
    _tmp_sevent_valid_ubin_0.set();
    }

    // Auto generated code. Readability is not guaranteed.
}
