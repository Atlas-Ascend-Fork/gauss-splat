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

__aicore__ inline void fully_fused_projection_bwd_cube(GM_ADDR means_, GM_ADDR quats_, GM_ADDR scales_, GM_ADDR conics_, GM_ADDR viewmats_, GM_ADDR ks_, GM_ADDR v_means2d_culling_, GM_ADDR v_depths_culling_, GM_ADDR v_conics_culling_, GM_ADDR v_colors_culling_, GM_ADDR v_opacities_culling_, GM_ADDR v_vi_culling_, GM_ADDR opacities_in_culling_, GM_ADDR vi_culling_, GM_ADDR sigma_, GM_ADDR depths_culling_, GM_ADDR compensations_or_ones_, GM_ADDR dense_indices_, GM_ADDR cnt_flat_, GM_ADDR v_p_w_, GM_ADDR v_quats_, GM_ADDR v_scales_, GM_ADDR v_colors_, GM_ADDR v_opacities_, GM_ADDR dL_dvi_, GM_ADDR dL_dsigma_, GM_ADDR workspace, int dense_g, int slice_count, int batch_count, int camera_count, int max_compact_count, int chunk_stride, int total_tasks, int scalar_one, int feature2, int feature3, int feature4, int view_cols, int k_cols, int if_depth_correct, int image_width, int image_height, int color_dims) {
    TPipe pipe;
    TPipe* pipe_ptr = GetTPipePtr();
    int _offset = 0;
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
    LocalTensor<float> sigma_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1);
    LocalTensor<float> depths_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> dvi_stage_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
    LocalTensor<float> dsigma_stage_ub = AllocateLocalTensor<TPosition::VECCALC, float>(1 * 1024);
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
    }

    // Auto generated code. Readability is not guaranteed.
}
