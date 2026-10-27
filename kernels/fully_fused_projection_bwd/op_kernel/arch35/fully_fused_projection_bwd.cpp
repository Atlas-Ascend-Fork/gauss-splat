/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "tensorutils.h"
#include "fully_fused_projection_bwd_cube.h"
#include "fully_fused_projection_bwd_vec.h"


extern "C" __global__ __aicore__ void
fully_fused_projection_bwd(GM_ADDR means, GM_ADDR quats, GM_ADDR scales, GM_ADDR conics, GM_ADDR viewmats, GM_ADDR Ks,
                           GM_ADDR vMeans2d, GM_ADDR vDepths, GM_ADDR vConics, GM_ADDR vColorsCulling,
                           GM_ADDR vOpacitiesCulling, GM_ADDR filter, GM_ADDR compensations, GM_ADDR vViCulling, GM_ADDR opacities,
                           GM_ADDR vi, GM_ADDR sigma, GM_ADDR depths, GM_ADDR denseIndices, GM_ADDR cnt, GM_ADDR vPW,
                           GM_ADDR vQuats, GM_ADDR vScales, GM_ADDR vR, GM_ADDR vColors, GM_ADDR vOpacities, GM_ADDR vViewmats, GM_ADDR vKs, 
                           GM_ADDR dLdvi, GM_ADDR dLdsigma, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    PipeBarrier<PIPE_ALL>();
    int dense_g = tiling_data.dense_g;
    int slice_count = tiling_data.slice_count;
    int batch_count = tiling_data.batch_count;
    int camera_count = tiling_data.camera_count;
    int max_compact_count = tiling_data.max_compact_count;
    int chunk_stride = tiling_data.chunk_stride;
    int total_tasks = tiling_data.total_tasks;
    int scalar_one = tiling_data.scalar_one;
    int feature2 = tiling_data.feature2;
    int feature3 = tiling_data.feature3;
    int feature4 = tiling_data.feature4;
    int view_cols = tiling_data.view_cols;
    int k_cols = tiling_data.k_cols;
    const uint32_t SF_DEPTH_CORRECT = 1;
    int if_depth_correct = (tiling_data.support_features & SF_DEPTH_CORRECT);
    int image_width = tiling_data.image_width;
    int image_height = tiling_data.image_height;
    int color_dims = tiling_data.color_dims;
    if ASCEND_IS_AIC{
        fully_fused_projection_bwd_cube(means, quats, scales, conics, viewmats, Ks, vMeans2d, vDepths, vConics, vColorsCulling, vOpacitiesCulling, vViCulling, opacities, vi, sigma, depths, compensations, denseIndices, cnt, vPW, vQuats, vScales, vColors, vOpacities, dLdvi, dLdsigma, workspace, dense_g, slice_count, batch_count, camera_count, max_compact_count, chunk_stride, total_tasks, scalar_one, feature2, feature3, feature4, view_cols, k_cols, if_depth_correct, image_width, image_height, color_dims);
    }
    if ASCEND_IS_AIV{
        fully_fused_projection_bwd_vec(means, quats, scales, conics, viewmats, Ks, vMeans2d, vDepths, vConics, vColorsCulling, vOpacitiesCulling, vViCulling, opacities, vi, sigma, depths, compensations, denseIndices, cnt, vPW, vQuats, vScales, vColors, vOpacities, dLdvi, dLdsigma, workspace, dense_g, slice_count, batch_count, camera_count, max_compact_count, chunk_stride, total_tasks, scalar_one, feature2, feature3, feature4, view_cols, k_cols, if_depth_correct, image_width, image_height, color_dims);
    }

}
