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
#include "gaussian_filter_cube.h"
#include "gaussian_filter_vec.h"

extern "C" __global__ __aicore__ void
gaussian_filter(GM_ADDR means, GM_ADDR colors, GM_ADDR det, GM_ADDR opacities, GM_ADDR means2d, GM_ADDR depths,
                GM_ADDR radius, GM_ADDR conics, GM_ADDR covars2d, GM_ADDR compensations, GM_ADDR quats,
                GM_ADDR scales, GM_ADDR vi, GM_ADDR sigma, GM_ADDR meansCulling, GM_ADDR colorsCulling,
                GM_ADDR means2dCulling, GM_ADDR depthsCulling, GM_ADDR radiusCulling, GM_ADDR covars2dCulling,
                GM_ADDR conicsCulling, GM_ADDR opacitiesCulling, GM_ADDR quatsCulling, GM_ADDR scalesCulling, 
                GM_ADDR viCulling, GM_ADDR compensationsCulling, GM_ADDR filter, GM_ADDR cnt, GM_ADDR filterFloat,
                GM_ADDR denseIndices, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    PipeBarrier<PIPE_ALL>();
    int B = tiling_data.batchNum;
    int C = tiling_data.cameraNum;
    int N = tiling_data.gaussNum;
    float width = tiling_data.width;
    float height = tiling_data.height;
    float near_plane = tiling_data.nearPlane;
    float far_plane = tiling_data.farPlane;
    const uint32_t SF_DEPTH_CORRECT = 1;
    float if_depth_correct = (float)(tiling_data.supportFeatures & SF_DEPTH_CORRECT);
    int color_dims = tiling_data.colorDims;
    int sigma_len = tiling_data.sigmaLen;
    int M = tiling_data.M;
    if ASCEND_IS_AIC{
        gaussian_filter_cube(means, colors, quats, scales, det, depths, means2d, radius, conics, covars2d, opacities, compensations, vi, sigma, meansCulling, colorsCulling, quatsCulling, scalesCulling, means2dCulling, depthsCulling, radiusCulling, covars2dCulling, conicsCulling, opacitiesCulling, viCulling, compensationsCulling, filterFloat, cnt, denseIndices, filter, workspace, B, C, N, width, height, near_plane, far_plane, if_depth_correct, color_dims, sigma_len, M);
    }
    if ASCEND_IS_AIV{
        gaussian_filter_vec(means, colors, quats, scales, det, depths, means2d, radius, conics, covars2d, opacities, compensations, vi, sigma, meansCulling, colorsCulling, quatsCulling, scalesCulling, means2dCulling, depthsCulling, radiusCulling, covars2dCulling, conicsCulling, opacitiesCulling, viCulling, compensationsCulling, filterFloat, cnt, denseIndices, filter, workspace, B, C, N, width, height, near_plane, far_plane, if_depth_correct, color_dims, sigma_len, M);
    }

}
