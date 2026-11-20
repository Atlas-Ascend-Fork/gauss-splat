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

__simt_vf__ __launch_bounds__(32) inline void _publish_camera_grad_simt(__ubuf__ float* camera_grad_ub, __gm__ float* v_viewmats_gm, __gm__ float* v_ks_gm, int slice_idx){
    int row = Simt::GetThreadIdx<0>();
    if (row >= 16) {
        return;
    }
    float value = camera_grad_ub[row * 64];
    int base_view = slice_idx * 16;
    int base_ks = slice_idx * 9;
    if (row == 0) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 0, value);
    } else if (row == 1) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 1, value);
    } else if (row == 2) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 2, value);
    } else if (row == 3) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 4, value);
    } else if (row == 4) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 5, value);
    } else if (row == 5) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 6, value);
    } else if (row == 6) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 8, value);
    } else if (row == 7) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 9, value);
    } else if (row == 8) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 10, value);
    } else if (row == 9) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 3, value);
    } else if (row == 10) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 7, value);
    } else if (row == 11) {
        Simt::AtomicAdd(v_viewmats_gm + base_view + 11, value);
    } else if (row == 12) {
        Simt::AtomicAdd(v_ks_gm + base_ks + 0, value);
    } else if (row == 13) {
        Simt::AtomicAdd(v_ks_gm + base_ks + 4, value);
    } else if (row == 14) {
        Simt::AtomicAdd(v_ks_gm + base_ks + 2, value);
    } else {
        Simt::AtomicAdd(v_ks_gm + base_ks + 5, value);
    }
}

__aicore__ inline void _publish_camera_grad(__ubuf__ float* camera_grad_ub, __gm__ float* v_viewmats_gm, __gm__ float* v_ks_gm, int slice_idx){
    Simt::VF_CALL<_publish_camera_grad_simt>(Simt::Dim3{32, 1, 1}, camera_grad_ub, v_viewmats_gm, v_ks_gm, slice_idx);
}
