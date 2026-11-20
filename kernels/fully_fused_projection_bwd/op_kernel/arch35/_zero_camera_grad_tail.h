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

__simt_vf__ __launch_bounds__(32) inline void _zero_camera_grad_tail_simt(__ubuf__ float* camera_grad_ub, int valid_cols){
    for (int idx = Simt::GetThreadIdx<0>(); idx < 16 * 64; idx += Simt::GetThreadNum<0>()) {
        int col = idx & 63;
        if (col >= valid_cols) {
            camera_grad_ub[idx] = 0.0f;
        }
    }
}

__aicore__ inline void _zero_camera_grad_tail(__ubuf__ float* camera_grad_ub, int valid_cols){
    if (valid_cols < 64) {
        Simt::VF_CALL<_zero_camera_grad_tail_simt>(Simt::Dim3{32, 1, 1}, camera_grad_ub, valid_cols);
    }
}
