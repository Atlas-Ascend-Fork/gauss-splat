/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FullyFusedProjectionBwdTilingData)
  TILING_DATA_FIELD_DEF(int32_t, dense_g);
  TILING_DATA_FIELD_DEF(int32_t, slice_count);
  TILING_DATA_FIELD_DEF(int32_t, batch_count);
  TILING_DATA_FIELD_DEF(int32_t, camera_count);
  TILING_DATA_FIELD_DEF(int32_t, max_compact_count);
  TILING_DATA_FIELD_DEF(int32_t, chunk_stride);
  TILING_DATA_FIELD_DEF(int32_t, total_tasks);
  TILING_DATA_FIELD_DEF(int32_t, scalar_one);
  TILING_DATA_FIELD_DEF(int32_t, feature2);
  TILING_DATA_FIELD_DEF(int32_t, feature3);
  TILING_DATA_FIELD_DEF(int32_t, feature4);
  TILING_DATA_FIELD_DEF(int32_t, view_cols);
  TILING_DATA_FIELD_DEF(int32_t, k_cols);
  TILING_DATA_FIELD_DEF(int32_t, support_features);
  TILING_DATA_FIELD_DEF(int32_t, image_width);
  TILING_DATA_FIELD_DEF(int32_t, image_height);
  TILING_DATA_FIELD_DEF(int32_t, color_dims);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FullyFusedProjectionBwd, FullyFusedProjectionBwdTilingData)
}
