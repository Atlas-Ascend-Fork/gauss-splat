/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GAUSSIAN_FILTER_TILING_H
#define GAUSSIAN_FILTER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GaussianFilterTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, batchNum);
TILING_DATA_FIELD_DEF(int64_t, cameraNum);
TILING_DATA_FIELD_DEF(int64_t, gaussNum);
TILING_DATA_FIELD_DEF(int64_t, width);
TILING_DATA_FIELD_DEF(int64_t, height);
TILING_DATA_FIELD_DEF(float, nearPlane);
TILING_DATA_FIELD_DEF(float, farPlane);
TILING_DATA_FIELD_DEF(int64_t, blockLength);
TILING_DATA_FIELD_DEF(int64_t, lastcoreNum);
TILING_DATA_FIELD_DEF(int64_t, perloopNum);
TILING_DATA_FIELD_DEF(int64_t, hasCompensations);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GaussianFilter, GaussianFilterTilingData)
} // namespace optiling
#endif // GAUSSIAN_FILTER_TILING_H