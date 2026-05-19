# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from gauss_splat.gsplat_c import gaussian_filter as _gaussian_filter

# pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
def gaussian_filter(means,
            colors,
            det,
            opacities,
            means2d,
            depths,
            radius,
            conics,
            covars2d,
            compensations,
            width,
            height,
            near_plane,
            far_plane):
    means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
        covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt = _gaussian_filter(
        means,
        colors,
        det,
        opacities,
        means2d,
        depths,
        radius,
        conics,
        covars2d,
        compensations,
        width,
        height,
        near_plane,
        far_plane)
    return means_culling, colors_culling, means2d_culling, depths_culling, radius_culling,\
           covars2d_culling, conics_culling, opacities_culling, proj_filter, cnt
