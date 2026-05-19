# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

__all__ = [
    "projection_three_dims_gaussian_fused",
    "calc_render",
    "gaussian_sort",
    "get_render_schedule_cpp",
    "spherical_harmonics",
    "flash_gaussian_build_mask",
    "gaussian_filter",
    "Rasterizer",
]

import os
import torch
from gauss_splat.gsplat_c import _init_op_api_so_path

from .ops.projection_three_dims_gaussian_fused import projection_three_dims_gaussian_fused
from .ops.calc_render import calc_render
from .ops.gaussian_sort import gaussian_sort
from .ops.get_render_schedule import get_render_schedule_cpp
from .ops.spherical_harmonics import spherical_harmonics
from .ops.flash_gaussian_build_mask import flash_gaussian_build_mask
from .ops.gaussian_filter import gaussian_filter
from .ops.rendering import Rasterizer


def _set_env():
    gauss_splat_root = os.path.dirname(os.path.abspath(__file__))
    gauss_splat_opp_path = os.path.join(gauss_splat_root, "packages", "vendors", "customize")
    ascend_custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    ascend_custom_opp_path = (
        gauss_splat_opp_path
        if not ascend_custom_opp_path
        else gauss_splat_opp_path + ":" + ascend_custom_opp_path
    )
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ascend_custom_opp_path
    gauss_splat_op_api_so_path = os.path.join(gauss_splat_opp_path, "op_api", "lib", "libcust_opapi.so")
    _init_op_api_so_path(gauss_splat_op_api_so_path)

_set_env()