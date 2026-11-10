# coding: utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.autograd import Function

from gauss_splat.gsplat_c import sh_preprocess_fwd
from gauss_splat.gsplat_c import sh_preprocess_bwd


class ShPreprocess(Function):
    """Compute camera-to-Gaussian unit directions in channel-first layout."""

    @staticmethod
    def forward(ctx, camtoworlds: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
        rays_d = sh_preprocess_fwd(camtoworlds, means)
        ctx.save_for_backward(means, camtoworlds, rays_d)
        return rays_d

    @staticmethod
    def backward(ctx, v_rays_d: torch.Tensor):
        means, camtoworlds, rays_d = ctx.saved_tensors
        v_means = sh_preprocess_bwd(means, camtoworlds, rays_d, v_rays_d)
        v_camtoworlds = None
        if ctx.needs_input_grad[0]:
            v_camtoworlds = torch.zeros_like(camtoworlds)
            v_camtoworlds[0, :3, 3] = -v_means.sum(dim=1)
        return v_camtoworlds, v_means


sh_preprocess = ShPreprocess.apply
