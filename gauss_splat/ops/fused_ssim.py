# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from math import exp, isfinite

import torch
from torch.autograd import Function

from gauss_splat import gsplat_c

# the fused_ssim_forward kernel takes the 1D window zero-padded to this length
_WEIGHT_PAD_LEN = 16


def _gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    g = torch.tensor([exp(-(x - window_size // 2) ** 2 / (2.0 * sigma * sigma))
                       for x in range(window_size)], dtype=torch.float32)
    return g / g.sum()


def make_ssim_weight(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """1D Gaussian window, zero-padded to length 16 (the kernel's fixed contract)."""
    if not isinstance(window_size, int) or window_size not in (7, 11, 15):
        raise ValueError("window_size must be one of 7, 11, or 15")
    if not isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    w = torch.zeros(_WEIGHT_PAD_LEN, dtype=torch.float32)
    w[:window_size] = _gaussian_1d(window_size, sigma)
    return w.to(device=device, dtype=dtype)


class FusedSSIM(Function):
    """Fused SSIM with kernel-computed intermediates and image gradients."""

    @staticmethod
    def forward(ctx, img1, img2, weight, window_size):
        ssim_map, mu1, mu2, S11, S22, S12 = \
            gsplat_c.fused_ssim_forward(img1, img2, weight, window_size)
        # Stash everything backward needs. img1/img2 are tiny enough vs the
        # 5 saved maps that keeping them as-is here is fine.
        ctx.save_for_backward(img1, img2, weight, mu1, mu2, S11, S22, S12)
        ctx.window_size = window_size
        return ssim_map

    @staticmethod
    def backward(ctx, grad_ssim_map):
        img1, img2, weight, mu1, mu2, S11, S22, S12 = ctx.saved_tensors
        ws = ctx.window_size
        need_g2 = ctx.needs_input_grad[1] if ctx.needs_input_grad is not None else False

        grad_img1, grad_img2_full = gsplat_c.fused_ssim_backward(
            grad_ssim_map.contiguous(), img1.contiguous(), img2.contiguous(), weight,
            mu1, mu2, S11, S22, S12, ws, 1 if need_g2 else 0)
        grad_img2 = grad_img2_full if need_g2 else None
        return grad_img1, grad_img2, None, None


def fused_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7,
               sigma: float = 0.75, size_average: bool = True,
               reduction: str = None) -> torch.Tensor:
    """Fused SSIM. img1/img2: [B,C,H,W]. Drop-in for rasterization.utils.SSIM.ssim
    (window_size=7, sigma=1.5/2=0.75). reduction overrides size_average when set:
    'mean' -> scalar, 'sum' -> scalar sum, 'none' -> per-image [B]. Default keeps
    the size_average behaviour (mean if True else per-image)."""
    if img1.dtype != torch.float32 or img2.dtype != torch.float32:
        raise ValueError("fused_ssim requires float32 images")
    if reduction not in (None, "mean", "sum", "none"):
        raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
    weight = make_ssim_weight(window_size, sigma, img1.device, img1.dtype)
    ssim_map = FusedSSIM.apply(img1.contiguous(), img2.contiguous(), weight, window_size)
    if reduction is None:
        reduction = "mean" if size_average else "none"
    if reduction == "mean":
        return ssim_map.mean()
    if reduction == "sum":
        return ssim_map.sum()
    if reduction == "none":
        return ssim_map.mean(1).mean(1).mean(1)
    raise ValueError(f"reduction must be mean/sum/none, got {reduction!r}")
