# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import gauss_splat.gsplat_c as _C


def _validate_bg_color(bg_color):
    """校验可选背景色输入；None 表示关闭。"""
    if bg_color is None:
        return
    if not torch.is_tensor(bg_color):
        raise ValueError("bg_color must be a torch.Tensor")
    if bg_color.dtype != torch.float32:
        raise ValueError(f"bg_color must be float32, got {bg_color.dtype}")
    if bg_color.dim() != 1 or bg_color.numel() != 3:
        raise ValueError(f"bg_color must have shape [3], got {tuple(bg_color.shape)}")


class CalcRender(torch.autograd.Function):
    """Render using the backend selected when the extension was built."""

    @staticmethod
    def forward(ctx, means, conic0s, conic1s, conic2s, opacities, colors,
                depths, tile_coords, offsets, sorted_gs_ids, bg_color=None):
        ctx.has_depth = depths is not None
        ctx.render_method = getattr(_C, "sf_render_method", "clip_gsids")
        if ctx.render_method not in ("clip_lastids", "clip_gsids"):
            raise RuntimeError(f"Unknown render method: {ctx.render_method}")
        _validate_bg_color(bg_color)
        if bg_color is not None:
            bg_color = bg_color.contiguous()
        sorted_gs_ids = sorted_gs_ids.to(torch.int64)
        gs = torch.cat([
            means, conic0s.unsqueeze(0), conic1s.unsqueeze(0),
            conic2s.unsqueeze(0), opacities.unsqueeze(0), colors,
            depths if ctx.has_depth else colors[0:1],
        ], dim=0).T.contiguous()

        ctx.has_bg = bg_color is not None
        if ctx.render_method == "clip_lastids":
            color, depth, last_cumsum, error, last_ids = _C.calc_render_fwd_clip_last_ids(
                gs, tile_coords, offsets, sorted_gs_ids, bg_color)
            saved = (gs, tile_coords, offsets, sorted_gs_ids, last_cumsum, error, last_ids)
            if ctx.has_bg:
                ctx.save_for_backward(*saved, bg_color)
            else:
                ctx.save_for_backward(*saved)
        else:
            if bg_color is not None:
                raise ValueError("bg_color is only supported by the A5 clip_lastids render backend")
            color, depth, last_cumsum, error, gs_clip, alpha_clip = (
                _C.calc_render_fwd_double_clip_gsids(gs, tile_coords, offsets, sorted_gs_ids))
            ctx.save_for_backward(gs, tile_coords, offsets, sorted_gs_ids,
                last_cumsum, error, gs_clip, alpha_clip)
            ctx.has_bg = False

        return (color, depth) if ctx.has_depth else color

    @staticmethod
    def backward(ctx, *v_args):
        v_color = v_args[0].contiguous()
        # clip lastids forward returns depth as [T, P]; the clip lastids backward contract
        # expects vDepth with shape [1, T, P].
        if ctx.has_depth:
            v_depth = (v_args[1] if v_args[1].dim() == 3 else v_args[1].unsqueeze(0)).contiguous()
        else:
            v_depth = torch.zeros_like(v_color[0:1])
        
        v_gs = None
        v_bg = None
        if ctx.render_method == "clip_lastids":
            saved = ctx.saved_tensors
            gs, tile_coords, offsets, sorted_gs_ids, last_cumsum, error, last_ids = saved[:7]
            bg_color = saved[7] if ctx.has_bg else None
            v_gs = _C.calc_render_bwd_clip_last_ids(v_color, v_depth, last_cumsum, error, gs,
                        tile_coords, offsets, sorted_gs_ids, last_ids, bg_color)
            if bg_color is not None and ctx.needs_input_grad[10]:
                v_bg = (v_color * torch.exp(last_cumsum).unsqueeze(0)).sum(dim=(1, 2))
        else:
            gs, tile_coords, offsets, sorted_gs_ids, last_cumsum, error, gs_clip,  alpha_clip = ctx.saved_tensors
            v_gs = _C.calc_render_bwd_var_clip_gsids(v_color, v_depth, last_cumsum, error, gs,
                        tile_coords, offsets, sorted_gs_ids, gs_clip, alpha_clip)

        # 返回值个数必须等于 forward 入参个数（新增 bg_color 后为 11）。
        return (v_gs[:, :2].T, v_gs[:, 2], v_gs[:, 3], v_gs[:, 4],
                v_gs[:, 5], v_gs[:, 6:9].T,
                v_gs[:, 9:].T if ctx.has_depth else None,
                None, None, None, v_bg)


calc_render = CalcRender.apply
