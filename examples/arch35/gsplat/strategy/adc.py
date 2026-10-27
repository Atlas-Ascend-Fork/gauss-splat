# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/strategy/default.py
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class ADCStrategy(Strategy):
    """Robust Adaptive Density Control (original-3DGS-style), per a validated
    NPU recipe. Combines:
      1. visibility-aware gradient accumulation (accumulate only for visible GSs,
         average by visibility count)
      2. 3D-scale-aware clone/split (small -> clone, large -> split)
      3. screen-space pruning (track max screen-space radius, prune too-large GSs)
      4. hard cap on number of GSs (cap_max, default 4,000,000)
      5. periodic opacity reset (also resets the optimizer state for opacities)

    This differs from the per-pixel-error "Revising Densification" variant; it is the
    careful original-ADC that recovers most of the gap on the existing NPU kernels.

    Point (1) fixes the dominant NPU-specific bug: the rasterizer returns
    means2d/radii/grad COMPACTED (visible GSs packed in front, in original-index
    ascending order). Treating `where(radii>0)` as the gaussian id silently
    misattributes gradients. The A5 rasterizer supplies the original IDs; the
    packed `proj_filter` is used as a fallback. Points (2)-(5) are robustness/regularization on
    top, of which point (4) is load-bearing for outdoor/unbounded scenes: without a
    hard cap, fixing (1) alone lets the gaussian count overshoot the kernel's safe
    memory ceiling (~3.7M) and crash/degrade quality on large scenes, even though it
    is strictly the best option for naturally-bounded indoor scenes.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    # (3) Screen-space (2d-scale) refine window; 0 = off by default.
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 20_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    # (4) hard cap
    cap_max: int = 6_000_000
    # (5) opacity reset toggle. NOTE the recipe's "reset" is an UPPER CAP that prevents
    # opacity from saturating (anti-saturation), NOT the gsplat force-low-to-0.005 reset.
    # gsplat's value=0.01 clamps opacities DOWN to ~0.01 -> blackout -> NPU can't recover.
    # Here reset_value is the max allowed opacity (high -> gentle anti-saturation cap).
    reset_opacity: bool = True
    reset_value: float = 0.5
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d"] = "means2d"
    lc_wsr_enable=False

    def initialize_state(self, scene_scale: float = 1.0, lc_wsr_enable=False) -> Dict[str, Any]:
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale,
                 "radii": None}  # (3) always track screen-space radii
        self.lc_wsr_enable = lc_wsr_enable
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(self, params, optimizers, state, step, info):
        assert self.key_for_gradient in info, "means2d is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed: bool = False):
        if step >= self.refine_stop_iter:
            return
        self._update_state(params, state, info, packed=packed)

        if (step > self.refine_start_iter
                and step % self.refine_every == 0
                and step % self.reset_every >= self.pause_refine_after_reset):
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, info)
            n_prune = self._prune_gs(params, optimizers, state, step, info)
            # (4) hard cap: grow freely, then if N>cap_max remove the lowest-opacity excess
            n_cur = len(params["means"])
            if n_cur > self.cap_max:
                n_excess = n_cur - self.cap_max
                if self.lc_wsr_enable:
                    opa = info["opacities"].flatten()
                else:
                    opa = torch.sigmoid(params["opacities"].flatten())
                _, low_ids = torch.topk(opa, k=n_excess, largest=False)
                cap_mask = torch.zeros(n_cur, dtype=torch.bool, device=opa.device)
                cap_mask[low_ids] = True
                if self.lc_wsr_enable:
                    remove(params=params, optimizers=optimizers, state=state, mask=cap_mask, info=info)
                else:
                    remove(params=params, optimizers=optimizers, state=state, mask=cap_mask)
            if self.verbose:
                print(f"Step {step}: +{n_dupli} dup +{n_split} split -{n_prune} prune "
                      f"-> {len(params['means'])} GSs")
            state["grad2d"].zero_()
            state["count"].zero_()
            state["radii"].zero_()
            torch.npu.empty_cache()

        # (5) periodic opacity reset (reset_opa also zeros the opacity optimizer state)
        if self.reset_opacity and step % self.reset_every == 0 and step > 0 and self.lc_wsr_enable is False:
            # cap opacities at reset_value (prevents saturation; NOT a force-low reset)
            reset_opa(params=params, optimizers=optimizers, state=state,
                      value=self.reset_value)

    def _update_state(self, params, state, info, packed: bool = False):
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids",
                    self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."

        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_gaussian = len(list(params.values())[0])
        device = grads.device
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=device)
        if state["radii"] is None:
            state["radii"] = torch.zeros(n_gaussian, device=device)

        # (1) visibility-aware accumulation. CRITICAL: the NPU rasterizer returns
        # means2d/radii/grad COMPACTED (visible GSs packed in front, in original-index
        # ascending order). Using where(radii>0) as gaussian-ids is WRONG — those are
        # compacted positions, not original ids -> grads get attributed to the wrong GSs.
        # Map compacted gradients back to original IDs. A2/A3 may only provide
        # proj_filter, a little-endian bitfield over the original Gaussian count.
        gaussian_ids = info.get("gaussian_ids")
        pf = info.get("proj_filter", None)
        if gaussian_ids is not None and not packed:
            nvis = gaussian_ids.numel()
            sel = (info["radii"][:nvis] > 0.0).all(dim=-1)
            gs_ids = gaussian_ids.to(device=device, dtype=torch.long)[sel]
            grads = grads[:nvis][sel]
            radii = info["radii"][:nvis][sel].max(dim=-1).values
        elif pf is not None and not packed:
            import numpy as np
            n_gs = int(info.get("n_gaussians", state["grad2d"].numel()))
            bits = np.unpackbits(pf.reshape(-1).cpu().numpy(), bitorder="little")[:n_gs].astype(bool)
            vis = torch.from_numpy(bits).to(grads.device)
            gs_ids = vis.nonzero(as_tuple=True)[0]                      # true original ids
            sel = (info["radii"] > 0.0).all(dim=-1)                     # True in [0:nvis] (compacted)
            grads_sel = grads[sel]
            radii_sel = info["radii"][sel].max(dim=-1).values
            # proj_filter (projection-level) vs compacted (render-level) can differ by ~1;
            # truncate to the common prefix (order-preserving) to align 1:1.
            nvis = min(gs_ids.numel(), grads_sel.shape[0])
            gs_ids = gs_ids[:nvis]
            grads = grads_sel[:nvis]
            radii = radii_sel[:nvis]
        elif packed:
            gs_ids = info["gaussian_ids"]
            radii = info["radii"].max(dim=-1).values
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)
            gs_ids = torch.where(sel)[1]
            grads = grads[sel]
            radii = info["radii"][sel].max(dim=-1).values
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids,
                                  torch.ones_like(gs_ids, dtype=torch.float32))
        # (3) track max screen-space radius (normalized to [0,1])
        state["radii"].scatter_reduce_(
            0,
            gs_ids,
            radii.to(state["radii"].dtype) / float(max(info["width"], info["height"])),
            reduce="amax",
            include_self=True,
        )

        if self.lc_wsr_enable:
            info["opacities"] = info["opacities"].flatten()

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step, info) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)   # (1) average over visibility
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        # (2) 3D-scale-aware clone vs split
        is_small = (torch.exp(params["scales"]).max(dim=-1).values
                    <= self.grow_scale3d * state["scene_scale"])
        is_dupli = is_grad_high & is_small
        is_large = ~is_small
        is_split = is_grad_high & is_large
        # (3) screen-space split, off by default (refine_scale2d_stop_iter == 0).
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d

        n_dupli = int(is_dupli.sum().item())
        if n_dupli > 0:
            if self.lc_wsr_enable:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli, info=info)
            else:
                duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
        # new dupli'd GSs are appended; they must not be split this round
        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)])
        n_split = int(is_split.sum().item())
        if n_split > 0:
            if self.lc_wsr_enable:
                split(params=params, optimizers=optimizers, state=state,
                      mask=is_split, revised_opacity=self.revised_opacity, info=info)
            else:
                split(params=params, optimizers=optimizers, state=state,
                      mask=is_split, revised_opacity=self.revised_opacity)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state, step, info) -> int:
        # (4) remove lowest-opacity GSs
        if self.lc_wsr_enable:
            is_prune = state["radii"] > self.prune_scale2d
        else:
            is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
            # (3) screen-space pruning, off by default (refine_scale2d_stop_iter == 0).
            if step < self.refine_scale2d_stop_iter:
                is_prune = is_prune | (state["radii"] > self.prune_scale2d)
        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            if self.lc_wsr_enable:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune, info=info)
            else:
                remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune
