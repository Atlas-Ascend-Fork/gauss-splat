# coding=utf-8
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import gauss_splat.gsplat_c as _C

# Must match MAX_TENSOR_NUM in kernels/fused_adam_multi_lr/op_host/arch35/fused_adam_multi_lr.cpp
_MAX_TENSORS_PER_CALL = 64


def fused_adam_multi_lr_step_batch(
    params: list,
    grads: list,
    exp_avgs: list,
    exp_avg_sqs: list,
    lrs: list,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
):
    """Batch FusedAdamMultiLr step: process multiple param tensors in a single kernel launch.

    Each tensor can have a different lr. All tensors must share the same
    beta1/beta2/eps/weight_decay/step.
    """
    _C.fused_adam_multi_lr(params, grads, exp_avgs, exp_avg_sqs, lrs, beta1, beta2, eps, weight_decay, step)


class NpuFusedAdamMultiLrOptimizer(torch.optim.Optimizer):
    """NPU fused Adam optimizer, drop-in replacement for torch.optim.Adam.

    Supports:
    - Multiple param_groups with independent lr/beta/eps/weight_decay
    - exp_avg / exp_avg_sq state management
    - bias_correction (always on, matching torch.optim.Adam default)
    - weight_decay (L2 regularization, decoupled=False)
    - Compatible with torch.optim.lr_scheduler (LambdaLR etc.)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, **kwargs):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for bg in self._collect_batch_groups():
                self._apply_batch_group(bg)

        return loss

    def _get_state(self, param):
        for k, v in self.state.items():
            if k is param:
                return v
        state = {
            "exp_avg": torch.zeros_like(param, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.zeros_like(param, memory_format=torch.preserve_format),
            "step": 0,
        }
        self.state[param] = state
        return state

    def _find_or_create_group(self, groups, group, step):
        # Signature excludes lr — different lr values can be batched together.
        # step is included so params at different bias-correction steps are
        # never merged into one launch (the kernel applies a single step).
        sig = (group["betas"][0], group["betas"][1], group["eps"], group["weight_decay"], step)
        for bg in groups:
            if bg["sig"] == sig:
                return bg
        bg = {
            "sig": sig,
            "beta1": group["betas"][0], "beta2": group["betas"][1],
            "eps": group["eps"], "weight_decay": group["weight_decay"],
            "step": step,
            "params": [], "grads": [],
            "exp_avgs": [], "exp_avg_sqs": [],
            "lrs": [],
        }
        groups.append(bg)
        return bg

    def _collect_batch_groups(self):
        # Collect all params with grads across all param_groups,
        # grouping by beta/eps/weight_decay signature so we can batch
        # tensors that share the same betas/eps/weight_decay.
        # lr is per-tensor (each param keeps its group's lr).
        groups = []
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("NpuFusedAdamMultiLrOptimizer does not support sparse gradients")

                state = self._get_state(param)
                state["step"] += 1

                bg = self._find_or_create_group(groups, group, state["step"])
                bg["params"].append(param)
                bg["grads"].append(param.grad)
                bg["exp_avgs"].append(state["exp_avg"])
                bg["exp_avg_sqs"].append(state["exp_avg_sq"])
                bg["lrs"].append(group["lr"])
        return groups

    def _apply_batch_group(self, bg):
        if not bg["params"]:
            return
        # The kernel accepts at most MAX_TENSOR_NUM tensors per launch; split
        # oversized groups into consecutive chunks. Equivalent math: the op
        # updates each tensor independently, and hp signature is shared.
        for start in range(0, len(bg["params"]), _MAX_TENSORS_PER_CALL):
            end = start + _MAX_TENSORS_PER_CALL
            fused_adam_multi_lr_step_batch(
                bg["params"][start:end], bg["grads"][start:end],
                bg["exp_avgs"][start:end], bg["exp_avg_sqs"][start:end],
                bg["lrs"][start:end], bg["beta1"], bg["beta2"], bg["eps"],
                bg["weight_decay"], bg["step"],
            )
