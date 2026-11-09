# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import tyro
import torch
import torch_npu

from gsplat.distributed import cli
from rasterization.config import Config
from rasterization.runner import Runner


def migrate_legacy_splats(splats):
    """Migrate legacy checkpoint keys to the current single-parameter schema.

    - sh0/shN        -> sh          (old 3DGS split SH)
    - features_do/ro -> features_sh (old CBS appearance features)
    """
    if "sh" not in splats and "sh0" in splats and "shN" in splats:
        splats["sh"] = torch.cat([splats.pop("sh0"), splats.pop("shN")], dim=1)
    if (
        "features_sh" not in splats
        and "features_do" in splats
        and "features_ro" in splats
    ):
        # features_do: [N,1,3], features_ro: [N,K-1,3] -> features_sh: [N,K,3]
        splats["features_sh"] = torch.cat(
            [splats.pop("features_do"), splats.pop("features_ro")], dim=1
        )
    return splats


def main(local_rank: int, world_rank, world_size: int, cfg: Config):

    runner = Runner(local_rank, world_rank, world_size, cfg)
    torch_npu.npu.synchronize()

    # ===== [Profiling 修改] 开始：端到端耗时采集（对齐 cbs_MetaGaussianSplat） =====
    # Time span of `fn` (device-side elapsed time between two NPU events, in seconds).
    def time_span(fn):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        ret = fn()
        end.record()
        torch_npu.npu.synchronize()
        return ret, start.elapsed_time(end) / 1000.0

    # e2e time logger (train / infer / total + per-step/per-image averages).
    def log_e2e_time(timings, render_times, num_train_steps, num_val_images):
        if world_rank != 0 or not timings:
            return
        train_t = timings.get("train", 0.0)
        infer_t = timings.get("infer", 0.0)
        total_t = train_t + infer_t
        train_per_step_ms = train_t / num_train_steps * 1000.0 if num_train_steps > 0 else 0.0
        infer_render_t = render_times.get("infer", 0.0)
        infer_per_image_ms = infer_render_t / num_val_images * 1000.0 if num_val_images > 0 else 0.0
        print(
            f"[E2E] train={train_t:.3f}s infer={infer_t:.3f}s total={total_t:.3f}s "
            f"train_per_step={train_per_step_ms:.3f}ms infer_per_image={infer_per_image_ms:.3f}ms"
        )
        record = {
            "train_e2e_time_s": train_t,
            "infer_e2e_time_s": infer_t,
            "total_e2e_time_s": total_t,
            "train_per_step_ms": train_per_step_ms,
            "infer_per_image_ms": infer_per_image_ms,
        }
        try:
            with open(f"{cfg.result_dir}/e2e_time.json", "w") as f:
                json.dump(record, f, indent=2)
        except Exception as e:
            print(f"[E2E] failed to dump e2e timing: {e}")

    timings = {}           # total device-side time per phase (seconds)
    render_times = {}      # pure render wall-clock time per phase (seconds)
    # ===== [Profiling 修改] 结束 =====

    if cfg.ckpt is not None:
        from rasterization.utils import to_channel_first

        # load checkpoint splats into the model
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        # Checkpoints are always saved in N-first [N,D]/[N,K,3] format.
        # Backward compatibility for older sh0/shN and features_do/ro schemas.
        for ckpt in ckpts:
            migrate_legacy_splats(ckpt["splats"])

        # Determine checkpoint N (from means) and concatenate along N (dim=0).
        ckpt_N = ckpts[0]["splats"]["means"].shape[0]
        loaded = {}
        current_layout_keys = set()
        for k in runner.splats.keys():
            if k in ckpts[0]["splats"]:
                loaded[k] = torch.cat([ckpt["splats"][k] for ckpt in ckpts], dim=0)
            else:
                # CBS-specific keys (features_sh, sigma) absent from 3DGS
                # checkpoints — re-initialize to match checkpoint N.
                init_N = runner.splats[k].shape[0]
                if init_N != ckpt_N and k == "features_sh":
                    C0 = 0.28209479177387814
                    camera_sh_degree = 4
                    K = (camera_sh_degree + 1) ** 2
                    new_data = torch.zeros((ckpt_N, K, 3), dtype=torch.float32)
                    new_data[:, 0, :] = 0.1 * torch.ones(
                        (ckpt_N, 3), dtype=torch.float32
                    ) / C0
                    loaded[k] = new_data
                    print(f"  [ckpt] key '{k}' resized {init_N} -> {ckpt_N}",
                          flush=True)
                else:
                    # Already in the runner's current layout; do not convert again.
                    loaded[k] = runner.splats[k].data.detach().clone()
                    current_layout_keys.add(k)
                    print(f"  [ckpt] key '{k}' not in checkpoint, using init value",
                          flush=True)
        if cfg.is_channel_first:
            for k in list(loaded.keys()):
                if k not in current_layout_keys:
                    loaded[k] = to_channel_first({k: loaded[k]})[k]
        for k, v in loaded.items():
            # Assign .data (not a new Parameter) so registered gradient hooks
            # (e.g. the SH high-order /20 hook) survive checkpoint loading.
            runner.splats[k].data = v.to(runner.device)
        step = ckpts[0]["step"]
        if cfg.perf_steps > 0:
            # perf mode: enable grad on splats, run forward+backward
            # iterations to trigger backward kernels for msprof collection.
            for k in runner.splats:
                runner.splats[k].requires_grad_(True)
            # ===== [Profiling 修改] 开始：包裹 perf_backward 计时 =====
            _, timings["train"] = time_span(
                lambda: runner.perf_backward(num_steps=cfg.perf_steps)
            )
            # ===== [Profiling 修改] 结束 =====
        elif cfg.perf_steps < 0:
            # forward-only perf mode: run N forward iterations (no backward)
            # ===== [Profiling 修改] 开始：包裹 perf_forward 计时 =====
            _, timings["train"] = time_span(
                lambda: runner.perf_forward(num_steps=-cfg.perf_steps)
            )
            # ===== [Profiling 修改] 结束 =====
        else:
            # run eval only (forward under no_grad)
            # ===== [Profiling 修改] 开始：包裹 eval 计时 =====
            ret, timings["infer"] = time_span(lambda: runner.eval(step=step))
            render_times["infer"] = ret[0] if ret else 0.0
            # ===== [Profiling 修改] 结束 =====
    else:
        # ===== [Profiling 修改] 开始：包裹 train + eval 计时 =====
        _, timings["train"] = time_span(runner.train)
        # Standalone inference (eval) speed test after training.
        ret, timings["infer"] = time_span(
            lambda: runner.eval(step=cfg.max_steps, stage="test")
        )
        render_times["infer"] = ret[0] if ret else 0.0
        # ===== [Profiling 修改] 结束 =====

    # ===== [Profiling 修改] 开始：汇总输出 e2e 时间 =====
    num_train_steps = cfg.max_steps if cfg.ckpt is None else abs(cfg.perf_steps)
    num_val_images = len(runner.valset)
    log_e2e_time(timings, render_times, num_train_steps, num_val_images)
    # ===== [Profiling 修改] 结束 =====


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single NPU training
    python -m train --data_dir data/360_v2/garden --result_dir results/garden
    ```

    ```bash
    # Perf profiling (forward+backward)
    python -m train --ckpt checkpoints/ckpt_31_rank0_20w.pt --perf_steps 10 --render_backend 3dgs
    ```

    NOTE: E2E timing is always enabled and always printed after train/infer.
    """

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)