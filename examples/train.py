# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import tyro
import torch

from gsplat.distributed import cli
from rasterization.config import Config
from rasterization.runner import Runner


def main(local_rank: int, world_rank, world_size: int, cfg: Config):

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
    else:
        runner.train()


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single NPU training
    python -m train --data_dir data/360_v2/garden --result_dir results/garden
    """

    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)