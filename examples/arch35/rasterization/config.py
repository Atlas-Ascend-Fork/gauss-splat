# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from dataclasses import dataclass, field
from typing import List, Optional
from typing_extensions import Literal

from gsplat.strategy import ADCStrategy

@dataclass
class Config:
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"

    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # tile size: 32 or 64
    tile_size: int = 32
    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [500, 4800, 7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [500, 4800, 7_000, 30_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [4800, 7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 1_000_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Enable GS densification
    densify: bool = False

    # Strategy for GS densification(only used when densify=True)
    strategy: ADCStrategy = field(default_factory=ADCStrategy)

    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # --- CBS extensions ---
    # Enable weight loss (CBS Vi weight supervision)
    use_weight_loss: bool = True
    # LCWSR: per-Gaussian view-dependent weight (vi) + depth sigma
    lc_wsr_enable: bool = False
    # Depth correction: w(d) = max(0, 1-d/sigma) * vi
    if_depth_correct: bool = True


    # Learnable background weight
    learnable_background: bool = False


    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # whether profiling
    profiling: bool = False
    # Render backend
    render_backend: Literal["cbs", "3dgs"] = "3dgs"
    # Perf profiling steps: >0 = forward+backward, <0 = forward-only, 0 = eval only
    perf_steps: int = 0


    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
