# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import json
import os
import time
import datetime
from collections import defaultdict
from types import SimpleNamespace
import imageio
import numpy as np
import torch
import torch_npu
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import tqdm
import yaml

from datasets.colmap import Dataset, Parser
from rasterization.utils import SSIM, set_random_seed, create_splats_with_optimizers
from gsplat.utils import save_ply
from rasterization.config import Config
from rasterization.rasterizer_3dgs import Rasterizer as Rasterizer3DGS
from gauss_splat import fused_ssim


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"npu:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)
        
        if cfg.render_backend == "3dgs":
            self.raserizer = Rasterizer3DGS(cfg)
        else:
            from rasterization.rasterizer import Rasterizer as CBSRasterizer

            self.raserizer = CBSRasterizer(cfg)

        # Model
        feature_dim = None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            learnable_background=cfg.learnable_background,
            lc_wsr_enable=cfg.lc_wsr_enable,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")
            
        # Densification strategy state (only used when cfg.densify=True).
        self.strategy_state = None
        if cfg.densify:
            if cfg.lc_wsr_enable:
                cfg.strategy.initialize_state(0.01, torch.log, self.splats["means"].shape[0], self.scene_scale, self.device)
            else:
                cfg.strategy.check_sanity(self.splats, self.optimizers)
                self.strategy_state = cfg.strategy.initialize_state(
                    scene_scale=self.scene_scale,
                    lc_wsr_enable=cfg.lc_wsr_enable
                )
 
        if cfg.profiling:
            name = cfg.data_dir.split("/")[-2]
            self.max_profiling_step = 32
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
            self.profiler = torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.NPU,    # 采集NPU数据开关
                                    torch_npu.profiler.ProfilerActivity.CPU],    # 采集框架侧数据开关
                        with_stack=True,    # 算子调用栈，Bool类型
                        record_shapes=False,    # 算子的InputShapes和InputTypes，Bool类
                        profile_memory=False,    # 算子的内存占用情况，Bool类型
                        # profile_memory=True,    # 算子的内存占用情况，Bool类型
                        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=2, 
                                                                skip_first=28),    # 设置不同step的行
                        experimental_config=experimental_config,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                            f"./profiling/{datetime.date.today().strftime('%m-%d')}/{name}_FFF"
                            ))    # 将采集到的性能数据导出为TensorBoard工具支持的格式


    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        camera_depth = {}
        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.lc_wsr_enable and "sigma" in self.optimizers:
            sigma_lr_init = 0.1 * self.scene_scale
            sigma_lr_final = 0.001 * self.scene_scale
            sigma_gamma = (sigma_lr_final / sigma_lr_init) ** (1.0 / max_steps)
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizers["sigma"], gamma=sigma_gamma
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]
            render_mode = "RGB+ED" if cfg.depth_loss else "RGB"
            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # metagauss render，不支持background color
            if self.cfg.profiling and step == 0:
                self.profiler.start()
            renders, render_depth, info = self.raserizer.ascend_rasterize_splats(
                cam=(camtoworlds, Ks, render_mode),
                size=(width, height),
                tile_size=cfg.tile_size, 
                active_sh_degree=sh_degree_to_use,
                splats=self.splats,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, render_depth

            if cfg.densify:
                cfg.strategy.step_pre_backward(
                    self.splats, self.optimizers, self.strategy_state, step, info
                )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            # ssimloss = 1.0 - SSIM.ssim(
            #     colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
            # )
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), window_size=11
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths_sample = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths_sample = depths_sample.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths_sample > 0.0, depths_sample, torch.zeros_like(depths_sample))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            
            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )
            if cfg.use_weight_loss and cfg.render_backend == "cbs":
                if step < 8500:
                    camera_depth[image_ids.item()] = depths.detach()
                    depth_used = depths
                else:
                    depth_used = camera_depth.get(image_ids.item(), depths)
                weight = info["weight"]
                weight_inv = weight
                norm_weight = weight_inv / weight_inv.mean().detach()
                depth_inv = depth_used
                depth2 = (depth_inv * depth_inv * depth_inv)
                norm_depth = (depth2 / depth2.mean()).detach()
                supervise_mask = norm_depth < 0.5
                weight_loss = F.l1_loss(norm_depth[supervise_mask] * 0, norm_weight[supervise_mask]) if step > 6500 else torch.tensor(0.0, device=device)
                weight_ratio = 0.1 * loss.detach()
                k_scale_weight = (weight_ratio / weight_loss.detach()).detach()
                loss = loss + (weight_loss * k_scale_weight if weight_loss > weight_ratio else weight_loss)


            loss.backward()

            if cfg.densify:
                means2d_grad = info["means2d"].grad.permute(0, 1, 3, 2)[0, 0]  # [N,2]
                strategy_info = dict(info)
                if cfg.lc_wsr_enable:
                    strategy_info["means2d"] = means2d_grad
                else:
                    strategy_info["means2d"] = SimpleNamespace(grad=means2d_grad)
                strategy_info["radii"] = info["radii"].permute(0, 1, 3, 2)[0, 0]
                strategy_info["proj_filter"] = info["proj_filter"][0, 0]
                cfg.strategy.step_post_backward(
                    self.splats,
                    self.optimizers,
                    self.strategy_state,
                    step,
                    strategy_info,
                )

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            if world_rank == 0 and step % 800 == 0:
                canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                canvas = canvas.reshape(-1, *canvas.shape[2:])
                imageio.imwrite(
                    f"{self.render_dir}/train_rank{self.world_rank}.png",
                    (canvas * 255).astype(np.uint8),
                )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.npu.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.npu.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                rgb = None
                save_ply(self.splats, f"{self.ply_dir}/point_cloud_{step}.ply", rgb)

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
            
            if self.cfg.profiling:
                if step < self.max_profiling_step:
                    self.profiler.step()
                elif step == self.max_profiling_step:
                    self.profiler.stop()
                    exit()

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            render_mode = "RGB"
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            
            torch_npu.npu.synchronize()
            tic = time.time()
            colors, depths, info = self.raserizer.ascend_rasterize_splats(
                cam=(camtoworlds, Ks, render_mode),
                size=(width, height),
                tile_size=cfg.tile_size, 
                active_sh_degree=cfg.sh_degree, 
                splats=self.splats,
            )
            torch_npu.npu.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            ###  nan
            colors = torch.nan_to_num(colors, nan=0.0, posinf=1e10, neginf=-1e10)
            
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            total_render_time = ellipse_time
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
            # ===== [Profiling 修改] 开始：eval 返回 total_render_time =====
            return total_render_time, len(valloader)
            # ===== [Profiling 修改] 结束 =====
        return 0.0, 0
    def perf_forward(self, num_steps: int = 10):
        """Forward-only perf mode: run N iterations without backward."""
        cfg = self.cfg
        device = self.device
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        for step in range(num_steps):
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            render_mode = "RGB+ED" if cfg.depth_loss else "RGB"
            with torch.no_grad():
                renders, render_depth, info = self.raserizer.ascend_rasterize_splats(
                    cam=(camtoworlds, Ks, render_mode),
                    size=(width, height),
                    tile_size=cfg.tile_size,
                    active_sh_degree=cfg.sh_degree,
                    splats=self.splats,
                )
            torch.npu.synchronize()
            print(f"  [perf_fwd] step {step} done", flush=True)

    def perf_backward(self, num_steps: int = 10):
        """Forward+backward perf mode: run N iterations with loss.backward()."""
        cfg = self.cfg
        device = self.device
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)
        for step in range(num_steps):
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            render_mode = "RGB+ED" if cfg.depth_loss else "RGB"
            renders, render_depth, info = self.raserizer.ascend_rasterize_splats(
                cam=(camtoworlds, Ks, render_mode),
                size=(width, height),
                tile_size=cfg.tile_size,
                active_sh_degree=cfg.sh_degree,
                splats=self.splats,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, render_depth
            l1loss = F.l1_loss(colors, pixels)
            if cfg.ssim_lambda > 0.0:
                ssimloss = 1.0 - SSIM.ssim(
                    colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            else:
                loss = l1loss
            loss.backward()
            torch.npu.synchronize()
            # Zero grads for next iteration
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            print(f"  [perf_bwd] step {step} done, loss={loss.item():.4f}", flush=True)
