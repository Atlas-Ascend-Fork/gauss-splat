# coding=utf-8
# Adapted from
# https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/rendering.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from typing import Tuple

from gauss_splat import Rasterizer as _GSRasterizer

from rasterization.config import Config


class Rasterizer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._impl = _GSRasterizer()

    def ascend_rasterize_splats(
        self,
        cam: Tuple,
        size: Tuple,
        tile_size: int,
        active_sh_degree: int,
        splats: dict,
    ) -> Tuple:
        camtoworlds, Ks, render_mode = cam
        return self._impl.rasterization(
            cam=(camtoworlds, Ks, render_mode),
            size=size,
            tile_size=tile_size,
            active_sh_degree=active_sh_degree,
            splats=splats,
            camera_model=self.cfg.camera_model,
        )
