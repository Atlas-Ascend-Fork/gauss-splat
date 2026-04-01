from typing import List, Optional, Tuple

import torch

def _init_op_api_so_path(so_path: str) -> None: ...

def projection_three_dims_gaussian_fused(
    means: torch.Tensor,
    covars: torch.Tensor = None,
    viewmats: torch.Tensor = None,
    Ks: torch.Tensor = None,
    width: int = 0,
    height: int = 0,
    eps: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    calc_compensations: bool = False,
    camera_model = 'pinhole'
) -> torch.Tensor: ...

def gaussian_build_mask(
    means,
    radii,
    tile_grid,
    image_width,
    image_height,
    tile_size
) -> torch.Tensor:...

def gaussian_sort(
    all_in_mask,
    depths
) -> torch.Tensor:...

def calc_render(
    means,
    conic0s,
    conic1s,
    conic2s,
    opacities,
    colors,
    depths,
    tile_coords,
    offsets,
    sorted_gs_ids
) -> torch.Tensor: ...

def spherical_harmonics(
    degrees_to_use,
    dirs,
    coeffs
) -> torch.Tensor:...

def get_render_schedule_cpp(nums, num_bins) -> torch.Tensor:...

def flash_gaussian_build_mask(
            means2d,
            opacity,
            conics,
            covars2d,
            cnt,
            tile_grid,
            image_width,
            image_height,
            tile_size
) -> torch.Tensor:...

def gaussian_filter(
    means,
    colors,
    det,
    opacities,
    means2d,
    depths,
    radius,
    conics,
    covars2d,
    compensations,
    width,
    height,
    near_plane,
    far_plane,
) -> torch.Tensor:...

__all__ = [
    "projection_three_dims_gaussian_fused",
    "calc_render",
    "gaussian_sort",
    "get_render_schedule_cpp",
    "spherical_harmonics",
    "flash_gaussian_build_mask",
    "gaussian_filter"
]
