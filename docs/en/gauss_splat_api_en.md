# Gauss Splat API Documentation

## Overview

`gauss_splat` is a Gaussian Splatting rendering library based on Huawei Ascend NPU. It provides a complete 3D Gaussian rendering pipeline, including core operations such as projection, sorting, and rendering, and supports differentiable rendering.

## External Interface List

### 1. Rasterizer Class

**Function**: Gaussian Splatting rasterizer, providing the complete rendering pipeline.

**Location**: `gauss_splat.Rasterizer`

#### Main Methods

##### `rasterization()`

Renders 3D Gaussian point clouds to 2D images.

**Parameters**:

- `cam` (Tuple): Camera parameter tuple
  - `camtoworlds` (Tensor): Camera-to-world transformation matrix, shape (C, 4, 4)
  - `Ks` (Tensor): Camera intrinsic matrix, shape (C, 3, 3)
  - `render_mode` (str): Render mode, optional values: "RGB", "D", "ED", "RGB+D", "RGB+ED"
- `size` (Tuple): Image size (width, height)
- `tile_size` (int): Tile size, default is 32
- `active_sh_degree` (int): Active spherical harmonics degree
- `splats` (dict): Gaussian point cloud dictionary, containing:
  - `means` (Tensor): Gaussian center positions, shape (N, 3)
  - `quats` (Tensor): Quaternions for rotation, shape (N, 4)
  - `scales` (Tensor): Scale parameters (log space), shape (N, 3)
  - `opacities` (Tensor): Opacities (logits), shape (N,)
  - `sh0` (Tensor): Spherical harmonics coefficients degree 0, shape (N, 1, 3)
  - `shN` (Tensor): Spherical harmonics coefficients higher degrees, shape (N, K, 3)
- `camera_model` (str): Camera model, optional values: "pinhole", "ortho", "fisheye", default is "pinhole"

**Return Values**:

- `render_colors` (Tensor): Rendered color image, shape (C, H, W, 3)
- `render_depth` (Tensor): Rendered depth image, shape (C, H, W, 1)
- `info` (dict): Metadata dictionary, containing:
  - `gaussian_ids`: Gaussian point IDs (currently None)
  - `means2d`: 2D projection coordinates, shape (B, C, 2, N)
  - `radii`: Projection radii, shape (B, C, 2, N)
  - `width`: Image width
  - `height`: Image height
  - `n_cameras`: Number of cameras

**Example**:

```python
rasterizer = gauss_splat.Rasterizer()
render_colors, render_depth, info = rasterizer.rasterization(
    cam=(camtoworlds, Ks, "RGB"),
    size=(width, height),
    tile_size=32,
    active_sh_degree=3,
    splats=splats_dict,
    camera_model="pinhole"
)
```

---

### 2. projection_three_dims_gaussian_fused

**Function**: 3D Gaussian projection fused operation, projects 3D Gaussians onto the 2D plane and performs clipping.

**Location**: `gauss_splat.projection_three_dims_gaussian_fused`

**Parameters**:

- `means` (Tensor): Gaussian center positions, shape (B, N, 3)
- `colors` (Tensor): Color values, shape (B, 3, N)
- `covars` (Tensor, optional): Covariance matrix, shape (B, N, 3, 3). Mutually exclusive with quat/scales
- `quat` (Tensor, optional): Quaternions for rotation, shape (B, N, 4). Mutually exclusive with covars
- `scales` (Tensor, optional): Scale parameters, shape (B, N, 3). Used together with quat
- `opacities` (Tensor): Opacities, shape (B, N)
- `viewmats` (Tensor): View transformation matrix, shape (B, C, 4, 4)
- `ks` (Tensor): Camera intrinsic matrix, shape (B, C, 3, 3)
- `width` (int): Image width
- `height` (int): Image height
- `eps` (float): Minimum Gaussian radius threshold, default 0.3
- `near_plane` (float): Near plane distance, default 0.01
- `far_plane` (float): Far plane distance, default 1e10
- `calc_compensations` (bool): Whether to calculate compensation factors, default False
- `camera_model` (str): Camera model, default "pinhole"

**Return Values**:

- `means2d` (Tensor): 2D projection coordinates, shape (B, C, 2, N)
- `depths` (Tensor): Depth values, shape (B, C, N)
- `conics` (Tensor): 2D covariance inverse matrix (conic parameters), shape (B, C, 3, N)
- `opacities` (Tensor): Filtered opacities, shape (B, C, N)
- `radius` (Tensor): Projection radii, shape (B, C, 2, N)
- `covars2d` (Tensor): 2D covariance matrices, shape (B, C, 3, N)
- `colors` (Tensor): Filtered colors, shape (B, C, 3, N)
- `cnt` (Tensor): Number of valid Gaussian points, shape (B, C)
- `proj_filter` (Tensor): Projection filter, shape (B, C, ceil(N/8))

**Example**:

```python
means2d, depths, conics, opacities, radius, covars2d, colors, cnt, proj_filter = \
    gauss_splat.projection_three_dims_gaussian_fused(
        means,  # means
        colors,  # colors
        None,  # covars
        quats,  # quat
        scales,  # scales
        opacities,  # opacities
        viewmats,  # viewmats
        Ks,  # ks
        1920,  # width
        1080,  # height
    )
```

---

### 3. calc_render

**Function**: Calculates Gaussian Splatting rendering, performing alpha blending and depth computation.

**Location**: `gauss_splat.calc_render`

**Parameters**:

- `means` (Tensor): 2D Gaussian center coordinates, shape (2, N)
- `conic0s` (Tensor): Covariance inverse matrix element 0, shape (N,)
- `conic1s` (Tensor): Covariance inverse matrix element 1, shape (N,)
- `conic2s` (Tensor): Covariance inverse matrix element 2, shape (N,)
- `opacities` (Tensor): Opacities, shape (N,)
- `colors` (Tensor): Colors, shape (3, N)
- `depths` (Tensor, optional): Depth values, shape (1, N). If None, depth is not rendered
- `tile_coords` (Tensor): Tile coordinates, shape (tileNum, 2, nPixel)
- `offsets` (Tensor): Offsets, shape(vectorCnt + (TileNum * 2))
- `sorted_gs_ids` (Tensor): Sorted Gaussian point IDs, shape (totalGauss)

These four parameters are one-dimensional vectors for a single view. CalcRender.forward adds the row dimension with unsqueeze(0) before concatenating with means/colors; do not add it in the caller.

**Return Values**:

- If `depths` is provided:
  - `color` (Tensor): Rendered color image, shape (3, tileNum, nPixel)
  - `depth` (Tensor): Rendered depth image, shape (1, tileNum, nPixel)
- If `depths` is not provided:
  - `color` (Tensor): Rendered color image, shape (3, tileNum, nPixel)

**Example**:

```python
render_colors, render_depths = gauss_splat.calc_render(
                                   means2d,  # means
                                   conics[:, 0],  # conic0s
                                   conics[:, 1],  # conic1s
                                   conics[:, 2],  # conic2s
                                   opacities,  # opacities
                                   colors,  # colors
                                   depths,  # depths
                                   pix_coords,  # tile_coords
                                   lb_sched,  # offsets
                                   sorted_gs_id,  # sorted_gs_ids
                               )
```

---

### 4. spherical_harmonics

**Function**: Computes spherical harmonics for view-dependent color calculation.

**Location**: `gauss_splat.spherical_harmonics`

**Parameters**:

- `degrees_to_use` (int): Spherical harmonics degree, range 0-4
- `dirs` (Tensor): Direction vectors, shape (B, N, 3)
- `coeffs` (Tensor): Spherical harmonics coefficients, shape (B, N, K, 3), where K=(degrees_to_use+1)^2

**Return Values**:

- `output` (Tensor): Computed color values, shape (B, 3, N)

**Example**:

```python
colors = gauss_splat.spherical_harmonics(
             3,  # degrees_to_use
             view_directions,  # dirs
             sh_coefficients,  # coeffs
         )
```

---

### 5. gaussian_sort

**Function**: Sorts Gaussian points by depth for correct alpha blending.

**Location**: `gauss_splat.gaussian_sort`

**Parameters**:

- `lb_sched` (Tensor): Load balancing scheduling tensor, shape(B, C, schedule_num)
- `gaussian_cnt` (Tensor): Gaussian point count per tile, shape(B, C, tile_num, 1)
- `depths` (Tensor): Depth values, shape(B, C, tile_num, N)
- `gs_ids` (Tensor): Gaussian point IDs, shape(B, C, tile_num, N)
- `sorted_offset` (Tensor): Sorting offset, shape(B*C)
- `max_tile_gauss` (int): Maximum number of Gaussian points per tile

**Return Values**:

- `sorted_gs_ids` (Tensor): Sorted Gaussian point IDs, shape(totalGauss)

**Example**:

```python
sorted_gs_ids = gauss_splat.gaussian_sort(
                    lb_sched_tensor,  # lb_sched
                    tile_sums,  # gaussian_cnt
                    tile_depths,  # depths
                    tile_gauss_ids,  # gs_ids
                    sorted_offset,  # sorted_offset
                    max_tile_gauss,  # max_tile_gauss
                )
```

---

### 6. flash_gaussian_build_mask

**Function**: Builds tile masks for Gaussian Splatting for tile-based rendering optimization.

**Location**: `gauss_splat.flash_gaussian_build_mask`

**Parameters**:

- `means2d` (Tensor): 2D projection coordinates, shape (B, C, 2, N)
- `opacity` (Tensor): Opacities, shape (B, C, 1, N)
- `conics` (Tensor): Covariance inverse matrix, shape(B, C, 3, N)
- `covars2d` (Tensor): 2D covariance matrix, shape(B, C, 3, N)
- `depths` (Tensor): Depth values, shape (B, C, 1, N)
- `cnt` (Tensor): Valid Gaussian point count, shape (B, C)
- `tile_grid` (Tensor): Tile grid coordinates
- `image_width` (int): Image width
- `image_height` (int): Image height
- `tile_size` (int): Tile size, default 64

**Return Values**:

- `tile_sum` (Tensor): Gaussian point sum per tile, shape (B, C, tile_num, 1)
- `tile_offset` (Tensor): Tile offsets, shape (B, C, tile_num, 1)
- `tile_depths` (Tensor): Tile depths, shape (B, C, tile_num, N)
- `gauss_index` (Tensor): Gaussian point indices, shape (B, C, tile_num, N)

**Example**:

```python
tile_sums, tile_offsets, tile_depths, tile_gauss_ids = \
    gauss_splat.flash_gaussian_build_mask(
        means2d=means2d,
        opacity=opacities,
        conics=conics,
        covars2d=covars2d,
        depths=depths,
        cnt=cnt,
        tile_grid=tile_grid,
        image_width=1920,
        image_height=1080,
        tile_size=32
    )
```

---

### 7. gaussian_filter

**Function**: Filters Gaussian points based on depth range and validity.

**Location**: `gauss_splat.gaussian_filter`

**Parameters**:

- `means` (Tensor): 3D Gaussian center positions, shape(B, 3, N)
- `colors` (Tensor): Color values, shape(B, 3, N)
- `det` (Tensor): Covariance determinant, shape(B, C, N)
- `opacities` (Tensor): Opacities, shape(B, N)
- `means2d` (Tensor): 2D projection coordinates, shape(B, C, 2, N)
- `depths` (Tensor): Depth values, shape(B, C, N)
- `radius` (Tensor): Projection radii, shape(B, C, 2, N)
- `conics` (Tensor): Covariance inverse matrix, shape(B, C, 3, N)
- `covars2d` (Tensor): 2D covariance matrix, shape(B, C, 3, N)
- `compensations` (Tensor, optional): Compensation factors, shape(B, C, N)
- `width` (int): Image width
- `height` (int): Image height
- `near_plane` (float): Near plane distance
- `far_plane` (float): Far plane distance

**Return Values**:

- `means_culling`: Filtered 3D coordinates, shape(B, C, 3, N)
- `colors_culling`: Filtered colors, shape(B, C, 3, N)
- `means2d_culling`: Filtered 2D coordinates, shape(B, C, 2, N)
- `depths_culling`: Filtered depths, shape(B, C, N)
- `radius_culling`: Filtered radii, shape(B, C, 2, N)
- `covars2d_culling`: Filtered 2D covariances, shape(B, C, 3, N)
- `conics_culling`: Filtered covariance inverse matrices, shape(B, C, 3, N)
- `opacities_culling`: Filtered opacities, shape(B, C, N)
- `proj_filter`: Projection filter, shape(B, C, ceil(N/8))
- `cnt`: Number of valid points, shape(B, C)

---

### 8. get_render_schedule_cpp

**Function**: Obtains the rendering schedule for load balancing.

**Location**: `gauss_splat.get_render_schedule_cpp`

**Parameters**:

- `nums_tensor` (Tensor): Number of Gaussian points per tile, shape (B, C, T). For batched inputs with `B*C > 1`, it must be an `int64` CPU tensor. For 1D inputs or `B*C == 1`, the implementation accepts the input device and returns the schedule on that device.
- `num_bins` (int): Number of bins (number of vector processors)

**Return Values**:

- `lb_sched_tensor` (Tensor): Load balancing scheduling tensor, shape(B, C, M)

**Example**:

```python
vector_num = acl.get_device_capability(0, 1)[0]
lb_sched_tensor = gauss_splat.get_render_schedule_cpp(
    nums_tensor=tile_sums_cpu,
    num_bins=vector_num
)
```

---

## Data Type Description

### Tensor Device Requirements

Except for the documented `get_render_schedule_cpp` batch case, input tensors must be on the NPU device and support automatic differentiation. For a batched `get_render_schedule_cpp` input with `B*C > 1`, `nums_tensor` must be an `int64` CPU tensor.

### Data Format

- **Position coordinates**: 3D coordinates in the world coordinate system (x, y, z)
- **Quaternions**: Quaternions used to represent rotation (w, x, y, z)
- **Scales**: Scale factors in log space, converted through `torch.exp(scales)` during rendering
- **Opacities**: In logit form, converted through `torch.sigmoid(opacities)` during rendering
- **Spherical harmonics coefficients**: Used for view-dependent colors, shape (N, K, 3), where K is the number of coefficients

### Camera Model

Three camera models are supported:

- **pinhole**: Pinhole camera model
- **ortho**: Orthographic projection
- **fisheye**: Fisheye lens

### Render Mode

- **RGB**: Render colors only
- **D**: Render depth only
- **ED**: Render edge depth only
- **RGB+D**: Render colors and depth
- **RGB+ED**: Render colors and edge depth

---

## Complete Rendering Pipeline Example

```python
import torch
import gauss_splat

# 1. Initialize the rasterizer
rasterizer = gauss_splat.Rasterizer()

# 2. Prepare Gaussian point cloud data
N = 10000  # Number of Gaussian points
splats = {
    "means": torch.randn(N, 3, device='npu:0'),           # Positions
    "quats": torch.randn(N, 4, device='npu:0'),           # Quaternions
    "scales": torch.randn(N, 3, device='npu:0'),          # Scales (log space)
    "opacities": torch.randn(N, device='npu:0'),          # Opacities (logits)
    "sh0": torch.randn(N, 1, 3, device='npu:0'),          # Spherical harmonics degree 0
    "shN": torch.randn(N, 15, 3, device='npu:0'),         # Spherical harmonics higher degrees
}

# 3. Prepare camera parameters
C = 1  # Number of cameras
camtoworlds = torch.eye(4, device='npu:0').unsqueeze(0).expand(C, -1, -1)
Ks = torch.tensor([
    [500.0, 0.0, 960.0],
    [0.0, 500.0, 540.0],
    [0.0, 0.0, 1.0]
], device='npu:0').unsqueeze(0).expand(C, -1, -1)

# 4. Render
render_colors, render_depths, info = rasterizer.rasterization(
    cam=(camtoworlds, Ks, "RGB"),
    size=(1920, 1080),
    tile_size=32,
    active_sh_degree=3,
    splats=splats,
    camera_model="pinhole"
)

print(f"Rendered color shape: {render_colors.shape}")  # (1, 1080, 1920, 3)
print(f"Rendered depth shape: {render_depths.shape}")  # (1, 1080, 1920, 1)
```

---

## Version Information

- Copyright © 2026 Huawei Technologies Co., Ltd.
- License: CANN Open Software License Agreement Version 2.0

---
