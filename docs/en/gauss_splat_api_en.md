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

- `render_colors` (Tensor): Rendered color image, shape (C, 3, H, W)
- `render_depth` (Tensor): Rendered depth image, shape (C, 1, H, W)
- `info` (dict): Metadata dictionary, containing:
  - `gaussian_ids`: Gaussian point IDs (currently None)
  - `means2d`: 2D projection coordinates
  - `radii`: Projection radii
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
- `colors` (Tensor): Color values, shape (B, N, 3) or (B, C, N, 3)
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

- `means2d` (Tensor): 2D projection coordinates, shape (B, C, N, 2)
- `depths` (Tensor): Depth values, shape (B, C, N)
- `conics` (Tensor): 2D covariance inverse matrix (conic parameters), shape (B, C, N, 3)
- `opacities` (Tensor): Filtered opacities, shape (B, C, N)
- `radius` (Tensor): Projection radii, shape (B, C, N)
- `covars2d` (Tensor): 2D covariance matrices, shape (B, C, N, 2, 2)
- `colors` (Tensor): Filtered colors, shape (B, C, N, 3)
- `cnt` (Tensor): Number of valid Gaussian points

**Example**:

```python
means2d, depths, conics, opacities, radius, covars2d, colors, cnt = \
    gauss_splat.projection_three_dims_gaussian_fused(
        means=means,
        colors=colors,
        quat=quats,
        scales=scales,
        opacities=opacities,
        viewmats=viewmats,
        ks=Ks,
        width=1920,
        height=1080
    )
```

---

### 3. calc_render

**Function**: Calculates Gaussian Splatting rendering, performing alpha blending and depth computation.

**Location**: `gauss_splat.calc_render`

**Parameters**:

- `means` (Tensor): 2D Gaussian center coordinates, shape (2, N)
- `conic0s` (Tensor): Covariance inverse matrix element 0, shape (1, N)
- `conic1s` (Tensor): Covariance inverse matrix element 1, shape (1, N)
- `conic2s` (Tensor): Covariance inverse matrix element 2, shape (1, N)
- `opacities` (Tensor): Opacities, shape (1, N)
- `colors` (Tensor): Colors, shape (3, N)
- `depths` (Tensor, optional): Depth values, shape (1, N). If None, depth is not rendered
- `tile_coords` (Tensor): Tile coordinates
- `offsets` (Tensor): Offsets
- `sorted_gs_ids` (Tensor): Sorted Gaussian point IDs

**Return Values**:

- If `depths` is provided:
  - `color` (Tensor): Rendered color image
  - `depth` (Tensor): Rendered depth image
- If `depths` is not provided:
  - `color` (Tensor): Rendered color image

**Example**:

```python
render_colors, render_depths = gauss_splat.calc_render(
    means=means2d,
    conic0s=conics[:, 0],
    conic1s=conics[:, 1],
    conic2s=conics[:, 2],
    opacities=opacities,
    colors=colors,
    depths=depths,
    tile_coords=pix_coords,
    offsets=lb_sched,
    sorted_gs_ids=sorted_gs_id
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

- `output` (Tensor): Computed color values, shape (B, N, 3)

**Example**:

```python
colors = gauss_splat.spherical_harmonics(
    degrees_to_use=3,
    dirs=view_directions,  # (1, N, 3)
    coeffs=sh_coefficients  # (1, N, 16, 3)
)
```

---

### 5. gaussian_sort

**Function**: Sorts Gaussian points by depth for correct alpha blending.

**Location**: `gauss_splat.gaussian_sort`

**Parameters**:

- `lb_sched` (Tensor): Load balancing scheduling tensor
- `gaussian_cnt` (Tensor): Gaussian point count per tile
- `depths` (Tensor): Depth values
- `gs_ids` (Tensor): Gaussian point IDs
- `sorted_offset` (Tensor): Sorting offset
- `max_tile_gauss` (int): Maximum number of Gaussian points per tile

**Return Values**:

- `sorted_gs_ids` (Tensor): Sorted Gaussian point IDs

**Example**:

```python
sorted_gs_ids = gauss_splat.gaussian_sort(
    lb_sched=lb_sched_tensor,
    gaussian_cnt=tile_sums,
    depths=tile_depths,
    gs_ids=tile_gauss_ids,
    sorted_offset=sorted_offset,
    max_tile_gauss=max_tile_gauss
)
```

---

### 6. flash_gaussian_build_mask

**Function**: Builds tile masks for Gaussian Splatting for tile-based rendering optimization.

**Location**: `gauss_splat.flash_gaussian_build_mask`

**Parameters**:

- `means2d` (Tensor): 2D projection coordinates, shape (B, C, N, 2)
- `opacity` (Tensor): Opacities, shape (B, C, N)
- `conics` (Tensor): Covariance inverse matrix, shape (B, C, N, 3)
- `covars2d` (Tensor): 2D covariance matrix, shape (B, C, N, 2, 2)
- `depths` (Tensor): Depth values, shape (B, C, N)
- `cnt` (Tensor): Valid Gaussian point count, shape (B, C)
- `tile_grid` (Tensor): Tile grid coordinates
- `image_width` (int): Image width
- `image_height` (int): Image height
- `tile_size` (int): Tile size, default 64

**Return Values**:

- `tile_sum` (Tensor): Gaussian point sum per tile
- `tile_offset` (Tensor): Tile offsets
- `tile_depths` (Tensor): Tile depths
- `gauss_index` (Tensor): Gaussian point indices

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

- `means` (Tensor): 3D Gaussian center positions
- `colors` (Tensor): Color values
- `det` (Tensor): Covariance determinant
- `opacities` (Tensor): Opacities
- `means2d` (Tensor): 2D projection coordinates
- `depths` (Tensor): Depth values
- `radius` (Tensor): Projection radii
- `conics` (Tensor): Covariance inverse matrix
- `covars2d` (Tensor): 2D covariance matrix
- `compensations` (Tensor, optional): Compensation factors
- `width` (int): Image width
- `height` (int): Image height
- `near_plane` (float): Near plane distance
- `far_plane` (float): Far plane distance

**Return Values**:

- `means_culling`: Filtered 3D coordinates
- `colors_culling`: Filtered colors
- `means2d_culling`: Filtered 2D coordinates
- `depths_culling`: Filtered depths
- `radius_culling`: Filtered radii
- `covars2d_culling`: Filtered 2D covariances
- `conics_culling`: Filtered covariance inverse matrices
- `opacities_culling`: Filtered opacities
- `proj_filter`: Projection filter
- `cnt`: Number of valid points

---

### 8. get_render_schedule_cpp

**Function**: Obtains the rendering schedule for load balancing.

**Location**: `gauss_splat.get_render_schedule_cpp`

**Parameters**:

- `nums_tensor` (Tensor): Number of Gaussian points per tile
- `num_bins` (int): Number of bins (number of vector processors)

**Return Values**:

- `lb_sched_tensor` (Tensor): Load balancing scheduling tensor

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

All input tensors must be on the NPU device and support automatic differentiation.

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

print(f"Rendered color shape: {render_colors.shape}")  # (1, 3, 1080, 1920)
print(f"Rendered depth shape: {render_depths.shape}")  # (1, 1, 1080, 1920)
```

---

## Version Information

- Copyright © 2026 Huawei Technologies Co., Ltd.
- License: CANN Open Software License Agreement Version 2.0

---
