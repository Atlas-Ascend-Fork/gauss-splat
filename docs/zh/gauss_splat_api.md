# Gauss Splat 接口说明文档

## 概述

`gauss_splat` 是一个基于华为昇腾NPU的高斯溅射（Gaussian Splatting）渲染库，提供了完整的3D高斯渲染流水线。该库包含投影、排序、渲染等核心操作，支持可微分渲染。

## 对外接口列表

### 1. Rasterizer 类

**功能**: 高斯溅射光栅化器，提供完整的渲染流水线。

**位置**: `gauss_splat.Rasterizer`

#### 主要方法

##### `rasterization()`

渲染3D高斯点云到2D图像。

**参数**:

- `cam` (Tuple): 相机参数元组
  - `camtoworlds` (Tensor): 相机到世界变换矩阵，形状 (C, 4, 4)
  - `Ks` (Tensor): 相机内参矩阵，形状 (C, 3, 3)
  - `render_mode` (str): 渲染模式，可选值: "RGB", "D", "ED", "RGB+D", "RGB+ED"
- `size` (Tuple): 图像尺寸 (width, height)
- `tile_size` (int): 分块大小，默认为32
- `active_sh_degree` (int): 激活的球谐函数阶数
- `splats` (dict): 高斯点云字典，包含:
  - `means` (Tensor): 高斯中心位置，形状 (N, 3)
  - `quats` (Tensor): 四元数表示旋转，形状 (N, 4)
  - `scales` (Tensor): 缩放参数（log空间），形状 (N, 3)
  - `opacities` (Tensor): 不透明度（logits），形状 (N,)
  - `sh0` (Tensor): 球谐系数第0阶，形状 (N, 1, 3)
  - `shN` (Tensor): 球谐系数高阶，形状 (N, K, 3)
- `camera_model` (str): 相机模型，可选值: "pinhole", "ortho", "fisheye"，默认为"pinhole"

**返回值**:

- `render_colors` (Tensor): 渲染的颜色图像，形状 (C, 3, H, W)
- `render_depth` (Tensor): 渲染的深度图像，形状 (C, 1, H, W)
- `info` (dict): 元数据字典，包含:
  - `gaussian_ids`: 高斯点ID（当前为None）
  - `means2d`: 2D投影坐标
  - `radii`: 投影半径
  - `width`: 图像宽度
  - `height`: 图像高度
  - `n_cameras`: 相机数量

**示例**:

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

**功能**: 3D高斯投影融合操作，将3D高斯投影到2D平面并进行裁剪。

**位置**: `gauss_splat.projection_three_dims_gaussian_fused`

**参数**:

- `means` (Tensor): 高斯中心位置，形状 (B, N, 3)
- `colors` (Tensor): 颜色值，形状 (B, N, 3) 或 (B, C, N, 3)
- `covars` (Tensor, optional): 协方差矩阵，形状 (B, N, 3, 3)。与quat/scales互斥
- `quat` (Tensor, optional): 四元数表示旋转，形状 (B, N, 4)。与covars互斥
- `scales` (Tensor, optional): 缩放参数，形状 (B, N, 3)。与quat一起使用
- `opacities` (Tensor): 不透明度，形状 (B, N)
- `viewmats` (Tensor): 视图变换矩阵，形状 (B, C, 4, 4)
- `ks` (Tensor): 相机内参矩阵，形状 (B, C, 3, 3)
- `width` (int): 图像宽度
- `height` (int): 图像高度
- `eps` (float): 最小高斯半径阈值，默认0.3
- `near_plane` (float): 近平面距离，默认0.01
- `far_plane` (float): 远平面距离，默认1e10
- `calc_compensations` (bool): 是否计算补偿因子，默认False
- `camera_model` (str): 相机模型，默认"pinhole"

**返回值**:

- `means2d` (Tensor): 2D投影坐标，形状 (B, C, N, 2)
- `depths` (Tensor): 深度值，形状 (B, C, N)
- `conics` (Tensor): 2D协方差逆矩阵（锥形参数），形状 (B, C, N, 3)
- `opacities` (Tensor): 过滤后的不透明度，形状 (B, C, N)
- `radius` (Tensor): 投影半径，形状 (B, C, N)
- `covars2d` (Tensor): 2D协方差矩阵，形状 (B, C, N, 2, 2)
- `colors` (Tensor): 过滤后的颜色，形状 (B, C, N, 3)
- `cnt` (Tensor): 有效高斯点数量

**示例**:

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

**功能**: 计算高斯溅射渲染，执行alpha混合和深度计算。

**位置**: `gauss_splat.calc_render`

**参数**:

- `means` (Tensor): 2D高斯中心坐标，形状 (2, N)
- `conic0s` (Tensor): 协方差逆矩阵元素0，形状 (1, N)
- `conic1s` (Tensor): 协方差逆矩阵元素1，形状 (1, N)
- `conic2s` (Tensor): 协方差逆矩阵元素2，形状 (1, N)
- `opacities` (Tensor): 不透明度，形状 (1, N)
- `colors` (Tensor): 颜色，形状 (3, N)
- `depths` (Tensor, optional): 深度值，形状 (1, N)。如为None则不渲染深度
- `tile_coords` (Tensor): 分块坐标
- `offsets` (Tensor): 偏移量
- `sorted_gs_ids` (Tensor): 排序后的高斯点ID

**返回值**:

- 如果提供`depths`:
  - `color` (Tensor): 渲染的颜色图像
  - `depth` (Tensor): 渲染的深度图像
- 如果不提供`depths`:
  - `color` (Tensor): 渲染的颜色图像

**示例**:

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

**功能**: 计算球谐函数，用于视角相关的颜色计算。

**位置**: `gauss_splat.spherical_harmonics`

**参数**:

- `degrees_to_use` (int): 球谐函数阶数，范围0-4
- `dirs` (Tensor): 方向向量，形状 (B, N, 3)
- `coeffs` (Tensor): 球谐系数，形状 (B, N, K, 3)，其中K=(degrees_to_use+1)²

**返回值**:

- `output` (Tensor): 计算得到的颜色值，形状 (B, N, 3)

**示例**:

```python
colors = gauss_splat.spherical_harmonics(
    degrees_to_use=3,
    dirs=view_directions,  # (1, N, 3)
    coeffs=sh_coefficients  # (1, N, 16, 3)
)
```

---

### 5. gaussian_sort

**功能**: 对高斯点按深度进行排序，用于正确的alpha混合。

**位置**: `gauss_splat.gaussian_sort`

**参数**:

- `lb_sched` (Tensor): 负载均衡调度张量
- `gaussian_cnt` (Tensor): 每个tile的高斯点计数
- `depths` (Tensor): 深度值
- `gs_ids` (Tensor): 高斯点ID
- `sorted_offset` (Tensor): 排序偏移量
- `max_tile_gauss` (int): 单个tile最大高斯点数

**返回值**:

- `sorted_gs_ids` (Tensor): 排序后的高斯点ID

**示例**:

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

**功能**: 构建高斯溅射的tile掩码，用于分块渲染优化。

**位置**: `gauss_splat.flash_gaussian_build_mask`

**参数**:

- `means2d` (Tensor): 2D投影坐标，形状 (B, C, N, 2)
- `opacity` (Tensor): 不透明度，形状 (B, C, N)
- `conics` (Tensor): 协方差逆矩阵，形状 (B, C, N, 3)
- `covars2d` (Tensor): 2D协方差矩阵，形状 (B, C, N, 2, 2)
- `depths` (Tensor): 深度值，形状 (B, C, N)
- `cnt` (Tensor): 有效高斯点计数，形状 (B, C)
- `tile_grid` (Tensor): tile网格坐标
- `image_width` (int): 图像宽度
- `image_height` (int): 图像高度
- `tile_size` (int): tile大小，默认64

**返回值**:

- `tile_sum` (Tensor): 每个tile的高斯点和
- `tile_offset` (Tensor): tile偏移量
- `tile_depths` (Tensor): tile深度
- `gauss_index` (Tensor): 高斯点索引

**示例**:

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

**功能**: 根据深度范围和有效性过滤高斯点。

**位置**: `gauss_splat.gaussian_filter`

**参数**:

- `means` (Tensor): 3D高斯中心位置
- `colors` (Tensor): 颜色值
- `det` (Tensor): 协方差行列式
- `opacities` (Tensor): 不透明度
- `means2d` (Tensor): 2D投影坐标
- `depths` (Tensor): 深度值
- `radius` (Tensor): 投影半径
- `conics` (Tensor): 协方差逆矩阵
- `covars2d` (Tensor): 2D协方差矩阵
- `compensations` (Tensor, optional): 补偿因子
- `width` (int): 图像宽度
- `height` (int): 图像高度
- `near_plane` (float): 近平面距离
- `far_plane` (float): 远平面距离

**返回值**:

- `means_culling`: 过滤后的3D坐标
- `colors_culling`: 过滤后的颜色
- `means2d_culling`: 过滤后的2D坐标
- `depths_culling`: 过滤后的深度
- `radius_culling`: 过滤后的半径
- `covars2d_culling`: 过滤后的2D协方差
- `conics_culling`: 过滤后的协方差逆矩阵
- `opacities_culling`: 过滤后的不透明度
- `proj_filter`: 投影过滤器
- `cnt`: 有效点数量

---

### 8. get_render_schedule_cpp

**功能**: 获取渲染调度方案，用于负载均衡。

**位置**: `gauss_splat.get_render_schedule_cpp`

**参数**:

- `nums_tensor` (Tensor): 每个tile的高斯点数量
- `num_bins` (int): 分箱数量（向量处理器数量）

**返回值**:

- `lb_sched_tensor` (Tensor): 负载均衡调度张量

**示例**:

```python
vector_num = acl.get_device_capability(0, 1)[0]
lb_sched_tensor = gauss_splat.get_render_schedule_cpp(
    nums_tensor=tile_sums_cpu,
    num_bins=vector_num
)
```

---

## 数据类型说明

### 张量设备要求

所有输入张量必须在NPU设备上，支持自动求导。

### 数据格式

- **位置坐标**: 世界坐标系下的3D坐标 (x, y, z)
- **四元数**: 用于表示旋转的四元数 (w, x, y, z)
- **缩放**: log空间的缩放因子，渲染时通过`torch.exp(scales)`转换
- **不透明度**: logit形式，渲染时通过`torch.sigmoid(opacities)`转换
- **球谐系数**: 用于视角相关颜色，形状为 (N, K, 3)，K为系数数量

### 相机模型

支持三种相机模型：

- **pinhole**: 针孔相机模型
- **ortho**: 正交投影
- **fisheye**: 鱼眼镜头

### 渲染模式

- **RGB**: 仅渲染颜色
- **D**: 仅渲染深度
- **ED**: 仅渲染边缘深度
- **RGB+D**: 渲染颜色和深度
- **RGB+ED**: 渲染颜色和边缘深度

---

## 完整渲染流水线示例

```python
import torch
import gauss_splat

# 1. 初始化光栅化器
rasterizer = gauss_splat.Rasterizer()

# 2. 准备高斯点云数据
N = 10000  # 高斯点数量
splats = {
    "means": torch.randn(N, 3, device='npu:0'),           # 位置
    "quats": torch.randn(N, 4, device='npu:0'),           # 四元数
    "scales": torch.randn(N, 3, device='npu:0'),          # 缩放（log空间）
    "opacities": torch.randn(N, device='npu:0'),          # 不透明度（logits）
    "sh0": torch.randn(N, 1, 3, device='npu:0'),          # 球谐第0阶
    "shN": torch.randn(N, 15, 3, device='npu:0'),         # 球谐高阶
}

# 3. 准备相机参数
C = 1  # 相机数量
camtoworlds = torch.eye(4, device='npu:0').unsqueeze(0).expand(C, -1, -1)
Ks = torch.tensor([
    [500.0, 0.0, 960.0],
    [0.0, 500.0, 540.0],
    [0.0, 0.0, 1.0]
], device='npu:0').unsqueeze(0).expand(C, -1, -1)

# 4. 渲染
render_colors, render_depths, info = rasterizer.rasterization(
    cam=(camtoworlds, Ks, "RGB"),
    size=(1920, 1080),
    tile_size=32,
    active_sh_degree=3,
    splats=splats,
    camera_model="pinhole"
)

print(f"渲染颜色形状: {render_colors.shape}")  # (1, 3, 1080, 1920)
print(f"渲染深度形状: {render_depths.shape}")  # (1, 1, 1080, 1920)
```

---

## 版本信息

- 版权所有 © 2026 华为技术有限公司
- 许可证: CANN Open Software License Agreement Version 2.0

---
