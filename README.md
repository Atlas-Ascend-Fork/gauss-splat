# gauss-splat

## 🔥Latest News

- [2026/06] gauss-splat项目首次上线，开源算子支持Atlas A2/A3系列产品。
- [2026/06] 支持10个3D Gaussian Splatting核心算子，包括球谐函数、协方差计算、投影、排序、筛选等。

## 🚀概述

gauss-splat是基于[CANN](https://hiascend.com/software/cann)的3D Gaussian Splatting渲染加速库，提供高性能的PyTorch扩展接口，覆盖3DGS训练和推理全流程。本库通过Ascend C算子加速核心计算，在Atlas A2/A3产品上实现显著性能提升。

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。

## 🛠️环境准备

### 系统要求


| 项目      | 要求                                      |
| :-------- | :---------------------------------------- |
| 硬件      | Atlas A2训练系列产品/Atlas A3训练系列产品 |
| 操作系统  | Linux (Ubuntu 18.04/20.04, CentOS 7/8)    |
| CANN版本  | 8.0.RC1及以上                             |
| Python    | 3.7-3.9                                   |
| PyTorch   | 1.11.0及以上                              |
| torch_npu | 配套CANN版本                              |

[环境部署](docs/install/quick_install.md)是体验本项目能力的前提，请先完成NPU驱动、CANN包安装、torch_npu安装等，确保环境正常。

## ⬇️安装

```bash
``编译安装
git clone -b ${tag_version} https://gitcode.com/cann/gauss-splat.git
cd gauss-splat
source ${CAN_INSTALL_PATH}/ascend-toolkit/set_env.sh
pip install -e .

```

## 🚀快速开始

### 完整渲染流程（使用Rasterizer）

```python
from gauss_splat import Rasterizer

# 初始化渲染器
rasterizer = Rasterizer()

# 准备高斯球参数
N = 10000  # 高斯球数量
C = 1      # 相机数量
means = torch.randn(N, 3).npu()
quats = torch.randn(N, 4).npu()
scales = torch.randn(N, 3).npu()
opacities = torch.randn(N).npu()
colors = torch.randn(N, 16, 3).npu()  # 球谐系数

# 准备相机参数
viewmats = torch.randn(C, 4, 4).npu()
Ks = torch.randn(C, 3, 3).npu()

# 渲染输出
rendered_images = rasterizer.rasterization(
    (means, quats, scales, opacities, colors),
    (viewmats, Ks, "pinhole", "RGB"),
    sh_degree=3,
    width=800,
    height=600
)
```

## 📖Python API


| Python接口名                           | 功能说明                                                                                      |
| :------------------------------------- | :-------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | 球谐函数计算，支持自动反向传播，将方向向量转换为视角相关颜色                                  |
| `projection_three_dims_gaussian_fused` | 投影筛选融合算子，完整的3D→2D投影流水线（协方差计算+投影变换+高斯筛选），支持自动反向传播    |
| `gaussian_sort`                        | 高斯球深度排序，对高斯球按深度进行Tile-based排序，用于渲染流水线                              |
| `flash_gaussian_build_mask`            | Flash渲染掩码构建，构建Flash Gaussian Splatting渲染所需的掩码和索引                           |
| `gaussian_filter`                      | 高斯球筛选（低级API），独立筛选算子，通常在`projection_three_dims_gaussian_fused`内部自动调用 |
| `Rasterizer`                           | 完整渲染器（推荐），封装完整渲染流水线（投影+排序+Tile构建+渲染计算）                         |
| `get_render_schedule_cpp`              | 渲染调度辅助，获取渲染调度信息（内部辅助函数）                                                |

**详细接口说明**： 各API的具体参数和返回值见[API参考文档](docs/python_api.md)

## 🔧底层算子接口（ACLNN）

对于需要直接调用底层算子的高级用户，本项目提供ACLNN级别的C++/Python接口。详细API文档见[docs/kernels/](docs/kernels/)。


| 序号 | 算子名称                           | Python接口                                                    | 文档链接                                                           |
| :--: | :--------------------------------- | :------------------------------------------------------------ | :----------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | [API文档](docs/kernels/aclnnSphericalHarmonicsForward.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | [API文档](docs/kernels/aclnnSphericalHarmonicsBwd.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | [API文档](docs/kernels/aclnnQuatScalesToCovars.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | [API文档](docs/kernels/aclnnProjectionThreeDimsGaussianForward.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | [API文档](docs/kernels/aclnnGaussianSort.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | [API文档](docs/kernels/aclnnGaussianFilter.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | [API文档](docs/kernels/aclnnFullyFusedProjectionBwd.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      | [API文档](docs/kernels/aclnnCalcRenderFwdDoubleClipGsids.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         | [API文档](docs/kernels/aclnnCalcRenderBwdVarClipGsids.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | [API文档](docs/kernels/aclnnFlashGaussianBuildMask.md)             |

**注意**：直接调用底层算子需要手动处理shape变换和内存管理，推荐使用高层Python API。

## 🏗️项目结构

```
gauss-splat/
├── gauss_splat/            # Python封装层（用户接口）
│   ├── ops/                # 高层Python算子接口
│   │   ├── spherical_harmonics.py       # 球谐函数（自动backward）
│   │   ├── projection_three_dims_gaussian_fused.py  # 投影融合算子
│   │   ├── gaussian_sort.py             # 排序算子
│   │   ├── flash_gaussian_build_mask.py # Flash掩码构建
│   │   ├── gaussian_filter.py           # 筛选算子
│   │   ├── calc_render.py               # 渲染计算
│   │   ├── rendering.py                 # Rasterizer类
│   │   └── get_render_schedule.py       # 辅助函数
│   ├── csrc/               # C++ binding实现
├── kernels/                # Ascend C算子实现（底层）
│   ├── spherical_harmonics_forward/
│   ├── spherical_harmonics_bwd/
│   ├── quat_scales_to_covars/
│   ├── projection_three_dims_gaussian_forward/
│   ├── gaussian_sort/
│   ├── gaussian_filter/
│   ├── fully_fused_projection_bwd/
│   ├── calc_render_fwd_double_clip_gsids/
│   ├── calc_render_bwd_var_clip_gsids/
│   └── flash_gaussian_build_mask/
├── docs/                   # 文档目录
│   └── kernels/            # ACLNN算子API文档
├── tests/                  # 测试代码
│   ├── test_spherical_harmonics.py      # Python测试
│   └── kernel_tests/      # C++算子测试
└── README.md               # 本文档
```

## 💬更多信息

- [算子API文档](docs/kernels/)：底层ACLNN算子详细说明
- [算子测试](tests/kernel_tests/)：C++测试代码与执行指南
- [贡献指南](CONTRIBUTING.md)：如何参与项目开发
- [安全声明](SECURITY.md)：安全使用指南
- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/gauss-splat)：CANN社区SIG

---

PS：本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues)提交问题。
- **社区互动**：通过GitCode [讨论](https://gitcode.com/cann/gauss-splat/discussions)参与交流。
- **技术专栏**：通过GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki)获取技术文章。
