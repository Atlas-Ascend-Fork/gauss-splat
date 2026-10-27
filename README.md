# gauss-splat

## 🔥Latest News

- [2026/10] 新增Ascend 950PR/Ascend 950DT产品支持，新增SIMT渲染、FusedSSIM等算子。
- [2026/07] gauss-splat项目首次上线，开源算子支持Atlas A2/A3系列产品。
- [2026/07] 支持3D Gaussian Splatting核心算子，包括球谐函数、协方差计算、投影、排序、筛选等。

## 🚀概述

gauss-splat是基于[CANN](https://hiascend.com/software/cann)的3D Gaussian Splatting渲染加速库，提供高性能的PyTorch扩展接口，覆盖3DGS训练和推理全流程。本库通过Ascend C算子加速核心计算，在Atlas A2/A3与Ascend 950PR/950DT产品上实现显著性能提升。

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。

## 🛠️环境准备

### 系统要求

| 项目      | 要求                                              |
| :-------- | :------------------------------------------------ |
| 硬件      | Atlas A2训练推理系列产品/Atlas A3训练推理系列产品/Ascend 950PR/Ascend 950DT |
| CANN版本  | 9.0.0及以上                                       |
| Python    | >=3.9                                           |
| PyTorch   | 配套CANN版本，>=2.7.1                               |
| torch_npu | 配套torch版本，>=26.0.0                            |

CANN环境准备

> **说明**：仅编译本项目不运行（编译态）时，只需安装CANN toolkit包；运行本项目（运行态）时，需安装NPU驱动、CANN toolkit包与CANN ops包。

1. **安装NPU驱动（运行态依赖）**：使用`npu-smi info`检查是否有NPU相关信息，若没有或无法明确识别芯片型号，请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》完成驱动安装。
2. **安装CANN软件包**：本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），使用CANN软件版本为CANN 9.0.0。请参考《[CANN快速安装](https://www.hiascend.com/cann/download)》，选择`稳定版本`，并根据产品系列、CPU架构、操作系统等下载对应软件包（`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-${chip_type}-ops_${version}_linux-${arch}.run`），最后参考页面提供的命令完成安装。

   > **说明**：`npu-smi info`输出可能无法直接体现910C芯片型号。910C所属产品系列为Atlas A3训练推理系列产品，对应的CANN Ops安装包SoC代号为`a3`，即`Ascend-cann-a3-ops_${version}_linux-${arch}.run`。
3. 本样例依赖的**torch与torch_npu版本为2.7.1和v26.0.0**，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)下载并安装torch与torch_npu安装包。

## ⬇️安装

```bash
``编译安装
conda create -n 3dgs python=3.9
conda activate 3dgs

git clone -b ${tag_version} https://gitcode.com/cann/gauss-splat.git
cd gauss-splat
source ${CAN_INSTALL_PATH}/ascend-toolkit/set_env.sh

pip install numpy==1.23 decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado cmake pyyaml expecttest protobuf

方法1：
bash build.sh --python=3.9
参数`--python`指定编译使用的python版本，支持3.9及以上版本。编译成功后会在当前目录下生成`build`,  `dist`文件夹，生成的whl包在`dist`目录下。

pip install dist/*.whl --force-reinstall

方法2：
pip install . --no-build-isolation
```

Atlas A5 请在配套的 A5 CANN 环境中使用 `bash build.sh --python=3.9 --a5` 编译，并参考 [A5 训练样例](examples/arch35/README.md)。

安装完成后，请将快速开始代码保存到项目源码目录之外，并从源码目录之外运行。例如，将脚本保存为与`gauss-splat`目录同级的`quick_start.py`后执行：

```bash
cd ..
python quick_start.py
```

请勿在项目根目录下直接运行该脚本。否则Python会优先导入源码目录中的`gauss_splat`包，而不是已安装的包含`gsplat_c`扩展的包，进而导致`ModuleNotFoundError: No module named 'gauss_splat.gsplat_c'`。

## 🚀快速开始

### 完整渲染流程（使用Rasterizer）

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

print(f"渲染颜色形状: {render_colors.shape}")  # (1, 1080, 1920, 3)
print(f"渲染深度形状: {render_depths.shape}")  # (1, 1080, 1920, 1)
```

### 其他

- 完整训练/推理样例可参考 `examples/arch22`（A2/A3）与 `examples/arch35`（A5）。
- 更多接口用法参考tests目录下的测试代码。

## 📖Python API

| Python接口名                             | 功能说明                                                                                         |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | 球谐函数计算，支持自动反向传播，将方向向量转换为视角相关颜色                                     |
| `projection_three_dims_gaussian_fused` | 投影筛选融合算子，完整的3D→2D投影流水线（协方差计算+投影变换+高斯筛选），支持自动反向传播       |
| `calc_render`                          | 高斯溅射渲染计算，执行基于Tile的Alpha混合与深度计算，根据编译后端（simt / clip_gsids）自动选择实现，支持自动反向传播 |
| `gaussian_sort`                        | 高斯球深度排序，对高斯球按深度进行Tile-based排序，用于渲染流水线                                 |
| `flash_gaussian_build_mask`            | Flash渲染掩码构建，构建Flash Gaussian Splatting渲染所需的掩码和索引                              |
| `gaussian_filter`                      | 高斯球筛选（低级API），独立筛选算子，通常在 `projection_three_dims_gaussian_fused`内部自动调用 |
| `fused_ssim`                           | 融合SSIM（结构相似度）前向/反向计算，用于训练损失，支持自动反向传播                              |
| `Rasterizer`                           | 完整渲染器（推荐），封装完整渲染流水线（投影+排序+Tile构建+渲染计算）                            |
| `get_render_schedule_cpp`              | 渲染调度辅助，获取渲染调度信息（内部辅助函数）                                                   |

**详细接口说明**： 各API的具体参数和返回值见[API参考文档](docs/zh/gauss_splat_api.md)

## 🔧底层算子接口（ACLNN）

对于需要直接调用底层算子的高级用户，本项目提供ACLNN级别的C++/Python接口。详细API文档见[docs/zh/kernels/](docs/zh/kernels/)。

| 序号 | 算子名称                           | Python接口                                                      | 支持产品   | 文档链接                                                        |
| :--: | :--------------------------------- | :-------------------------------------------------------------- | :--------: | :-------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnSphericalHarmonicsForward.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnSphericalHarmonicsBwd.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnQuatScalesToCovars.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnProjectionThreeDimsGaussianForward.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnGaussianSort.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnGaussianFilter.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFullyFusedProjectionBwd.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      |  A2/A3    | [API文档](docs/zh/kernels/aclnnCalcRenderFwdDoubleClipGsids.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         |  A2/A3    | [API文档](docs/zh/kernels/aclnnCalcRenderBwdVarClipGsids.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFlashGaussianBuildMask.md)             |
|  11  | CalcRenderSimtFwd                  | `gauss_splat.gsplat_c.calc_render_simt_fwd`                   |    A5     | [API文档](docs/zh/kernels/aclnnCalcRenderSimtFwd.md)                  |
|  12  | CalcRenderSimtBwd                  | `gauss_splat.gsplat_c.calc_render_simt_bwd`                   |    A5     | [API文档](docs/zh/kernels/aclnnCalcRenderSimtBwd.md)                  |
|  13  | FusedSsimForward                   | `gauss_splat.gsplat_c.fused_ssim_forward`                     | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFusedSsimForward.md)                   |
|  14  | FusedSsimBackward                  | `gauss_splat.gsplat_c.fused_ssim_backward`                    | A2/A3/A5  | [API文档](docs/zh/kernels/aclnnFusedSsimBackward.md)                  |

**注意**：直接调用底层算子需要手动处理shape变换和内存管理，推荐使用高层Python API。

## 🏗️项目结构

```text
gauss-splat/
├── gauss_splat/            # Python封装层（用户接口）
│   ├── ops/                # 高层Python算子接口
│   │   ├── spherical_harmonics.py       # 球谐函数（自动backward）
│   │   ├── projection_three_dims_gaussian_fused.py  # 投影融合算子
│   │   ├── gaussian_sort.py             # 排序算子
│   │   ├── flash_gaussian_build_mask.py # Flash掩码构建
│   │   ├── gaussian_filter.py           # 筛选算子
│   │   ├── calc_render.py               # 渲染计算（simt / clip_gsids）
│   │   ├── fused_ssim.py                # 融合SSIM（训练损失）
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
│   ├── calc_render_simt_fwd/
│   ├── calc_render_simt_bwd/
│   ├── fused_ssim_forward/
│   ├── fused_ssim_backward/
│   └── flash_gaussian_build_mask/
├── docs/                   # 文档目录
│   └── kernels/            # ACLNN算子API文档
├── examples/               # 训练/推理样例
│   ├── arch22/             # A2/A3样例
│   └── arch35/             # A5样例
├── tests/                  # 测试代码
│   ├── test_spherical_harmonics.py      # Python测试
│   ├── host/               # host(CPU)测试
│   └── kernel_tests/      # C++算子测试
└── README.md               # 本文档
```

## 🧪测试

测试前请先参考上文《安装》完成编译安装，并加载 CANN 环境（`source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh`）。

### Python 测试

`tests/` 为算子 Python 用例，`tests/host/` 为纯 CPU 用例。请从仓库外执行，避免导入源码目录中的 `gauss_splat`：

```bash
REPO=<gauss-splat 仓库绝对路径>; cd /tmp
python -m pytest -v "$REPO/tests"
python -B -m unittest discover -s "$REPO/tests/host" -p 'test_*.py' -v
```

### C++ 算子测试

```bash
cd <gauss-splat 仓库路径>
bash tests/kernel_tests/build_kernel_tests.sh   # 编译，产物在 tests/kernel_tests/build_out/
bash tests/kernel_tests/run_kernel_tests.sh     # 运行并打印结果
```

> 说明：
> - `build_kernel_tests.sh` 会把算子包内的 `aclnn_*.h` 链接到 `${ASCEND_TOOLKIT_HOME}/include/aclnnop/`。若该目录不可写，链接会失败并报 `aclnnop/...h: No such file or directory`，请改用可写的 include 目录或调整该目录权限。
> - 渲染正/反向算子按架构区分：A2/A3 使用 `test_calc_render_fwd_double_clip_gsids`、`test_calc_render_bwd_var_clip_gsids`；A5 使用 `test_calc_render_simt_fwd`。脚本会根据当前构建的 vendor（`customize`/`customize_arch35`）自动跳过不适用的用例。

## 💬更多信息

- [算子API文档](docs/zh/kernels/)：底层ACLNN算子详细说明
- [算子测试](tests/kernel_tests/)：C++测试代码（运行方式见上文《测试》）
- [贡献指南](CONTRIBUTING.md)：如何参与项目开发
- [安全声明](SECURITY.md)：安全使用指南
- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/render)：CANN社区render SIG

---

PS：本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues)提交问题。
- **社区互动**：通过GitCode [讨论](https://gitcode.com/cann/gauss-splat/discussions)参与交流。
- **技术专栏**：通过GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki)获取技术文章。
