# gauss-splat

## 🔥Latest News

- [2026/07] gauss-splat项目首次上线，开源算子支持Atlas A2/A3系列产品。
- [2026/07] 支持10个3D Gaussian Splatting核心算子，包括球谐函数、协方差计算、投影、排序、筛选等。

## 🚀概述

gauss-splat是基于[CANN](https://hiascend.com/software/cann)的3D Gaussian Splatting渲染加速库，提供高性能的PyTorch扩展接口，覆盖3DGS训练和推理全流程。本库通过Ascend C算子加速核心计算，在Atlas A2/A3产品上实现显著性能提升。

## 📌版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明。

## 🛠️环境准备

### 系统要求

| 项目      | 要求                                              |
| :-------- | :------------------------------------------------ |
| 硬件      | Atlas A2训练推理系列产品/Atlas A3训练推理系列产品 |
| CANN版本  | 8.5.0及以上（详见tag）                                       |
| Python    | >=3.7                                           |
| PyTorch   | 配套CANN版本，>=2.7.1                               |
| torch_npu | 配套torch版本 ，>=7.3.0                             |

CANN环境准备

1. 本样例的执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），使用CANN软件版本为CANN 8.5。 请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run`与 `Ascend-cann-${chip_type}-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu)进行安装。
2. 本样例依赖的**torch与torch_npu版本为2.7.1和v7.3.0**，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)下载并安装torch与torch_npu安装包。

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
参数`--python`指定编译使用的python版本，支持3.8及以上版本。编译成功后会在当前目录下生成`build`,  `dist`文件夹，生成的whl包在`dist`目录下。
pip install dist/*.whl --force-reinstall

方法2：
pip install . --no-build-isolation
```

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

### 测试

NPU 测试需要 Linux、Atlas A2/A3、上述配套版本的 CANN、PyTorch、torch_npu 和 ACL Python 接口，并已完成项目 wheel 的编译安装。先激活安装项目的 Python 环境并加载 CANN 环境变量。测试默认使用 `npu:0`。

在仓库根目录执行以下命令。`CAN_INSTALL_PATH` 为 CANN 安装根目录，与上文安装命令中的路径一致。切换到仓库外运行 Python 测试，避免源码目录遮蔽已安装的 `gauss_splat.gsplat_c` 扩展；也请勿将仓库根目录加入 `PYTHONPATH`。

```bash
REPO="$(pwd)"
source "${CAN_INSTALL_PATH}/ascend-toolkit/set_env.sh"
python -m pip install pytest
cd /tmp
python -c "import acl, torch, torch_npu, gauss_splat.gsplat_c; print(torch.npu.is_available())"

# 渲染前向和反向精度回归
python -m pytest -v --tb=long "$REPO/tests/test_calc_render_fwd.py" "$REPO/tests/test_calc_render_bwd.py"
# 全部 Python 测试
python -m pytest -v --tb=long "$REPO/tests"
# 也可单独运行测试脚本
python "$REPO/tests/test_calc_render_fwd.py"
# 指定单个 UT 用例
python -m pytest "$REPO/tests/test_spherical_harmonics.py::TestSphericalHarmonicsForward::test_spherical_harmonics_forward"

# C++ 算子调用测试，不能替代 Python 精度回归
bash "$REPO/tests/kernel_tests/build_kernel_tests.sh" && bash "$REPO/tests/kernel_tests/run_kernel_tests.sh"
```

C++ 构建日志位于 `tests/kernel_tests/build_out/build.log`，运行汇总位于同目录的 `run_results.txt`，各用例的标准输出和错误保存在 `*_run.log`。编译失败、可执行文件缺失、运行失败或没有测试时，脚本返回非零退出码。

参考实现和诊断工具的 CPU 回归仅需 PyTorch 和 NumPy，不需要 CANN 或 NPU。在新的终端中从仓库根目录执行：

```bash
REPO="$(pwd)"
python -m unittest discover -s "$REPO/tests" -p test_render_test_utils.py -v
# 测试脚本的模拟回归，仅需 Python 和 Bash
python -m unittest discover -s "$REPO/tests" -p test_kernel_test_scripts.py -v
```

渲染测试使用固定种子 `1234`、正定协方差和独立的 CPU 参考输入；反向测试同时验证七类梯度。比较保留当前安装的 `torch_npu.testing.TestCase.assertRtolEqual` 默认规则，不放宽容差。失败信息包含输出字段、用例、shape、dtype、设备、比较器默认参数、误差统计和最差位置；非有限值单独报错。参考实现按当前内核在完整四高斯组前检查透射率，剩余不足四个时不再检查。

若仍失败，请保留完整日志并记录硬件型号及 CANN、torch、torch_npu 版本；不能仅凭精度断言失败判断为硬件或内核缺陷。

## 📖Python API

| Python接口名                             | 功能说明                                                                                         |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | 球谐函数计算，支持自动反向传播，将方向向量转换为视角相关颜色                                     |
| `projection_three_dims_gaussian_fused` | 投影筛选融合算子，完整的3D→2D投影流水线（协方差计算+投影变换+高斯筛选），支持自动反向传播       |
| `gaussian_sort`                        | 高斯球深度排序，对高斯球按深度进行Tile-based排序，用于渲染流水线                                 |
| `flash_gaussian_build_mask`            | Flash渲染掩码构建，构建Flash Gaussian Splatting渲染所需的掩码和索引                              |
| `gaussian_filter`                      | 高斯球筛选（低级API），独立筛选算子，通常在 `projection_three_dims_gaussian_fused`内部自动调用 |
| `Rasterizer`                           | 完整渲染器（推荐），封装完整渲染流水线（投影+排序+Tile构建+渲染计算）                            |
| `get_render_schedule_cpp`              | 渲染调度辅助，获取渲染调度信息（内部辅助函数）                                                   |

**详细接口说明**： 各API的具体参数和返回值见[API参考文档](docs/zh/gauss_splat_api.md)

## 🔧底层算子接口（ACLNN）

对于需要直接调用底层算子的高级用户，本项目提供ACLNN级别的C++/Python接口。详细API文档见[docs/zh/kernels/](docs/zh/kernels/)。

| 序号 | 算子名称                           | Python接口                                                      | 文档链接                                                        |
| :--: | :--------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | [API文档](docs/zh/kernels/aclnnSphericalHarmonicsForward.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | [API文档](docs/zh/kernels/aclnnSphericalHarmonicsBwd.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | [API文档](docs/zh/kernels/aclnnQuatScalesToCovars.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | [API文档](docs/zh/kernels/aclnnProjectionThreeDimsGaussianForward.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | [API文档](docs/zh/kernels/aclnnGaussianSort.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | [API文档](docs/zh/kernels/aclnnGaussianFilter.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | [API文档](docs/zh/kernels/aclnnFullyFusedProjectionBwd.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      | [API文档](docs/zh/kernels/aclnnCalcRenderFwdDoubleClipGsids.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         | [API文档](docs/zh/kernels/aclnnCalcRenderBwdVarClipGsids.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | [API文档](docs/zh/kernels/aclnnFlashGaussianBuildMask.md)             |

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

- [算子API文档](docs/zh/kernels/)：底层ACLNN算子详细说明
- [测试执行说明](tests/README.md)：Python UT、无 NPU 用例与 C++ 算子测试的具体命令
- [C++ 算子测试代码](tests/kernel_tests/)：构建与运行脚本及测试用例
- [贡献指南](CONTRIBUTING.md)：如何参与项目开发
- [安全声明](SECURITY.md)：安全使用指南
- [许可证](LICENSE)：CANN Open Software License Agreement Version 2.0
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/render)：CANN社区render SIG

---

PS：本项目功能和文档正在持续更新和完善中，欢迎您关注最新版本。

- **问题反馈**：通过GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues)提交问题。
- **社区互动**：通过GitCode [讨论](https://gitcode.com/cann/gauss-splat/discussions)参与交流。
- **技术专栏**：通过GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki)获取技术文章。
