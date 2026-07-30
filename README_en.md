# gauss-splat

## Latest News

- [2026/06] The gauss-splat project is initially launched, with open-source operators supporting Atlas A2/A3 series products.
- [2026/06] Supports 10 core 3D Gaussian Splatting operators, including spherical harmonics, covariance computation, projection, sorting, and filtering.

## Overview

gauss-splat is a 3D Gaussian Splatting rendering acceleration library based on [CANN](https://hiascend.com/software/cann). It provides high-performance PyTorch extension interfaces that cover the entire 3DGS training and inference workflow. This library accelerates core computations through Ascend C operators, achieving significant performance improvements on Atlas A2/A3 products.

## Version Compatibility

The source code of this project is released along with CANN software versions. For the mapping between CANN software versions and project tags, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).

## Environment Preparation

### System Requirements

| Item      | Requirement                                              |
| :-------- | :------------------------------------------------ |
| Hardware      | Atlas A2 training/inference series products/Atlas A3 training/inference series products |
| CANN version  | 8.5.0 or later (see tag)                                       |
| Python    | >=3.7                                           |
| PyTorch   | Compatible with CANN version, >=2.7.1                               |
| torch_npu | Compatible with torch version, >=7.3.0                             |

CANN Environment Preparation

1. The execution of this sample depends on the CANN toolkit (cann-toolkit) and CANN binary operator package (cann-kernels). The CANN software version used is CANN 8.5. Download `Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-${chip_type}-ops_${version}_linux-${arch}.run` packages from the [CANN software package download address](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0), and refer to the [CANN installation documentation](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu) for installation.
2. The torch and torch_npu versions required by this sample are **2.7.1 and v7.3.0**. Download and install the torch and torch_npu packages from [Ascend Extension for PyTorch plugin](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md).

## Installation

```bash
# Compile and install
conda create -n 3dgs python=3.9
conda activate 3dgs

git clone -b ${tag_version} https://gitcode.com/cann/gauss-splat.git
cd gauss-splat
source ${CAN_INSTALL_PATH}/ascend-toolkit/set_env.sh

pip install numpy==1.23 decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado cmake pyyaml expecttest protobuf

# Method 1:
bash build.sh --python=3.9
# The `--python` parameter specifies the Python version used for compilation, supporting version 3.8 and later.
# After successful compilation, `build` and `dist` folders are generated in the current directory, and the generated whl package is in the `dist` directory.
pip install dist/*.whl --force-reinstall

# Method 2:
pip install . --no-build-isolation
```

## Quick Start

### Complete Rendering Pipeline (Using Rasterizer)

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

### Others

Refer to the test code in the tests directory.

## Python API

| Python Interface Name                   | Description                                                                                        |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | Spherical harmonics computation, supports automatic backward propagation, converts direction vectors to view-dependent colors                                     |
| `projection_three_dims_gaussian_fused` | Projection filtering fused operator, complete 3D-to-2D projection pipeline (covariance computation + projection transformation + Gaussian filtering), supports automatic backward propagation       |
| `gaussian_sort`                        | Gaussian sphere depth sorting, performs tile-based depth sorting on Gaussian spheres for the rendering pipeline                                 |
| `flash_gaussian_build_mask`            | Flash rendering mask construction, builds the masks and indices required for Flash Gaussian Splatting rendering                              |
| `gaussian_filter`                      | Gaussian sphere filtering (low-level API), independent filtering operator, typically called automatically within `projection_three_dims_gaussian_fused` |
| `Rasterizer`                           | Complete rasterizer (recommended), encapsulates the complete rendering pipeline (projection + sorting + tile construction + rendering computation)                            |
| `get_render_schedule_cpp`              | Rendering schedule helper, obtains rendering schedule information (internal helper function)                                                   |

**Detailed Interface Description**: For specific parameters and return values of each API, refer to the [API Reference Documentation](docs/en/gauss_splat_api_en.md).

## Underlying Operator Interface (ACLNN)

For advanced users who need to directly call underlying operators, this project provides ACLNN-level C++/Python interfaces. For detailed API documentation, refer to [docs/en/kernels/](docs/en/kernels/).

| No. | Operator Name                      | Python Interface                                                | Documentation Link                                               |
| :--: | :--------------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | [API Documentation](docs/en/kernels/aclnnSphericalHarmonicsForward_en.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | [API Documentation](docs/en/kernels/aclnnSphericalHarmonicsBwd_en.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | [API Documentation](docs/en/kernels/aclnnQuatScalesToCovars_en.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | [API Documentation](docs/en/kernels/aclnnProjectionThreeDimsGaussianForward_en.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | [API Documentation](docs/en/kernels/aclnnGaussianSort_en.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | [API Documentation](docs/en/kernels/aclnnGaussianFilter_en.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | [API Documentation](docs/en/kernels/aclnnFullyFusedProjectionBwd_en.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      | [API Documentation](docs/en/kernels/aclnnCalcRenderFwdDoubleClipGsids_en.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         | [API Documentation](docs/en/kernels/aclnnCalcRenderBwdVarClipGsids_en.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | [API Documentation](docs/en/kernels/aclnnFlashGaussianBuildMask_en.md)             |

**Note**: Directly calling underlying operators requires manual handling of shape transformations and memory management. Using the high-level Python API is recommended.

## Project Structure

```text
gauss-splat/
├── gauss_splat/            # Python wrapper layer (user interface)
│   ├── ops/                # High-level Python operator interfaces
│   │   ├── spherical_harmonics.py       # Spherical harmonics (automatic backward)
│   │   ├── projection_three_dims_gaussian_fused.py  # Projection fused operator
│   │   ├── gaussian_sort.py             # Sorting operator
│   │   ├── flash_gaussian_build_mask.py # Flash mask construction
│   │   ├── gaussian_filter.py           # Filtering operator
│   │   ├── calc_render.py               # Rendering computation
│   │   ├── rendering.py                 # Rasterizer class
│   │   └── get_render_schedule.py       # Helper functions
│   ├── csrc/               # C++ binding implementation
├── kernels/                # Ascend C operator implementation (low-level)
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
├── docs/                   # Documentation directory
│   └── kernels/            # ACLNN operator API documentation
├── tests/                  # Test code
│   ├── test_spherical_harmonics.py      # Python tests
│   └── kernel_tests/      # C++ operator tests
└── README.md               # This document
```

## More Information

- [Operator API Documentation](docs/en/kernels/): Detailed description of underlying ACLNN operators
- [Operator Tests](tests/kernel_tests/): C++ test code and execution guide
- [Contribution Guide](CONTRIBUTING.md): How to participate in project development
- [Security Statement](SECURITY.md): Security usage guide
- [License](LICENSE): CANN Open Software License Agreement Version 2.0
- [SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/render): CANN community render SIG

---

Note: The features and documentation of this project are being continuously updated and improved. Stay tuned for the latest version.

- **Issue Feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues).
- **Community Interaction**: Participate in discussions through GitCode [Discussions](https://gitcode.com/cann/gauss-splat/discussions).
- **Technical Articles**: Access technical articles through GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki).
