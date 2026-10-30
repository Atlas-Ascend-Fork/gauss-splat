# gauss-splat

## Latest News

- [2026/10] Added support for Ascend 950PR/Ascend 950DT products, along with SIMT rendering, FusedSSIM, and other operators.
- [2026/07] The gauss-splat project is initially launched, with open-source operators supporting Atlas A2/A3 series products.
- [2026/07] Supports core 3D Gaussian Splatting operators, including spherical harmonics, covariance computation, projection, sorting, and filtering.

## Overview

gauss-splat is a 3D Gaussian Splatting rendering acceleration library based on [CANN](https://hiascend.com/software/cann). It provides high-performance PyTorch extension interfaces that cover the entire 3DGS training and inference workflow. This library accelerates core computations through Ascend C operators, achieving significant performance improvements on Atlas A2/A3 and Ascend 950PR/950DT products.

## Version Compatibility

The source code of this project is released along with CANN software versions. For the mapping between CANN software versions and project tags, refer to the corresponding version description in the [release repository](https://gitcode.com/cann/release-management).

## Environment Preparation

### System Requirements

| Item      | Requirement                                              |
| :-------- | :------------------------------------------------ |
| Hardware      | Atlas A2 training/inference series products/Atlas A3 training/inference series products/Ascend 950PR/Ascend 950DT |
| CANN version  | 9.0.0 or later                                       |
| Python    | >=3.9                                           |
| PyTorch   | Compatible with CANN version, >=2.7.1                               |
| torch_npu | Compatible with torch version, >=v26.0.0                             |

CANN Environment Preparation

> **Note**: When only compiling this project without running it (compile-time state), you only need to install the CANN toolkit package. When running this project (run-time state), you need to install the NPU driver, CANN toolkit package, and CANN ops package.

1. **Install the NPU driver (run-time dependency)**: Run `npu-smi info` to check whether NPU information is displayed. If there is no NPU information or the chip model cannot be clearly identified, refer to [CANN Quick Installation](https://www.hiascend.com/cann/download) to install the driver.
2. **Install CANN software packages**: The execution of this sample depends on the CANN toolkit (cann-toolkit) and CANN binary operator package (cann-kernels). The CANN software version used is CANN 9.0.0. Refer to [CANN Quick Installation](https://www.hiascend.com/cann/download), select the **stable version**, and download the corresponding packages based on the product series, CPU architecture, and OS (`Ascend-cann-toolkit_${version}_linux-${arch}.run` and `Ascend-cann-${chip_type}-ops_${version}_linux-${arch}.run`). Then follow the commands provided on the page to complete the installation.

   > **Note**: The `npu-smi info` output may not directly indicate the 910C chip model. The 910C chip belongs to the Atlas A3 training/inference series products, and its corresponding SoC code in the CANN ops package is `a3`, that is, `Ascend-cann-a3-ops_${version}_linux-${arch}.run`.
3. The torch and torch_npu versions required by this sample are **2.7.1 and v26.0.0**. Download and install the torch and torch_npu packages from [Ascend Extension for PyTorch plugin](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md).

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
# The `--python` parameter specifies the Python version used for compilation, supporting version 3.9 and later.

# After successful compilation, `build` and `dist` folders are generated in the current directory, and the generated whl package is in the `dist` directory.
pip install dist/*.whl --force-reinstall

# Method 2:
pip install . --no-build-isolation
```

For Atlas A5, build in a matching A5 CANN environment with `bash build.sh --python=3.9 --a5`. See the [A5 training example](examples/arch35/README.md).

After installation, save the Quick Start code outside the project source directory and run it from outside that directory. For example, save the script as `quick_start.py` next to the `gauss-splat` directory, then run:

```bash
cd ..
python quick_start.py
```

Do not run the script directly from the project root. Otherwise, Python imports the source-tree `gauss_splat` package before the installed package that contains the `gsplat_c` extension, resulting in `ModuleNotFoundError: No module named 'gauss_splat.gsplat_c'`.

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

print(f"Rendered color shape: {render_colors.shape}")  # (1, 1080, 1920, 3)
print(f"Rendered depth shape: {render_depths.shape}")  # (1, 1080, 1920, 1)
```

### Others

- Complete training/inference samples are available in `examples`.
- For more interface usage, refer to the test code in the tests directory.

## Python API

| Python Interface Name                   | Description                                                                                        |
| :--------------------------------------- | :----------------------------------------------------------------------------------------------- |
| `spherical_harmonics`                  | Spherical harmonics computation, supports automatic backward propagation, converts direction vectors to view-dependent colors                                     |
| `projection_three_dims_gaussian_fused` | Projection filtering fused operator, complete 3D-to-2D projection pipeline (covariance computation + projection transformation + Gaussian filtering), supports automatic backward propagation       |
| `calc_render`                          | Gaussian splatting rendering computation, performs tile-based alpha blending and depth computation, automatically selecting the implementation based on the built backend (clip_last_ids / clip_gsids), supports automatic backward propagation |
| `gaussian_sort`                        | Gaussian sphere depth sorting, performs tile-based depth sorting on Gaussian spheres for the rendering pipeline                                 |
| `flash_gaussian_build_mask`            | Flash rendering mask construction, builds the masks and indices required for Flash Gaussian Splatting rendering                              |
| `gaussian_filter`                      | Gaussian sphere filtering (low-level API), independent filtering operator, typically called automatically within `projection_three_dims_gaussian_fused` |
| `fused_ssim`                           | Fused SSIM (structural similarity) forward/backward computation for training loss, supports automatic backward propagation                              |
| `Rasterizer`                           | Complete rasterizer (recommended), encapsulates the complete rendering pipeline (projection + sorting + tile construction + rendering computation)                            |
| `get_render_schedule_cpp`              | Rendering schedule helper, obtains rendering schedule information (internal helper function)                                                   |

**Detailed Interface Description**: For specific parameters and return values of each API, refer to the [API Reference Documentation](docs/en/gauss_splat_api_en.md).

## Underlying Operator Interface (ACLNN)

For advanced users who need to directly call underlying operators, this project provides ACLNN-level C++/Python interfaces. For detailed API documentation, refer to [docs/en/kernels/](docs/en/kernels/).

| No. | Operator Name                      | Python Interface                                                | Supported Products | Documentation Link                                               |
| :--: | :--------------------------------- | :-------------------------------------------------------------- | :----------------: | :-------------------------------------------------------------- |
|  1  | SphericalHarmonicsForward          | `gauss_splat.gsplat_c.spherical_harmonics_forward`            | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnSphericalHarmonicsForward_en.md)          |
|  2  | SphericalHarmonicsBwd              | `gauss_splat.gsplat_c.spherical_harmonics_bwd`                | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnSphericalHarmonicsBwd_en.md)              |
|  3  | QuatScalesToCovars                 | `gauss_splat.gsplat_c.quat_scales_to_covars`                  | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnQuatScalesToCovars_en.md)                 |
|  4  | ProjectionThreeDimsGaussianForward | `gauss_splat.gsplat_c.projection_three_dims_gaussian_forward` | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnProjectionThreeDimsGaussianForward_en.md) |
|  5  | GaussianSort                       | `gauss_splat.gsplat_c.gaussian_sort`                          | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnGaussianSort_en.md)                       |
|  6  | GaussianFilter                     | `gauss_splat.gsplat_c.gaussian_filter`                        | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnGaussianFilter_en.md)                     |
|  7  | FullyFusedProjectionBwd            | `gauss_splat.gsplat_c.fully_fused_projection_bwd`             | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFullyFusedProjectionBwd_en.md)            |
|  8  | CalcRenderFwdDoubleClipGsids       | `gauss_splat.gsplat_c.calc_render_fwd_double_clip_gsids`      | A2/A3             | [API Documentation](docs/en/kernels/aclnnCalcRenderFwdDoubleClipGsids_en.md)       |
|  9  | CalcRenderBwdVarClipGsids          | `gauss_splat.gsplat_c.calc_render_bwd_var_clip_gsids`         | A2/A3             | [API Documentation](docs/en/kernels/aclnnCalcRenderBwdVarClipGsids_en.md)          |
|  10  | FlashGaussianBuildMask             | `gauss_splat.gsplat_c.flash_gaussian_build_mask`              | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFlashGaussianBuildMask_en.md)             |
|  11  | CalcRenderFwdClipLastids       | `gauss_splat.gsplat_c.calc_render_fwd_clip_last_ids`          | A5                | [API Documentation](docs/en/kernels/aclnnCalcRenderFwdClipLastids_en.md)                |
|  12  | CalcRenderBwdClipLastids       | `gauss_splat.gsplat_c.calc_render_bwd_clip_last_ids`          | A5                | [API Documentation](docs/en/kernels/aclnnCalcRenderBwdClipLastids_en.md)                |
|  13  | FusedSsimForward                   | `gauss_splat.gsplat_c.fused_ssim_forward`                     | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFusedSsimForward_en.md)                   |
|  14  | FusedSsimBackward                  | `gauss_splat.gsplat_c.fused_ssim_backward`                    | A2/A3/A5          | [API Documentation](docs/en/kernels/aclnnFusedSsimBackward_en.md)                  |

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
│   │   ├── calc_render.py               # Rendering computation (simt / clip_gsids)
│   │   ├── fused_ssim.py                # Fused SSIM (training loss)
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
│   ├── calc_render_fwd_clip_last_ids/
│   ├── calc_render_bwd_clip_last_ids/
│   ├── fused_ssim_forward/
│   ├── fused_ssim_backward/
│   └── flash_gaussian_build_mask/
├── docs/                   # Documentation directory
│   └── kernels/            # ACLNN operator API documentation
├── examples/               # Training/inference samples
├── tests/                  # Test code
│   ├── host/               # Host (CPU) tests
│   └── kernels/            # C++ operator tests
│   └── ops/                # Python operator tests
└── README.md               # This document
```

## Testing

Before running the tests, complete the compilation and installation as described in the Installation section above, and load the CANN environment (`source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh`).

### Python Tests

`tests/` contains the operator cases and `tests/host/` contains pure CPU cases. Run them outside the repository to avoid importing the source-tree `gauss_splat` package:

```bash
REPO=<absolute path to the gauss-splat repository>; cd /tmp
python -m pytest -v "$REPO/tests"
python -B -m unittest discover -s "$REPO/tests/host" -p 'test_*.py' -v
```

### C++ Operator Tests

```bash
cd <gauss-splat repository path>
bash tests/kernels/build_kernel_tests.sh   # Build; binaries go to tests/kernels/build_out/
bash tests/kernels/run_kernel_tests.sh     # Run and print results
```

> Notes:
> - `build_kernel_tests.sh` links the package's `aclnn_*.h` into `${ASCEND_TOOLKIT_HOME}/include/aclnnop/`. If that directory is not writable the link fails with `aclnnop/...h: No such file or directory`; use a writable include directory or adjust the directory permissions.
> - The render forward/backward operators are architecture-specific: A2/A3 uses `test_calc_render_fwd_double_clip_gsids` and `test_calc_render_bwd_var_clip_gsids`, while A5 uses `test_calc_render_fwd_clip_last_ids` and `test_calc_render_bwd_clip_last_ids`. The script automatically skips the cases that do not apply to the current build vendor (`customize`/`customize_arch35`).

## More Information

- [Operator API Documentation](docs/en/kernels/): Detailed description of underlying ACLNN operators
- [Operator Tests](tests/kernels/): C++ test code (see the Testing section above for how to run)
- [Contribution Guide](CONTRIBUTING.md): How to participate in project development
- [Security Statement](SECURITY.md): Security usage guide
- [License](LICENSE): CANN Open Software License Agreement Version 2.0
- [SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/render): CANN community render SIG

---

Note: The features and documentation of this project are being continuously updated and improved. Stay tuned for the latest version.

- **Issue Feedback**: Submit issues through GitCode [Issues](https://gitcode.com/cann/gauss-splat/issues).
- **Community Interaction**: Participate in discussions through GitCode [Discussions](https://gitcode.com/cann/gauss-splat/discussions).
- **Technical Articles**: Access technical articles through GitCode [Wiki](https://gitcode.com/cann/gauss-splat/wiki).
