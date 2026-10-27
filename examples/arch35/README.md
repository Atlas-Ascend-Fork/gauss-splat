# Atlas A5 3DGS 训练样例

本目录是 Atlas A5 的单卡训练与评估流程。`examples/arch22` 是 A2/A3 流程；两个样例使用同一套 `gauss_splat` 扩展，但训练脚本和策略实现不同。

## 环境与安装

在 Atlas A5 Linux 机器上安装与硬件匹配的 NPU 驱动、CANN toolkit、A5 ops 包，以及彼此兼容的 PyTorch 和 torch_npu。具体版本以目标机器的 CANN 发布配套关系为准。先加载 CANN 环境变量，再从仓库根目录编译 A5 扩展：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --python=3.9 --a5
pip install dist/*.whl --force-reinstall
cd examples/arch35
pip install -r requirements.txt
```

`--python` 应与当前环境的 Python 次版本一致。不要从旧版 `cann-recipes-spatial-intelligence/ops/ascendc` 编译本仓库算子；本仓库的 A5 实现由根目录 `build.sh --a5` 选择。

## 数据与运行

在 `examples/arch35` 目录执行：

```bash
python datasets/download_dataset.py
python train.py --data_dir data/360_v2/garden --result_dir results/garden
python train.py --data_dir data/360_v2/garden --result_dir results/garden --ckpt results/garden/ckpts/ckpt_29999_rank0.pt
```

`train_all.sh` 用于按脚本中的场景列表批量训练。`rasterization/config.py` 定义训练配置，`rasterization/runner.py` 驱动训练与评估，`rasterization/rasterizer.py` 实现 A5 渲染流程。

## 引用

```bibtex
@article{kerbl3Dgaussians,
    author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal={ACM Transactions on Graphics},
    volume={42},
    number={4},
    year={2023}
}

@article{ye2025gsplat,
    title={gsplat: An open-source library for Gaussian splatting},
    author={Ye, Vickie and Li, Ruilong and Kerr, Justin and others},
    journal={Journal of Machine Learning Research},
    volume={26},
    year={2025}
}
```
