# Atlas A2/A3/A5 3DGS 训练样例

本目录是 Atlas A2/A3/A5 的单卡训练与评估流程。

## 环境与安装

在目标机器上安装与硬件匹配的 NPU 驱动、CANN toolkit，以及彼此兼容的 PyTorch 和 torch_npu（版本以目标机器的 CANN 发布配套关系为准）。先加载 CANN 环境变量，再从仓库根目录按架构编译扩展：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Ascend910B / Atlas A2/A3
bash build.sh --python=3.12

# Ascend950 / Atlas A5
bash build.sh --python=3.12 --a5

pip install dist/*.whl --force-reinstall
cd examples
pip install -r requirements.txt
```

`--python` 应与当前环境的 Python 次版本一致。不要从旧版 `cann-recipes-spatial-intelligence/ops/ascendc` 编译本仓库算子；本仓库的 A5 实现由根目录 `build.sh --a5` 选择。

## 数据与运行

在 `examples` 目录执行：

```bash
python datasets/download_dataset.py
python train.py --data_dir data/360_v2/garden --result_dir results/garden
python train.py --data_dir data/360_v2/garden --result_dir results/garden --ckpt results/garden/ckpts/ckpt_29999_rank0.pt

# 配置 densify 高斯球数量上限
python train.py --data_dir data/360_v2/garden --result_dir results/garden --densify --max-n-gaussian 1000000
```

### `max_n_gaussian` 作用说明

`--max-n-gaussian`（对应 `Config.max_n_gaussian`）用于覆盖 densify Strategy 的 `cap_max`：

- 仅在 `--densify` 开启时生效。
- `max_n_gaussian > 0`：覆盖 `strategy.cap_max`；`max_n_gaussian <= 0`（默认）：沿用 Strategy 默认值（ADC 6,000,000 / Default 3,000,000）。

### 高斯球 DFX 日志说明

```text
[GS]init rank=<rank> world_size=<world_size> num_GS=<初始化点数> strategy=<adc|default|off> max_n_gaussian=<实际生效上限|off> (cfg.max_n_gaussian=<CLI原始值>)
[DENSIFY][config] rank=<rank> world_size=<world_size> strategy=<adc|default> refine=[<start>-<stop>]/<every> reset_every=<reset_every> pause_refine_after_reset=<pause> grow_grad2d=<grow_grad2d> prune_opa=<prune_opa>
[DENSIFY] rank=<rank> step=<step> strategy=<adc|default> before=<density前点数> removed=<prune+capped> added=<dup+split> net=<after-before> after=<density后点数> cap(dup=<clone数> split=<split数> prune=<常规裁剪数> capped=<硬上限裁剪数>)
[GS]step=<step> rank=<rank> num_GS=<当前点数> max_n_gaussian=<实际生效上限|off>
```

- `[GS]init`：每个 rank 训练启动时打印一次，包含 `rank/world_size`；`max_n_gaussian` 为 CLI 覆盖后的实际值，括号内保留 `cfg.max_n_gaussian` 原值；关闭 densify 时策略为 `off`。
- `[DENSIFY][config]`：开启 densify 时每个 rank 打印一次，记录实际生效的 refine 窗口、周期、reset 周期和增密阈值，便于与 `adjust_steps` 后的配置核对。
- `[GS]step`：每个 rank 按 `tb_every` 周期打印，`num_GS` 为该 rank 的本地数量，用于多卡训练观察各 rank 的高斯球数量变化。
- `[DENSIFY]`：每个 rank 在每个 densify refine 步骤打印一次，`added=dup+split`（split 为净增 1），`removed=prune+capped`，`net=after-before`。
- `cap(...)`：本步 clone 复制数、split 数、常规裁剪数（不透明度/屏幕半径/尺度）和被硬上限裁掉的点数。

`train_all.sh` 用于按脚本中的场景列表批量训练。`rasterization/config.py` 定义训练配置，`rasterization/runner.py` 驱动训练与评估。

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
