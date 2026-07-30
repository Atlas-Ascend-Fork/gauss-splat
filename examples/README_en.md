# 3D Gaussian Splatting Training and Inference Example Based on Ascend Platform

3D Gaussian Splatting (3DGS) is a differentiable rendering algorithm proposed by French and German researchers in 2023. It uses 3D Gaussian ellipsoids as the core primitive and completes the reconstruction and rendering of 3D scenes through differentiable rasterization, breaking the bottleneck of traditional modeling where "speed and fidelity cannot be achieved simultaneously." This project aims to provide an Ascend-adapted version of 3DGS.

## Running the Example

This sample supports single-card training and inference on the Ascend Atlas A2 environment.

### CANN Environment Preparation

CANN version and gauss_splat library have been installed.

### Installing Python External Dependencies

```bash
cd examples
pip install -r requirements.txt
```

### Dataset Preparation

Execute in the `examples` path:

```bash
python datasets/download_dataset.py
```

The dataset is downloaded and extracted to the `examples/data/360_v2` path.

### Quick Start

This sample provides training and inference scripts for a single-card environment. Before executing the scripts, configure the environment variables by referring to the CANN software installation tutorial in the [Ascend community](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Run the training script:**

```bash
ASCEND_RT_VISIBLE_DEVICES=1 python train.py
```

`ASCEND_RT_VISIBLE_DEVICES=1` specifies NPU device card 1. If not specified, card 0 is started by default. The configuration file for model training and inference is `rasterization/config.py`. `--data_dir` specifies the training dataset path, which defaults to `data/360_v2/garden`. `--result_dir` specifies the training result output path, which defaults to `results/garden`.

```bash
python train.py --data_dir data/360_v2/bonsai --result_dir results/bonsai
```

**Run the inference script:**

```bash
python train.py --data_dir data/360_v2/bonsai --result_dir results/bonsai --ckpt path_to_ckpt.pt
```

The inference command passes the `--ckpt` parameter to specify the saved training model, such as `results/bonsai/ckpts/ckpt_29999_rank0.pt`, to render and evaluate the input dataset. The scene category of the dataset and the evaluation model must be consistent. The inference and evaluation results are saved in the `result_dir` directory.

**Train all scene datasets:**

`train_all.sh` executes the training and inference of all scenes in the dataset in the order specified in the script.

```bash
bash train_all.sh
```

## Directory Structure Description

```text
├── examples                
│  ├── datasets             # Dataset download and parsing module
|  ├── gsplat               # Utility functions and distributed training module provided by the gsplat framework
|  ├── rasterization        # Main algorithm implementation
|     ├── config.py         # Model training and inference configuration file
|     ├── utils.py          # Rendering and model training related utility functions
|     ├── runner.py         # Training and inference engine
|  ├── README.md            # Project description
|  ├── requirements.txt     # External dependencies
|  ├── train.py             # Training and inference startup script
|  ├── train_all.sh         # Training and inference launch script for all scenes
└── ...
```

## Citation

```text
@article{kerbl3Dgaussians,
    author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal={ACM Transactions on Graphics},
    number={4},
    volume={42},
    month={July},
    year= {2023},
    url={https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

@article{ye2025gsplat,
    title={gsplat: An open-source library for Gaussian splatting},
    author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Angjoo Kanazawa},
    journal={Journal of Machine Learning Research},
    volume={26},
    number={34},
    pages={1--17},
    year={2025}
}
```
