# 基于昇腾平台的3D Gausssian Spaltting的训推示例

3D Gausssian Spaltting(3DGS)是2023年由法国和德国研究人员提出的一种可微渲染算法，以3D高斯椭球为核心图元，通过可微光栅化完成三维场景的重建与渲染，打破传统建模“速度与保真度不可兼得”的瓶颈。本项目旨在提供3DGS的昇腾适配版本。

## 执行样例

本样例支持昇腾Atlas A2环境的单卡训练和推理。

### CANN环境准备

已安装CANN版本和gauss_splat库

### python外部依赖安装

```
cd examples
pip install -r requirements.txt
```

### 数据集准备

在`examples`路径下执行：

```
python datasets/download_dataset.py
```

数据集会被下载并解压到`examples/data/360_v2`路径下。

### 快速启动

本样例准备了单卡环境下的训练和推理脚本。执行脚本前，参考[Ascend社区](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)中的CANN安装软件教程配置环境变量：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**运行训练脚本：**

```
ASCEND_RT_VISIBLE_DEVICES=1 python train.py
```

`ASCEND_RT_VISIBLE_DEVICES=1`指定npu设备1卡，不指定则默认启动0卡。模型训推的配置文件为`rasterization/config.py`，`--data_dir`指定训练数据集路径，默认为`data/360_v2/garden`，`--result_dir`指定训练结果输出路径，默认为`results/garden`。

```
python train.py --data_dir data/360_v2/bonsai --result_dir results/bonsai
```

**运行推理脚本：**

```
python train.py --data_dir data/360_v2/bonsai --result_dir results/bonsai --ckpt path_to_ckpt.pt
```

推理命令传入`--ckpt`参数指定训练保存的模型，如`results/bonsai/ckpts/ckpt_29999_rank0.pt`，对输入数据集进行渲染与评估。数据集与评估模型的场景类别需要保持一致，推理和评估结果保存在`result_dir`目录下。

**训练所有场景数据集：**

`train_all.sh`按脚本中指定顺序执行数据集中所有场景的训练和推理。

```
bash train_all.sh
```

## 目录结构介绍

```
├── examples                
│  ├── datasets             # 数据集下载及解析模块
|  ├── gsplat               # gsplat框架提供的工具函数和分布式训练模块
|  ├── rasterization        # 算法主要实现
|     ├── config.py         # 模型训练、推理配置文件
|     ├── utils.py          # 渲染和模型训练相关工具函数
|     ├── runner.py         # 训练和推理引擎
|  ├── README.md            # 项目说明
|  ├── requirements.txt     # 外部依赖
|  ├── train.py             # 训练和推理启动脚本
|  ├── train_all.sh         # 所有场景的训推拉起脚本
└── ...
```

## Citation

```
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

