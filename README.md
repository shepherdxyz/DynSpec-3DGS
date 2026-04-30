<h1 align="left">DYNAMIC 3D GAUSSIAN RECONSTRUCTION WITH SPECULAR REFLECTION</h1>

This repository contains the official implementation associated with the paper "DYNAMIC 3D GAUSSIAN RECONSTRUCTION WITH SPECULAR REFLECTION".

<p align="left">
  <a href="https://ieeexplore.ieee.org/document/11084729"><b>[📄 Paper]</b></a>
</p>

<p align="center">
  <img src="[assets/teaser.gif或图片路径]" width="100%" alt="Teaser Image"/>
</p>

## 🛠️ Installation (环境配置)

为了确保代码能够顺利运行，请按照以下步骤配置环境。我们的代码在 `Ubuntu 22.04`、`Python 3.8` 和 `CUDA 11.8` 下测试通过。

```bash
# 1. 克隆仓库
git clone https://github.com/shepherdxyz/DynSpec-3DGS.git --recursive

# 2. 创建虚拟环境
conda create -n [你的环境名] python=3.8
conda activate [你的环境名]

# 3. 安装依赖库
pip install -r requirements.txt

```

## 💾 Datasets

我们使用的数据集为<a href="https://jokeryan.github.io/projects/nerf-ds/">NERF-DS</a>。并按照如下目录结构组织文件：
```
├── NERF-DS/
│   ├── as/
│   ├── basin/
│   └── bell/
---
```

## 🚀 Quick Start

### Training

运行以下命令在指定数据集上启动训练：
```bash
python train.py -s --use_env_scope --env_scope_center --env_scope_radius --start_deform_env
```
**⚙️ Parameters (参数说明):**
*   `-s`: 数据集的路径。
*   `--use_env_scope`: 用于消除背景干扰。
*   `--env_scope_center and --env_scope_radius`: 控制环境光生效范围的球形区域中心及半径。
*   `--start_deform_env`: 动态环境贴图（Dynamic Environment Map）开始优化的迭代步数。

### Evaluation
```bash
python eval.py --model_path output/name_of_the_scene --save_images
```

---
