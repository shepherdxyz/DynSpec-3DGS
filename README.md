<h1 align="left">DYNAMIC 3D GAUSSIAN RECONSTRUCTION WITH SPECULAR REFLECTION</h1>

This repository contains the official implementation associated with the paper "DYNAMIC 3D GAUSSIAN RECONSTRUCTION WITH SPECULAR REFLECTION".

<p align="left">
  <a href="https://ieeexplore.ieee.org/document/11084729"><b>[📄 Paper]</b></a>
</p>

<p align="center">
  <img src="[assets/teaser.gif或图片路径]" width="100%" alt="Teaser Image"/>
</p>

## 🛠️ Installation (环境配置)

为了确保代码能够顺利运行，请按照以下步骤配置环境。我们的代码在 `Ubuntu 22.04`、`Python 3.8` 和 `CUDA 11.6` 下测试通过。

```bash
# 1. 克隆仓库
git clone https://github.com/shepherdxyz/DynSpec-3DGS.git --recursive

# 2. 创建虚拟环境
conda create -n [你的环境名] python=3.8
conda activate [你的环境名]

# 3. 安装依赖库
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)
pip install -r requirements.txt

```

## 💾 Datasets

我们使用的数据集为<a href="https://jokeryan.github.io/projects/nerf-ds/"><b>[NERF-DS]</b></a>。并按照如下目录结构组织文件：

├── NERF-DS/
│   ├── as/
│   ├── basin/
│   └── bell/
```
---

## 🚀 Quick Start

### Training
运行以下命令在默认数据集上启动训练：
```bash
python train.py --config configs/default.yaml --dataset_path
```

### Evaluation
下载预训练模型放在 `checkpoints/` 文件夹下，然后运行：
```bash
python eval.py --checkpoint checkpoints/best_model.pth --render_mode all
```

---
