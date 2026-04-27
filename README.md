<h1 align="left">DYNAMIC 3D GAUSSIAN RECONSTRUCTION WITH SPECULAR REFLECTION</h1>

This repository contains the official implementation associated with the paper "DYNAMIC 3D GAUSSIAN RECONSTRUCTION WITH SPECULAR REFLECTION".

<p align="left">
  <a href="https://ieeexplore.ieee.org/document/11084729"><b>[📄 Paper]</b></a>
</p>

<p align="center">
  <img src="[assets/teaser.gif或图片路径]" width="100%" alt="Teaser Image"/>
</p>

> 本仓库是 ICIP 2025 论文 *"[你的论文全名]"* 的官方实现代码。在这里用一到两句话精炼总结你的论文做出了什么核心贡献（例如：我们提出了一种基于XXX的新方法，在XXX任务上实现了Y%的性能提升，同时保持了实时渲染速度）。

## 🛠️ Installation (环境配置)

为了确保代码能够顺利运行，请按照以下步骤配置环境。我们的代码在 `Ubuntu 20.04`、`Python 3.8` 和 `CUDA 11.6` 下测试通过（*请替换为你的实际环境*）。

```bash
# 1. 克隆仓库
git clone [https://github.com/](https://github.com/)[你的用户名]/[你的仓库名].git --recursive
cd [你的仓库名]

# 2. 创建虚拟环境
conda create -n [你的环境名] python=3.8
conda activate [你的环境名]

# 3. 安装依赖库
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)
pip install -r requirements.txt
