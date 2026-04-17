# FoundationPose

## 1. 环境配置

### 使用 Conda 环境（推荐）

本项目使用 Conda 管理环境，已提供环境配置文件。

#### 创建环境：
conda env create -f environment.yml

#### 激活环境：
conda activate robo_env


## 2. 模型权重下载

由于文件大小限制，本仓库不包含预训练模型权重文件。

### 下载地址：
Google Drive 下载链接：https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing

### 需要下载的权重文件：

| 模块 | 版本/文件夹名 |
| --- | --- |
| Refiner（姿态精炼器） | 2023-10-28-18-33-37 |
| Scorer（评分器） | 2024-01-11-20-02-45 |

### 安装步骤：

1. 下载上述两个文件夹
2. 在项目根目录创建 weights/ 文件夹
3. 将下载的文件夹放入 weights/ 目录

**最终目录结构：**

FoundationPose/
├── weights/
│   ├── 2023-10-28-18-33-37/   # Refiner 权重
│   └── 2024-01-11-20-02-45/   # Scorer 权重
├── environment.yml
├── requirements.txt
└── ... (其他代码文件)

