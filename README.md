# Redundant Media Detector / 媒体冗余检测器

## Table of Contents / 目录
- [English](#english)
  - [Overview](#overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
- [中文](#中文)
  - [项目简介](#项目简介)
  - [功能特点](#功能特点)
  - [运行环境](#运行环境)
  - [安装说明](#安装说明)
  - [使用方法](#使用方法)
  - [参与贡献](#参与贡献)
  - [许可协议](#许可协议)

---

## English

### Overview
Redundant Media Detector is a lightweight toolkit for discovering duplicate or near-duplicate items inside large collections of images and videos. The scripts combine frame extraction, perceptual hashing, and similarity scoring to highlight content that can be safely deduplicated.

### Features
- **OpticalImageCompare.py** – Compares still images using feature matching and perceptual hashing to flag visually similar files.
- **VideoChecker.py** – Scans video files for repeated frame sequences and generates similarity statistics.
- **VideoDetector.py** – Orchestrates a full detection pipeline that breaks videos into frames and runs similarity comparison utilities.
- Command-line switches for adjusting similarity thresholds and output directories.
- Modular functions that can be imported into other Python projects.

### Requirements
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- (Optional) `tqdm` for progress bars when scanning large datasets

### Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If a `requirements.txt` file is not present, install the dependencies manually:
```bash
pip install opencv-python numpy tqdm
```

### Usage
Prepare a workspace that contains the media files you want to analyze. All scripts support the `-h` / `--help` option for additional parameters.

#### Image comparison
```bash
python OpticalImageCompare.py --input /path/to/images --threshold 0.8
```
- `--input`: Directory containing the reference images.
- `--threshold`: Floating point value between `0` and `1` that controls how strict the similarity comparison is.

#### Video redundancy scan
```bash
python VideoChecker.py --input /path/to/videos --window 30
```
- `--input`: Directory containing one or more video files.
- `--window`: Number of frames considered for rolling similarity calculations.

#### Integrated detection pipeline
```bash
python VideoDetector.py --video /path/to/video.mp4 --output ./reports
```
- `--video`: Path to the video file that will be analyzed.
- `--output`: Directory where similarity reports and extracted frames are stored.

### Contributing
1. Fork the repository and create a feature branch.
2. Run code formatters or linters as appropriate for your changes.
3. Submit a pull request describing the problem solved and relevant testing notes.

### License
This project is distributed under the MIT License (unless otherwise noted). Refer to the repository's `LICENSE` file for full terms.

---

## 中文

### 项目简介
媒体冗余检测器是一套轻量级工具，用于在大规模图像和视频集合中发现重复或近似重复的内容。脚本结合帧提取、感知哈希和相似度计算，帮助快速识别可安全去重的媒体文件。

### 功能特点
- **OpticalImageCompare.py**：利用特征匹配与感知哈希对静态图像进行比较，标记视觉上相似的文件。
- **VideoChecker.py**：扫描视频文件的重复帧序列，并生成相似度统计数据。
- **VideoDetector.py**：整合完整检测流程，包含视频帧提取与相似度比对。
- 支持命令行参数，可调整相似度阈值和输出目录。
- 模块化函数，可在其他 Python 项目中复用。

### 运行环境
- Python 3.8 及以上版本
- OpenCV（`opencv-python`）
- NumPy
- （可选）`tqdm`，用于在扫描大型数据集时显示进度条

### 安装说明
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 用户执行 .venv\Scripts\activate
pip install -r requirements.txt
```
若仓库中未提供 `requirements.txt`，请手动安装依赖：
```bash
pip install opencv-python numpy tqdm
```

### 使用方法
在分析前，请准备好包含待检测媒体文件的工作目录。所有脚本均支持 `-h` / `--help` 参数以获取详细说明。

#### 图像相似度比较
```bash
python OpticalImageCompare.py --input /path/to/images --threshold 0.8
```
- `--input`：包含参考图像的目录。
- `--threshold`：介于 `0` 和 `1` 的浮点值，用于控制相似度判断的严格程度。

#### 视频冗余扫描
```bash
python VideoChecker.py --input /path/to/videos --window 30
```
- `--input`：包含一个或多个视频文件的目录。
- `--window`：用于滚动相似度计算的帧数窗口大小。

#### 综合检测流程
```bash
python VideoDetector.py --video /path/to/video.mp4 --output ./reports
```
- `--video`：待分析的视频文件路径。
- `--output`：存放相似度报告和提取帧的目录。

### 参与贡献
1. Fork 本仓库，并创建新的功能分支。
2. 根据需要运行代码格式化工具或静态检查工具。
3. 提交 Pull Request，说明所解决的问题和相关测试结果。

### 许可协议
本项目默认采用 MIT 许可协议（除非另有说明）。详细条款请查阅仓库中的 `LICENSE` 文件。

