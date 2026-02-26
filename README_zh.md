# TACIT Benchmark v0.1.0

**面向生成式与判别式模型的程序化视觉推理基准测试**

[English](README.md) | **中文**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace-TACIT--benchmark-yellow.svg)](https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark)

---

## 概述

TACIT Benchmark 是一个面向视觉推理能力评估的程序化基准测试。其核心特性如下:

- **语言无关**: 所有任务仅使用符号与视觉元素, 不依赖自然语言线索
- **确定性验证**: 每道题目均可通过程序化方式进行精确验证, 无需人工判断
- **参数化难度**: 各任务提供独立的难度维度, 支持细粒度的能力评估
- **双轨评估**: 同时支持生成式模型与判别式模型的评估

本基准涵盖 6 个推理领域的 10 项任务, 旨在测试模型在不借助语言提示的前提下进行隐式视觉推理的能力。

---

## 任务总览

| 编号 | 任务名称 | 领域 | 描述 | 难度维度 |
|------|---------|------|------|---------|
| 1 | 多层迷宫 | 空间/路径规划 | 包含多层结构与传送门的二维迷宫求解 | 网格尺寸、层数、传送门数量 |
| 2 | 瑞文推理矩阵 | 模式/序列 | 补全 3x3 视觉矩阵中缺失的图块 | 变换规则数量、规则复杂度 |
| 3 | 元胞自动机前向预测 | 模式/序列 | 根据初始状态与规则预测未来状态 | 网格尺寸、规则复杂度、步数 |
| 4 | 元胞自动机逆向推断 | 模式/序列 | 根据前后两个状态推断转换规则 | 规则空间大小、步数、歧义度 |
| 5 | 视觉逻辑网格 | 逻辑约束 | 使用纯符号线索完成约束满足网格 | 网格维度、约束数量、约束类型 |
| 6 | 平面图 k-着色 | 图/连通性 | 为平面图分配 k 种颜色使相邻节点不同色 | 节点数、边密度、k 值 |
| 7 | 图同构检测 | 图/连通性 | 判断两个不同布局的图是否同构 | 节点数、结构相似度、布局扭曲度 |
| 8 | 非结检测 | 拓扑 | 判断二维结图投影是否为平凡结 | 交叉数、Reidemeister 复杂度 |
| 9 | 正交投影识别 | 几何/投影 | 根据三维等轴测视图选择正确的二维正交投影 | 面数、凹面复杂度 |
| 10 | 等轴测重建 | 几何/投影 | 根据三组正交投影重建三维等轴测视图 | 面数、投影歧义度 |

**设计要点:**
- 任务 3 与 4 构成正向/逆向对, 在同一领域测试定性不同的推理能力
- 任务 9 与 10 同样构成正向/逆向对
- 任务 7 与 8 为二元判断任务, 生成轨采用视觉指示, 判别轨采用候选对比较

---

## 双轨评估体系

### 轨道 1 -- 生成式评估

- **输入**: 题目图像
- **输出**: 模型生成 PNG 格式的解答图像
- **评估方式**: 基于计算机视觉（OpenCV + scikit-image）的视觉解析与程序化验证
- **适用模型**: 图像生成模型、多模态生成模型
- **流程**: 模型 PNG 输出 → 基于CV的视觉解析器 → `generator.verify()` → 通过/失败

> **说明:** SVG 仍作为题目生成的源格式, 但轨道 1 的验证基于 PNG 图像, 采用计算机视觉技术（像素采样、颜色计数、SSIM 结构相似性、BFS 路径检测）而非 SVG 元数据解析。

### 轨道 2 -- 判别式评估

- **输入**: 题目图像 + N 个候选解答图像
- **输出**: 模型选择正确的候选索引
- **评估方式**: 精确匹配正确索引
- **适用模型**: 具备视觉能力的大语言模型及其他图像理解模型

### 干扰项机制

每个生成器均会产出题目、正确解答与 N 个干扰项。干扰项为"近似正确"的解答 -- 结构上合理, 但恰好违反一项约束条件。干扰项难度本身也是一个可调参数: 简单干扰项包含明显违规, 困难干扰项仅包含细微的单约束违反。

### 跨轨信号

同一模型在轨道 1 与轨道 2 上的性能差距, 反映了其生成式推理与判别式推理之间的能力差异 -- 这本身即为一项有价值的研究发现。

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/danielxmed/tacit_benchmark_0.1.0.git
cd tacit_benchmark_0.1.0

# 以开发模式安装
pip install -e .

# 安装全部可选依赖
pip install -e ".[all]"
```

### 生成题目

```bash
# 为单个任务生成题目
tacit generate --task maze --difficulty easy --count 10 --seed 42

# 使用配置文件生成完整基准测试集
tacit generate --config configs/default.yaml
```

### 评估模型

```bash
# 轨道 1: 生成式评估
tacit evaluate --track generative --model-output ./results/ --tasks all

# 轨道 2: 判别式评估
tacit evaluate --track discriminative --model-output ./results/ --tasks all
```

### 发布至 HuggingFace

```bash
# 生成并发布冻结快照至 HuggingFace
tacit publish --config configs/full_release.yaml --hf-repo tylerxdurden/TACIT-benchmark
```

---

## 项目结构

```
tacit_benchmark_0.1.0/
├── README.md / README_zh.md        # 英文/中文说明文档
├── LICENSE                          # Apache-2.0 许可证
├── pyproject.toml                   # 项目配置与依赖管理
├── tacit/
│   ├── core/
│   │   ├── renderer.py              # SVG/PNG 渲染抽象层
│   │   ├── verifier.py              # 验证接口基类
│   │   ├── distractor.py            # 干扰项生成框架
│   │   ├── types.py                 # Puzzle, Solution, DifficultyParams 等类型定义
│   │   └── parsers/                 # 各任务的结构化图像解析器
│   ├── generators/                  # 每个任务对应一个生成器模块
│   │   ├── maze.py                  # 任务 1: 多层迷宫
│   │   ├── raven.py                 # 任务 2: 瑞文推理矩阵
│   │   ├── ca_forward.py            # 任务 3: 元胞自动机前向预测
│   │   ├── ca_inverse.py            # 任务 4: 元胞自动机逆向推断
│   │   ├── logic_grid.py            # 任务 5: 视觉逻辑网格
│   │   ├── graph_coloring.py        # 任务 6: 平面图 k-着色
│   │   ├── graph_isomorphism.py     # 任务 7: 图同构检测
│   │   ├── unknot.py                # 任务 8: 非结检测
│   │   ├── ortho_projection.py      # 任务 9: 正交投影识别
│   │   └── iso_reconstruction.py    # 任务 10: 等轴测重建
│   └── evaluation/
│       ├── harness.py               # 评估核心调度器 (任务无关)
│       ├── track1.py                # 生成轨: CV视觉解析 -> 验证
│       ├── track2.py                # 判别轨: 索引匹配
│       ├── metrics.py               # 评分与聚合
│       └── report.py                # 评估报告生成
├── configs/
│   ├── default.yaml                 # 开发/测试配置
│   └── full_release.yaml            # 完整基准测试配置
├── tests/                           # 测试套件
└── data/                            # 生成数据 (已加入 .gitignore)
```

---

## 技术栈

| 类别 | 依赖 |
|------|------|
| 核心 | Python 3.11+, NumPy, SciPy, Pillow, svgwrite, PyYAML |
| 计算机视觉 | OpenCV (cv2), scikit-image (SSIM、像素分析) |
| 图论 | NetworkX, pynauty (规范形式计算) |
| 拓扑 | pyknotid (结不变量计算) |
| 三维几何 | trimesh, numpy-stl |
| 渲染 | CairoSVG (SVG 转 PNG 光栅化) |
| 分发 | huggingface_hub, datasets |

---

## 引用

如果您在研究中使用了 TACIT Benchmark, 请引用以下文献:

```bibtex
@misc{medeiros_2026,
    author       = {Daniel Nobrega Medeiros},
    title        = {TACIT-benchmark},
    year         = 2026,
    url          = {https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark},
    doi          = {10.57967/hf/7904},
    publisher    = {Hugging Face}
}
```

**HuggingFace 数据集:** [tylerxdurden/TACIT-benchmark](https://huggingface.co/datasets/tylerxdurden/TACIT-benchmark)

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

---

## 作者

**Daniel Nobrega Medeiros** -- 机器学习研究者、医师

- GitHub: [danielxmed](https://github.com/danielxmed)
- Google Scholar: [scholar.google.com.br/citations?user=D_6AZoEAAAAJ](https://scholar.google.com.br/citations?user=D_6AZoEAAAAJ&hl=pt-BR)
- arXiv: [arxiv.org/abs/2602.07061](https://arxiv.org/abs/2602.07061)

研究方向: 机械可解释性、隐式计算与非语言推理、扩散Transformer中的视觉推理。
