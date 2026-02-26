# TACIT Benchmark v0.1.0 -- 模型评估指南

**版本:** 0.1.0
**作者:** Daniel Nobrega Medeiros
**许可证:** Apache-2.0

---

## 概述

TACIT Benchmark 提供双赛道评估体系,支持对生成式模型和判别式模型的视觉推理能力进行标准化测量。本文档详细说明两条评估赛道的输入输出规范、目录结构、结果格式以及完整的端到端评估流程。

**核心设计原则:**
- 所有验证均为确定性、程序化的,采用基于CV的视觉解析器分析 PNG 图像
- 每道谜题均附带近似错误的干扰项(distractor),用于判别式评估
- 难度参数化,支持按任务、按难度级别的精细分析

---

## 赛道 1 -- 生成式评估(Generative Evaluation)

### 任务描述

模型接收谜题图像(PNG 格式)作为输入,需自主生成解答图像(PNG 格式)。评估系统通过基于CV的视觉解析器分析候选解答 PNG,再调用对应任务生成器的 `verify()` 方法进行确定性验证。

**评估流水线:**

```
谜题图像 (PNG) --> 模型 --> 候选解答 PNG --> 基于CV的视觉解析器 --> generator.verify() --> 通过/未通过
```

**适用模型类型:**
- 图像到图像生成模型
- 强多模态生成模型
- 任何能输出 PNG 图像的系统

### 模型输出目录结构

模型需按以下目录结构组织输出文件。文件命名规则为 `{puzzle_id}.png`,其中 `puzzle_id` 的格式为 `{task}_{difficulty}_{seed:04d}`。

```
model_output/
├── maze/
│   ├── easy/
│   │   ├── maze_easy_0042.png
│   │   ├── maze_easy_0043.png
│   │   └── ...
│   ├── medium/
│   │   └── ...
│   └── hard/
│       └── ...
├── raven/
│   ├── easy/
│   │   ├── raven_easy_0042.png
│   │   └── ...
│   ├── medium/
│   │   └── ...
│   └── hard/
│       └── ...
├── ca_forward/
│   └── ...
├── ca_inverse/
│   └── ...
├── logic_grid/
│   └── ...
├── graph_coloring/
│   └── ...
├── graph_isomorphism/
│   └── ...
├── unknot/
│   └── ...
├── ortho_projection/
│   └── ...
└── iso_reconstruction/
    └── ...
```

### 验证机制

每个任务生成器实现独立的 `verify(puzzle, candidate_png) -> VerificationResult` 方法。验证器使用计算机视觉技术分析候选 PNG 图像。验证逻辑因任务而异:

| 任务 | 验证方式 |
|------|---------|
| maze | 检测蓝色路径像素(#2266FF),通过BFS追踪连通分量,验证起点到终点的连通性 |
| raven | 与参考标准瓦片 PNG 进行 SSIM（结构相似性）比较（阈值 0.997） |
| ca_forward | 在网格单元中心进行像素采样,逐单元格比对模拟结果 |
| ca_inverse | 规则表单元格像素采样,逐条目精确匹配 |
| logic_grid | 单元中心的符号颜色采样,验证拉丁方合法性 + 所有约束满足 |
| graph_coloring | 在节点中心位置进行像素采样,检查相邻节点无同色 + 恰好使用 k 种颜色 |
| graph_isomorphism | 绿色 vs 红色像素计数,判断 isomorphic / not isomorphic |
| unknot | 绿色 vs 红色像素计数,判断 unknot / knot |
| ortho_projection | 填充/空白单元格像素采样,与参考标准 PNG 比对 |
| iso_reconstruction | 与参考标准 PNG 进行 SSIM 比较（阈值 0.99999） |

### 结果输出格式(赛道 1)

```json
{
  "track": "generative",
  "model": "model_name",
  "version": "0.1.0",
  "results": [
    {
      "puzzle_id": "maze_easy_0042",
      "task": "maze",
      "difficulty": "easy",
      "passed": true,
      "reason": ""
    },
    {
      "puzzle_id": "raven_hard_0100",
      "task": "raven",
      "difficulty": "hard",
      "passed": false,
      "reason": "Tile attributes do not match: ['shape', 'rotation']"
    }
  ],
  "summary": {
    "overall_accuracy": 0.72,
    "accuracy_by_task": {
      "maze": 0.85,
      "raven": 0.60
    },
    "accuracy_by_difficulty": {
      "easy": 0.90,
      "medium": 0.70,
      "hard": 0.55
    }
  }
}
```

---

## 赛道 2 -- 判别式评估(Discriminative Evaluation)

### 任务描述

模型接收谜题图像和 N 个候选解答图像(1 个正确 + N-1 个干扰项),需从中选出正确解答的索引。评估系统通过精确索引匹配判定正误。

**评估流水线:**

```
谜题图像 + N 个候选图像 --> 模型 --> 选择索引 --> 与正确索引比对 --> 正确/错误
```

**适用模型类型:**
- 具备视觉能力的大语言模型(VLM)
- 任何能处理多图像输入的模型
- 视觉问答系统

### 候选方案构成

对于每道谜题,候选方案集包含:
- **1 个正确解答**: 由生成器产生并经 `verify()` 验证
- **N-1 个干扰项**: 近似正确但恰好违反一项结构约束的错误解答

候选方案的呈现顺序经过随机打乱,正确解答的位置(索引)记录在元数据中。

### 模型输出格式(赛道 2)

模型需提交 JSON 格式的结果文件,包含每道谜题的选择结果:

```json
{
  "track": "discriminative",
  "model": "model_name",
  "version": "0.1.0",
  "results": [
    {
      "puzzle_id": "maze_easy_0042",
      "task": "maze",
      "difficulty": "easy",
      "correct_index": 2,
      "selected_index": 2
    },
    {
      "puzzle_id": "raven_hard_0100",
      "task": "raven",
      "difficulty": "hard",
      "correct_index": 0,
      "selected_index": 3
    },
    {
      "puzzle_id": "graph_coloring_medium_0055",
      "task": "graph_coloring",
      "difficulty": "medium",
      "correct_index": 1,
      "selected_index": 1
    }
  ]
}
```

### 结果汇总格式

```json
{
  "track": "discriminative",
  "model": "model_name",
  "summary": {
    "overall_accuracy": 0.68,
    "accuracy_by_task": {
      "maze": 0.80,
      "raven": 0.55,
      "graph_coloring": 0.70
    },
    "accuracy_by_difficulty": {
      "easy": 0.85,
      "medium": 0.65,
      "hard": 0.45
    },
    "total_puzzles": 500,
    "correct_count": 340
  }
}
```

---

## 干扰项系统(Distractor System)

### 设计理念

干扰项是经过精心构造的"近似正确解答",它们在视觉上高度逼真,但恰好违反了一项结构约束。这一设计确保赛道 2 不会退化为简单的模式匹配任务,而是真正考验模型的结构推理能力。

### 干扰项特性

- 每个干扰项恰好违反**一项**结构约束
- 违反类型被记录在元数据中,便于后续分析
- 干扰项难度是一级参数:
  - **简单干扰项**: 违反较明显,容易辨别
  - **困难干扰项**: 违反极细微,仅涉及单一约束的微小偏差

### 各任务干扰项违反类型

#### 任务 1: 多层迷宫(maze)

| 违反类型 | 说明 |
|---------|------|
| `wall_breach` | 路径穿越墙壁单元格 |
| `portal_skip` | 跨层移动时未经过传送门 |
| `disconnected` | 路径中存在不连续的跳跃(相邻步骤的单元格不相邻) |
| `wrong_exit` | 路径未到达正确的出口位置 |

#### 任务 2: 渐进矩阵(raven)

| 违反类型 | 说明 |
|---------|------|
| `wrong_shape` | 形状属性不正确 |
| `wrong_color` | 颜色属性不正确 |
| `wrong_rotation` | 旋转角度不正确 |
| `wrong_count` | 图形数量不正确 |

#### 任务 3: 元胞自动机正向预测(ca_forward)

| 违反类型 | 说明 |
|---------|------|
| `wrong_cell` | 随机翻转若干单元格的状态 |
| `wrong_step_count` | 模拟了错误的步数(多于或少于 k 步) |
| `wrong_rule` | 使用了不同的转移规则 |

#### 任务 4: 元胞自动机逆向推断(ca_inverse)

| 违反类型 | 说明 |
|---------|------|
| `off_by_one_rule` | 规则表中有一个条目被修改 |
| `transposed_rule` | 规则表中两个不同值的条目被互换 |
| `partial_rule` | 规则表中多个条目被修改(局部正确的规则) |

#### 任务 5: 视觉逻辑网格(logic_grid)

| 违反类型 | 说明 |
|---------|------|
| `constraint_violation` | 违反恰好一条约束 |
| `symbol_swap` | 同一行内两个符号互换(保持行合法但破坏列) |
| `non_unique` | 生成一个不同的拉丁方(满足拉丁方性质但违反约束) |

#### 任务 6: 平面图 k-着色(graph_coloring)

| 违反类型 | 说明 |
|---------|------|
| `adjacent_conflict` | 存在相邻节点使用相同颜色 |
| `missing_color` | 使用的颜色数少于 k(仅用 k-1 种) |
| `wrong_k` | 使用的颜色数多于 k(用了 k+1 种) |

#### 任务 7: 图同构检测(graph_isomorphism)

| 违反类型 | 说明 |
|---------|------|
| `opposite_answer` | 给出相反的判断(二元任务) |

#### 任务 8: 非结检测(unknot)

| 违反类型 | 说明 |
|---------|------|
| `opposite_answer` | 给出相反的判断(二元任务) |

#### 任务 9: 正交投影识别(ortho_projection)

| 违反类型 | 说明 |
|---------|------|
| `wrong_axis` | 沿错误的投影轴进行投影 |
| `missing_feature` | 投影中缺少部分几何特征 |
| `extra_feature` | 投影中多出虚假的几何特征 |
| `mirrored` | 投影被左右镜像翻转 |

#### 任务 10: 等轴测重建(iso_reconstruction)

| 违反类型 | 说明 |
|---------|------|
| `wrong_depth` | 修改深度信息(保持部分投影正确) |
| `missing_face` | 移除部分体素导致缺失面 |
| `extra_volume` | 添加额外体素产生虚假体积 |
| `rotated` | 对三维体绕某轴旋转 90 度 |

---

## 跨赛道分析信号

在同一模型上同时运行赛道 1 和赛道 2,其性能差异本身即构成重要的研究信号:

- **赛道 1 >> 赛道 2**: 模型具有较强的生成能力,但判别(选择)能力较弱
- **赛道 2 >> 赛道 1**: 模型能"识别"正确答案,但缺乏独立生成的能力
- **两赛道均高**: 模型在视觉推理方面表现全面
- **两赛道均低**: 模型在该任务领域的视觉推理能力有限

这一跨赛道差距(cross-track gap)测量的是模型生成式推理与判别式推理之间的能力落差,本身即为一项研究贡献。

---

## 完整示例(End-to-End Walkthrough)

以下以多层迷宫任务为例,展示完整的评估流程。

### 步骤 1: 生成谜题

```bash
tacit generate --task maze --difficulty easy --count 10 --seed 42 --output-dir data
```

生成的数据包含:
- 谜题 PNG: 不含解答路径的迷宫图像
- 解答 PNG: 包含正确路径的迷宫图像
- 干扰项 PNG: 包含错误路径的迷宫图像
- 元数据: puzzle_id, seed, difficulty 参数, 干扰项违反类型

### 步骤 2: 模型推理(赛道 1 -- 生成式)

模型接收谜题图像,生成解答 PNG 文件,并按规定的目录结构放置:

```
model_output/
└── maze/
    └── easy/
        ├── maze_easy_0042.png
        ├── maze_easy_0043.png
        └── ...
```

模型生成的 PNG 图像需要在视觉上呈现解答。对于迷宫任务,解答 PNG 应显示以蓝色（#2266FF）绘制的路径叠加在迷宫网格上。

### 步骤 3: 运行评估(赛道 1)

```bash
tacit evaluate --track generative --model-output ./model_output --tasks maze --output results_track1.json
```

评估系统执行以下操作:
1. 从模型输出目录读取候选 PNG
2. 基于CV的视觉解析器分析 PNG 图像,检测蓝色路径像素并通过BFS追踪连通分量
3. 调用 `MazeGenerator.verify()` 进行验证:
   - 检查路径起点是否为迷宫入口
   - 检查路径终点是否为迷宫出口
   - 检查每个单元格是否为通道(非墙壁)
   - 检查连续单元格是否相邻或通过传送门相连
4. 输出 `VerificationResult(passed=True/False, reason="...")`

### 步骤 4: 模型推理(赛道 2 -- 判别式)

模型接收:
- 谜题图像(不含解答)
- N 个候选方案图像(1 个正确 + N-1 个干扰项,顺序已打乱)

模型输出选择的索引:

```json
{
  "results": [
    {
      "puzzle_id": "maze_easy_0042",
      "task": "maze",
      "difficulty": "easy",
      "correct_index": 2,
      "selected_index": 2
    }
  ]
}
```

### 步骤 5: 运行评估(赛道 2)

```bash
tacit evaluate --track discriminative --model-output ./results_track2.json --tasks maze --output results_track2_eval.json
```

评估系统对每道谜题检查 `selected_index == correct_index`,计算准确率并按任务和难度分组汇总。

### 步骤 6: 分析结果

输出的评估报告包含:
- 整体准确率
- 按任务分组的准确率
- 按难度分组的准确率
- 各干扰项违反类型的混淆分析(判断模型在哪类错误上最容易混淆)

---

## CLI 命令参考

### 生成谜题

```bash
# 生成单一任务
tacit generate --task maze --difficulty hard --count 200 --seed 42

# 从配置文件批量生成
tacit generate --config configs/default.yaml

# 自定义输出目录和干扰项数量
tacit generate --task raven --difficulty medium --count 50 --seed 0 \
    --output-dir ./my_data --distractors 6
```

### 运行评估

```bash
# 赛道 1 评估
tacit evaluate --track generative --model-output ./results/ --tasks all

# 赛道 2 评估
tacit evaluate --track discriminative --model-output ./results/ --tasks all

# 仅评估特定任务
tacit evaluate --track generative --model-output ./results/ \
    --tasks maze,raven,graph_coloring

# 自定义输出路径
tacit evaluate --track generative --model-output ./results/ \
    --tasks all --output my_report.json
```

### 发布至 HuggingFace

```bash
tacit publish --config configs/release.yaml \
    --hf-repo tylerxdurden/TACIT-benchmark
```

---

## 引用格式

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
