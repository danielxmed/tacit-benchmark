# TACIT Benchmark v0.1.0 -- 技术架构

**版本:** 0.1.0
**作者:** Daniel Nobrega Medeiros

---

## 架构概述

TACIT Benchmark 采用分层架构:薄核心层(类型、渲染器、验证器、干扰项框架)为 10 个独立的任务生成器提供统一的协议和共享基础设施,上层为任务无关的评估工具链(双赛道评估、指标聚合、报告生成)、CLI 入口以及 HuggingFace 发布管线。

```
┌───────────────────────────────────────────────────────┐
│                     CLI (cli.py)                      │
├───────────────────────────────────────────────────────┤
│              评估工具链 (evaluation/)                  │
│   harness.py | track1.py | track2.py | metrics.py    │
├───────────────────────────────────────────────────────┤
│              10 个任务生成器 (generators/)              │
│   maze | raven | ca_forward | ca_inverse | logic_grid │
│   graph_coloring | graph_isomorphism | unknot         │
│   ortho_projection | iso_reconstruction               │
├───────────────────────────────────────────────────────┤
│                核心层 (core/)                          │
│   types.py | renderer.py | verifier.py | distractor.py│
│   parsers/ (base + 任务专用解析器)                     │
└───────────────────────────────────────────────────────┘
```

**设计原则:**
- 每个生成器独立拥有其谜题逻辑、验证方法和干扰项生成
- 核心层仅提供最小公共抽象,不过度封装
- SVG 优先生成,多分辨率 PNG 用于分发
- 配置驱动的规模扩展,无需修改代码

---

## 核心类型系统

### 数据类型 (`tacit/core/types.py`)

系统定义了以下核心数据类型:

#### DifficultyParams

```python
@dataclass(frozen=True)
class DifficultyParams:
    level: str                    # 难度级别标识 ("easy", "medium", "hard")
    params: dict[str, Any]        # 任务特定的难度参数字典
```

冻结的不可变数据类,确保难度参数在传递过程中不被意外修改。`params` 字典的键和值因任务而异(参见各任务规范)。

#### DifficultyRange

```python
@dataclass(frozen=True)
class DifficultyRange:
    name: str                     # 参数名称
    min_val: float                # 最小值
    max_val: float                # 最大值
    step: float | None = None     # 步长(可选)
    description: str = ""         # 人类可读描述
```

声明性地描述一个难度轴的取值范围,供外部工具(如配置生成器、文档系统)查询。

#### VerificationResult

```python
@dataclass(frozen=True)
class VerificationResult:
    passed: bool                  # 是否通过验证
    reason: str = ""              # 人类可读的验证结果说明
    details: dict[str, Any] = field(default_factory=dict)  # 详细信息
```

验证结果的统一表示。`details` 字段可包含任务特定的诊断信息(如差异单元格数量、冲突边列表等)。

#### PuzzleInstance

```python
@dataclass
class PuzzleInstance:
    task: str                             # 任务标识
    puzzle_id: str                        # 唯一实例标识 (格式: {task}_{level}_{seed:04d})
    seed: int                             # 随机种子(保证可重现)
    difficulty: DifficultyParams          # 难度参数
    puzzle_svg: str                       # 谜题 SVG 字符串
    solution_svg: str                     # 解答 SVG 字符串
    distractor_svgs: list[str]            # 干扰项 SVG 字符串列表
    distractor_violations: list[str]      # 干扰项违反类型列表
    metadata: dict[str, Any]              # 任务特定元数据
```

一个完整谜题实例的全部数据。`__post_init__` 强制 `distractor_svgs` 与 `distractor_violations` 长度一致。

#### GeneratorProtocol

```python
@runtime_checkable
class GeneratorProtocol(Protocol):
    def generate(self, difficulty: DifficultyParams, seed: int) -> PuzzleInstance: ...
    def verify(self, puzzle: PuzzleInstance, candidate_png: bytes) -> VerificationResult: ...
    def difficulty_axes(self) -> list[DifficultyRange]: ...
```

运行时可检查的协议,定义所有生成器必须实现的接口。

---

## 生成器协议

### 抽象基类 (`tacit/generators/base.py`)

所有 10 个任务生成器继承自 `BaseGenerator`:

```python
class BaseGenerator(ABC):
    def __init__(self, task_name: str) -> None
    def generate(self, difficulty, seed, num_distractors=4) -> PuzzleInstance

    # 子类必须实现
    @abstractmethod def _generate_puzzle(self, difficulty, rng) -> tuple[Any, Any]
    @abstractmethod def _generate_puzzle_svg(self, puzzle_data) -> str
    @abstractmethod def _generate_solution_svg(self, puzzle_data, solution_data) -> str
    @abstractmethod def _generate_distractor(self, puzzle_data, solution_data, violation_type, rng) -> tuple[str, str]
    @abstractmethod def _available_violations(self) -> list[str]
    @abstractmethod def verify(self, puzzle, candidate_png) -> VerificationResult
    @abstractmethod def difficulty_axes(self) -> list[DifficultyRange]
```

### 生成流程

`generate()` 方法编排完整的谜题生成流程:

```
1. 初始化谜题 RNG (seed)
2. 调用 _generate_puzzle() 获取 (puzzle_data, solution_data)
3. 调用 _generate_puzzle_svg(puzzle_data) 渲染谜题
4. 调用 _generate_solution_svg(puzzle_data, solution_data) 渲染解答
5. 初始化干扰项 RNG (seed + 2^31)
6. 循环生成 num_distractors 个干扰项:
   - 循环选择违反类型 (round-robin)
   - 调用 _generate_distractor() 生成干扰项 SVG
7. 构造并返回 PuzzleInstance
```

**关键设计决策:** 谜题和干扰项使用**独立的 RNG 流**。谜题 RNG 由 `seed` 初始化,干扰项 RNG 由 `seed + 2^31` 初始化。这保证了无论干扰项数量如何变化,谜题本身的确定性不受影响。

### 生成器注册

| 生成器类 | 任务标识 | 源文件 |
|---------|---------|-------|
| `MazeGenerator` | `maze` | `generators/maze.py` |
| `RavenGenerator` | `raven` | `generators/raven.py` |
| `CAForwardGenerator` | `ca_forward` | `generators/ca_forward.py` |
| `CAInverseGenerator` | `ca_inverse` | `generators/ca_inverse.py` |
| `LogicGridGenerator` | `logic_grid` | `generators/logic_grid.py` |
| `GraphColoringGenerator` | `graph_coloring` | `generators/graph_coloring.py` |
| `GraphIsomorphismGenerator` | `graph_isomorphism` | `generators/graph_isomorphism.py` |
| `UnknotGenerator` | `unknot` | `generators/unknot.py` |
| `OrthoProjectionGenerator` | `ortho_projection` | `generators/ortho_projection.py` |
| `IsoReconstructionGenerator` | `iso_reconstruction` | `generators/iso_reconstruction.py` |

### 共享工具模块

部分生成器共享内部工具模块:

| 模块 | 用途 | 使用者 |
|------|------|-------|
| `_ca_common.py` | CA 网格生成、规则生成、模拟、SVG 渲染/解析 | ca_forward, ca_inverse |
| `_geometry_common.py` | 体素生成、正交投影、等轴测渲染 | ortho_projection, iso_reconstruction |

---

## 渲染层

### 共享渲染模块 (`tacit/core/renderer.py`)

所有生成器通过统一的渲染抽象输出视觉内容,确保 10 项任务之间的视觉一致性。

#### 视觉样式系统

```python
STYLE = {
    "background": "#FFFFFF",
    "line_width": 2,
    "line_color": "#222222",
    "grid_color": "#CCCCCC",
    "highlight_color": "#FF4444",
    "solution_color": "#2266FF",
    "font_family": "monospace",
    "font_size": 14,
    "colors": [
        "#E63946",   # 红色
        "#457B9D",   # 钢蓝
        "#2A9D8F",   # 青色
        "#E9C46A",   # 黄色
        "#F4A261",   # 橙色
        "#264653",   # 深青
        "#6A0572",   # 紫色
        "#1B998B",   # 薄荷
        "#FF6B6B",   # 珊瑚
        "#4ECDC4",   # 绿松石
    ],
}
```

#### 渲染 API

| 函数 | 功能 |
|------|------|
| `create_canvas(width, height)` | 创建带白色背景的 SVG 画布 |
| `draw_rect(canvas, x, y, w, h, ...)` | 绘制矩形 |
| `draw_circle(canvas, cx, cy, r, ...)` | 绘制圆形 |
| `draw_line(canvas, x1, y1, x2, y2, ...)` | 绘制线段 |
| `draw_path(canvas, d, ...)` | 绘制 SVG 路径 |
| `draw_text(canvas, x, y, text, ...)` | 绘制文本 |
| `svg_to_string(canvas)` | 导出 SVG 字符串 |
| `svg_to_png(canvas, width)` | 光栅化为 PNG (单分辨率) |
| `svg_to_png_multi(canvas, widths)` | 光栅化为多分辨率 PNG |
| `save_svg(canvas, path)` | 保存 SVG 文件 |
| `save_png(canvas, path, width)` | 保存 PNG 文件 |

#### 底层库

- **SVG 生成:** svgwrite
- **PNG 光栅化:** cairosvg (高质量 SVG 到 PNG 转换)

### 图像格式策略

| 阶段 | 格式 | 说明 |
|------|------|------|
| 生成 | SVG | 矢量格式,无损,精确 |
| 分发 | PNG (256, 512, 1024) | 多分辨率,兼容性好 |
| 研究 | SVG (可选) | 原始矢量文件,供需要的研究者使用 |

---

## 验证合约

### 验证器基类 (`tacit/core/verifier.py`)

```python
class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, puzzle: PuzzleInstance, candidate_png: bytes) -> VerificationResult: ...
    @abstractmethod
    def extract_structure(self, svg_string: str) -> Any: ...
```

### 验证流水线

赛道 1 的验证流程:

```
候选 PNG --> 基于CV的视觉解析器 --> 语义表示 --> verify() --> VerificationResult
```

每个生成器实现自己的 `verify()` 方法,验证逻辑完全由任务领域决定:

| 验证类别 | 任务 | 方法 |
|---------|------|------|
| BFS路径追踪 | maze | 检测蓝色路径像素,BFS追踪连通分量,验证连通性 |
| SSIM比较 | raven | 与参考标准瓦片 PNG 进行 SSIM 比较（阈值 0.997） |
| 像素采样 | ca_forward | 网格单元中心像素采样,逐单元格状态比较 |
| 像素采样 | ca_inverse | 规则表单元格像素采样,逐条目精确匹配 |
| 颜色采样 | logic_grid | 单元中心符号颜色采样,拉丁方合法性 + 约束检查 |
| 节点采样 | graph_coloring | 节点中心像素采样,无相邻同色 + 精确 k 色 |
| 颜色计数 | graph_isomorphism, unknot | 绿色 vs 红色像素计数 |
| 像素采样/SSIM | ortho_projection, iso_reconstruction | 填充/空白单元格采样或 SSIM 比较 |

### 视觉解析器 (`tacit/core/cv_utils.py`)

基于CV的视觉解析模块,负责从 PNG 图像中提取语义结构:

| 解析方法 | 适用任务 | 功能 |
|---------|---------|------|
| BFS路径追踪 | maze | 检测蓝色路径像素,追踪连通分量 |
| SSIM比较 | raven, iso_reconstruction | 结构相似性比较 |
| 像素采样 | ca_forward, ca_inverse, ortho_projection | 网格/规则表单元格颜色采样 |
| 颜色计数 | graph_isomorphism, unknot | 绿色vs红色像素计数 |
| 节点采样 | graph_coloring | 节点中心位置颜色采样 |
| 颜色采样 | logic_grid | 单元中心符号颜色采样 |

---

## 干扰项系统

### 干扰项框架 (`tacit/core/distractor.py`)

```python
class BaseDistractorGenerator(ABC):
    @abstractmethod
    def generate_distractor(self, puzzle_data, solution_data, violation_type, rng) -> tuple[str, str]: ...
    @abstractmethod
    def available_violations(self) -> list[str]: ...
    def generate_set(self, puzzle_data, solution_data, count, rng) -> tuple[list[str], list[str]]: ...
```

### 设计原则

1. **恰好一项违反:** 每个干扰项精确违反一项结构约束,其余结构保持正确
2. **类型多样性:** 通过轮询(round-robin)在可用违反类型间循环,确保干扰项集合的多样性
3. **可追溯性:** 每个干扰项的违反类型被记录在 `distractor_violations` 列表中
4. **难度分级:**
   - 简单干扰项: 违反较为明显(如穿墙)
   - 困难干扰项: 违反极为细微(如规则表中仅一个条目不同)

### 干扰项在双赛道中的角色

| 赛道 | 干扰项使用方式 |
|------|-------------|
| 赛道 1 (生成式) | **不使用** -- 模型独立生成解答,验证器直接判定 |
| 赛道 2 (判别式) | **核心组件** -- 与正确解答混合,构成多选候选集 |

---

## 评估工具链

### 架构组件 (`tacit/evaluation/`)

#### EvaluationHarness (`harness.py`)

评估协调器,管理生成器注册和任务分发:

```python
class EvaluationHarness:
    def __init__(self) -> None                             # 加载所有已注册生成器
    def available_tasks(self) -> list[str]                 # 列出可用任务
    def get_generator(self, task_name: str) -> BaseGenerator  # 获取生成器实例
    def run_track2(self, task_name, results) -> dict[str, float]  # 运行赛道 2 评估
```

#### Track 1 评估 (`track1.py`)

```python
def evaluate_generative(
    generator: BaseGenerator,
    puzzle: PuzzleInstance,
    candidate_png: bytes,
) -> VerificationResult
```

将候选 PNG 传递给任务特定生成器的 `verify()` 方法。

#### Track 2 评估 (`track2.py`)

```python
@dataclass(frozen=True)
class DiscriminativeResult:
    correct: bool
    selected_index: int
    correct_index: int

def evaluate_discriminative(correct_index: int, selected_index: int) -> DiscriminativeResult
```

精确索引匹配:当且仅当 `selected_index == correct_index` 时判定正确。

#### 指标聚合 (`metrics.py`)

```python
def compute_accuracy(results: list[bool]) -> float
def compute_accuracy_by_difficulty(results: list[dict]) -> dict[str, float]
def compute_accuracy_by_task(results: list[dict]) -> dict[str, float]
```

支持整体、按难度、按任务三个维度的准确率聚合。

#### 报告生成 (`report.py`)

负责将评估结果格式化为结构化的 JSON 报告。

---

## CLI 命令接口

### 入口点 (`tacit/cli.py`)

CLI 使用 Click 框架构建,提供三个主命令:

#### `tacit generate`

```bash
tacit generate [OPTIONS]
```

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--task` | str | None | 任务名称 |
| `--difficulty` | str | "easy" | 难度级别 |
| `--count` | int | 10 | 生成数量 |
| `--seed` | int | 42 | 随机种子 |
| `--config` | Path | None | YAML 配置文件路径 |
| `--output-dir` | Path | "data" | 输出目录 |
| `--distractors` | int | 4 | 每道谜题的干扰项数量 |

#### `tacit evaluate`

```bash
tacit evaluate [OPTIONS]
```

| 选项 | 类型 | 说明 |
|------|------|------|
| `--track` | Choice | "generative" 或 "discriminative"(必选) |
| `--model-output` | Path | 模型输出目录(必选) |
| `--tasks` | str | 逗号分隔的任务名称或 "all" |
| `--output` | Path | 评估报告输出路径 |

#### `tacit publish`

```bash
tacit publish [OPTIONS]
```

| 选项 | 类型 | 说明 |
|------|------|------|
| `--config` | Path | 生成配置文件(必选) |
| `--hf-repo` | str | HuggingFace 仓库(必选) |
| `--version-tag` | str | 版本标签(默认从 pyproject.toml 读取) |

### 已知任务列表

```python
KNOWN_TASKS = [
    "maze", "raven", "ca_forward", "ca_inverse", "logic_grid",
    "graph_coloring", "graph_isomorphism", "unknot",
    "ortho_projection", "iso_reconstruction",
]
```

---

## 配置系统

### 配置文件格式

系统使用 YAML 格式的配置文件驱动批量生成:

```yaml
# configs/default.yaml
version: "0.1.0"
seed: 42
output_dir: "data"
resolutions: [256, 512]
distractors_per_puzzle: 4

tasks:
  maze:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {grid_size: 8, layers: 1, portals: 0}
      medium: {grid_size: 16, layers: 2, portals: 2}
      hard: {grid_size: 32, layers: 3, portals: 5}
  raven:
    enabled: true
    count_per_difficulty: 10
    difficulties:
      easy: {rules: 1, complexity: "additive"}
      medium: {rules: 2, complexity: "additive"}
      hard: {rules: 3, complexity: "compositional"}
  # ... 其他任务配置
```

### 正式发布配置

`configs/full_release.yaml` 用于生成完整的基准测试集:
- `count_per_difficulty: 200`
- `resolutions: [256, 512, 1024]`
- `distractors_per_puzzle: 6`
- 新增 `expert` 难度级别

---

## HuggingFace 发布管线

### 发布流程

```
tacit publish --config configs/full_release.yaml --hf-repo tylerxdurden/TACIT-benchmark
```

执行以下步骤:
1. 读取配置文件
2. 按配置生成所有任务的所有难度级别的谜题实例
3. 将 SVG 光栅化为多分辨率 PNG
4. 计算每个文件的校验和(SHA-256)
5. 生成元数据文件(版本、种子、生成参数、校验和)
6. 推送至 HuggingFace 仓库

### HuggingFace 快照目录结构

```
tylerxdurden/TACIT-benchmark/
├── README.md                      # 数据集卡片(包含引用信息)
├── README_zh.md                   # 中文数据集卡片
├── metadata.json                  # 版本、种子、生成参数、校验和
├── task_01_maze/
│   ├── easy/
│   │   ├── puzzle_XXXX.png
│   │   ├── solution_XXXX.png
│   │   ├── distractors_XXXX/
│   │   │   ├── distractor_0.png
│   │   │   ├── distractor_1.png
│   │   │   └── ...
│   │   └── meta_XXXX.json
│   ├── medium/
│   │   └── ...
│   ├── hard/
│   │   └── ...
│   └── task_info.json             # 难度轴描述、验证说明
├── task_02_raven/
│   └── ...
├── task_03_ca_forward/
│   └── ...
├── task_04_ca_inverse/
│   └── ...
├── task_05_logic_grid/
│   └── ...
├── task_06_graph_coloring/
│   └── ...
├── task_07_graph_isomorphism/
│   └── ...
├── task_08_unknot/
│   └── ...
├── task_09_ortho_projection/
│   └── ...
└── task_10_iso_reconstruction/
    └── ...
```

### 每个谜题实例的元数据格式

```json
{
    "task": "maze",
    "difficulty": {"grid_size": 32, "layers": 3, "portals": 5},
    "seed": 42,
    "puzzle_svg": "path/to/puzzle.svg",
    "puzzle_png": {"256": "...", "512": "...", "1024": "..."},
    "solution_svg": "path/to/solution.svg",
    "solution_png": {"256": "...", "512": "...", "1024": "..."},
    "distractors_png": ["distractor_0.png", "...", "distractor_N.png"],
    "distractor_violations": ["wall_breach", "portal_skip", "..."],
    "verification_hash": "sha256:..."
}
```

---

## 项目目录结构

```
tacit_benchmark_0.1.0/
├── ABOUT_THE_AUTHOR.md
├── LICENSE                        # Apache-2.0
├── README.md / README_zh.md
├── pyproject.toml                 # 包配置与依赖
├── tacit/
│   ├── __init__.py                # __version__ = "0.1.0"
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py               # 核心数据类型
│   │   ├── renderer.py            # SVG/PNG 渲染抽象
│   │   ├── cv_utils.py            # 计算机视觉工具（像素采样、颜色检测、SSIM）
│   │   ├── verifier.py            # 验证器基类
│   │   ├── distractor.py          # 干扰项框架
│   │   └── parsers/               # 结构解析器（兼容层）
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── maze_parser.py
│   │       ├── raven_parser.py
│   │       ├── knot_parser.py
│   │       └── graph_parser.py
│   ├── generators/                # 10 个任务生成器
│   │   ├── __init__.py
│   │   ├── base.py                # 抽象基类
│   │   ├── maze.py
│   │   ├── raven.py
│   │   ├── ca_forward.py
│   │   ├── ca_inverse.py
│   │   ├── logic_grid.py
│   │   ├── graph_coloring.py
│   │   ├── graph_isomorphism.py
│   │   ├── unknot.py
│   │   ├── ortho_projection.py
│   │   ├── iso_reconstruction.py
│   │   ├── _ca_common.py          # CA 共享工具
│   │   └── _geometry_common.py    # 几何共享工具
│   ├── evaluation/                # 评估工具链
│   │   ├── __init__.py
│   │   ├── harness.py             # 评估协调器
│   │   ├── track1.py              # 生成式评估
│   │   ├── track2.py              # 判别式评估
│   │   ├── metrics.py             # 指标聚合
│   │   └── report.py              # 报告生成
│   └── cli.py                     # CLI 入口
├── configs/
│   ├── default.yaml               # 开发/测试配置
│   └── full_release.yaml          # 正式发布配置
├── scripts/
│   └── publish_hf.py              # HuggingFace 发布脚本
├── tests/                         # 测试套件
├── docs/
│   ├── en/                        # 英文文档
│   └── zh/                        # 中文文档
└── data/                          # 生成数据(gitignore)
    ├── svg/
    ├── png/
    └── metadata/
```

---

## 技术栈

### 核心依赖

| 库 | 版本要求 | 用途 |
|----|---------|------|
| Python | >= 3.11 | 运行时 |
| numpy | >= 1.24 | 数值计算、网格操作 |
| scipy | >= 1.11 | Delaunay 三角剖分(图着色) |
| svgwrite | >= 1.4 | SVG 生成 |
| cairosvg | >= 2.7 | SVG 到 PNG 光栅化 |
| Pillow | >= 10.0 | 图像处理 |
| opencv-python | >= 4.8 | 计算机视觉(像素采样、颜色检测) |
| scikit-image | >= 0.21 | SSIM（结构相似性）比较 |
| networkx | >= 3.1 | 图算法(着色、同构、连通性) |
| trimesh | >= 4.0 | 三维几何(体素操作) |
| numpy-stl | >= 3.0 | STL 格式支持 |
| pyyaml | >= 6.0 | YAML 配置解析 |
| click | >= 8.1 | CLI 框架 |

### 可选依赖

| 依赖组 | 库 | 用途 |
|-------|-----|------|
| publish | huggingface_hub >= 0.20, datasets >= 2.16 | HuggingFace 发布 |
| dev | pytest >= 7.4, pytest-cov >= 4.1 | 测试框架 |
| topology | pyknotid >= 0.5 | 纽结不变量计算(任务 8) |
| graph | pynauty >= 1.1 | 图规范形式(任务 7) |

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 语言依赖性 | 最小化(仅符号/变量) | 测试视觉推理而非语言理解 |
| 验证方式 | 基于CV的确定性验证 | 无歧义,完全可重现,从 PNG 图像中提取结构 |
| 难度体系 | 按任务类型自适应 | 尊重不同领域中"困难"的不同含义 |
| 评估赛道 | 双赛道(生成+判别) | 扩大模型覆盖范围,跨赛道差距本身为研究信号 |
| 干扰项设计 | 近似正确、单约束违反 | 防止赛道 2 退化为简单模式匹配 |
| 图像格式 | SVG 生成 -> 多分辨率 PNG 分发 | 生成精确性与消费通用性兼得 |
| 分发方式 | GitHub 仓库 + HuggingFace 冻结快照 | 可重现性(生成器) + 可比较性(快照) |
| 规模设计 | v0.1.0 约 2,000-5,000 例,可扩展至 50,000+ | 配置驱动,无需代码变更即可扩展 |
| 渲染层 | 薄共享抽象,底层使用专用库 | 视觉一致性,避免过度工程化 |
| 文档语言 | 英文 + 中文 | 从第一天起面向国际受众 |

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
