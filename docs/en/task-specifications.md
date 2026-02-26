# TACIT Benchmark v0.1.0 -- Task Specifications

This document provides detailed specifications for all 10 tasks in the TACIT Benchmark. For each task, it covers the domain and reasoning type, visual description of the puzzle, solution requirements, difficulty parameters, verification logic, and distractor types.

**Source code references:** All generators are in `tacit/generators/`. Difficulty configs are in `configs/default.yaml`.

---

## Table of Contents

1. [Multi-Layer Mazes](#1-multi-layer-mazes)
2. [Raven's Progressive Matrices](#2-ravens-progressive-matrices)
3. [Cellular Automata Forward Prediction](#3-cellular-automata-forward-prediction)
4. [Cellular Automata Inverse Inference](#4-cellular-automata-inverse-inference)
5. [Visual Logic Grids](#5-visual-logic-grids)
6. [Planar Graph k-Coloring](#6-planar-graph-k-coloring)
7. [Graph Isomorphism Detection](#7-graph-isomorphism-detection)
8. [Unknot Detection](#8-unknot-detection)
9. [Orthographic Projection Identification](#9-orthographic-projection-identification)
10. [Isometric Reconstruction](#10-isometric-reconstruction)

---

## 1. Multi-Layer Mazes

**Task name:** `maze`
**Generator:** `tacit/generators/maze.py` (`MazeGenerator`)
**Domain:** Spatial / Pathfinding
**Reasoning type:** Spatial navigation, cross-layer connectivity

### Puzzle Description

The puzzle image shows one or more 2D maze grids arranged side-by-side, each labeled "Layer 1", "Layer 2", etc. The grid uses a wall/passage cell structure where wall cells are filled dark and passage cells are white. A green circle marked "S" indicates the start position (always on layer 1). A red circle marked "E" indicates the end position (always on the last layer). Colored circles at matching positions across layers indicate portal pairs that allow the path to transition between layers.

For single-layer mazes, there is one grid with no portals. For multi-layer mazes, the solver must navigate within each layer and use portals to cross between layers.

### Solution Requirements

The solution SVG must contain a hidden text element `<text id="maze-path">` encoding the path as semicolon-delimited `layer,row,col` triples. The path must form a continuous route from the start cell to the end cell.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `grid_size` | 4 -- 128 | 4 | Width/height of the maze grid (internal grid is `2*grid_size + 1`) |
| `layers` | 1 -- 8 | 1 | Number of maze layers |
| `portals` | 0 -- 20 | 1 | Number of portal pairs connecting adjacent layers |

**Example difficulty configs:**

```yaml
easy:   {grid_size: 8,  layers: 1, portals: 0}
medium: {grid_size: 16, layers: 2, portals: 2}
hard:   {grid_size: 32, layers: 3, portals: 5}
```

### Verification Logic

The verifier (`MazeGenerator.verify()`) performs four structural checks on the extracted path:

1. **Start check:** First cell in path equals the designated start position.
2. **End check:** Last cell in path equals the designated end position.
3. **Passage check:** Every cell in the path is a passage (not a wall) in its respective layer's grid.
4. **Connectivity check:** Every pair of consecutive cells is either (a) adjacent on the same layer (Manhattan distance = 1) or (b) connected by a portal between layers.

The verifier regenerates the maze from the puzzle's seed to access the grid data and portal positions.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `wall_breach` | Path passes through a wall cell. A wall-adjacent detour is inserted into the correct path. |
| `portal_skip` | Path transitions between layers without using a portal. The portal crossing step is replaced with a direct jump. |
| `disconnected` | Path has a gap -- consecutive cells are not adjacent. A segment is removed from the middle of the path. |
| `wrong_exit` | Path ends at a different passage cell instead of the designated end. The last cell is replaced with a random passage cell on the end layer. |

---

## 2. Raven's Progressive Matrices

**Task name:** `raven`
**Generator:** `tacit/generators/raven.py` (`RavenGenerator`)
**Domain:** Pattern / Sequence
**Reasoning type:** Attribute transformation inference, pattern completion

### Puzzle Description

The puzzle image shows a 3x3 grid of tiles. Each tile contains geometric shapes with attributes: shape (circle, square, triangle, diamond, pentagon, hexagon), color (from a 10-color palette), size (small, medium, large), rotation (0, 90, 180, 270 degrees), and count (1, 2, 3, or 4 instances). The bottom-right tile is missing, shown as a red "?" symbol.

Transformation rules govern how attributes change across rows and/or columns. In "additive" rules, the attribute value is determined by the column index. In "compositional" rules, the value is determined by `(row + col) % 3`, creating diagonal patterns that are harder to detect.

### Solution Requirements

The solution SVG must contain `data-tacit-*` attributes embedded as an HTML comment, specifying all five tile attributes: `shape`, `color`, `size`, `rotation`, and `count`.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `rules` | 1 -- 4 | 1 | Number of simultaneous transformation rules |
| `complexity` | 0 -- 1 | 1 | 0 = additive (column-based), 1 = compositional (row+col) |

**Example difficulty configs:**

```yaml
easy:   {rules: 1, complexity: "additive"}
medium: {rules: 2, complexity: "additive"}
hard:   {rules: 3, complexity: "compositional"}
```

### Verification Logic

The verifier (`RavenGenerator.verify()`) extracts tile attributes from both the candidate and expected solution SVGs using regex matching on `data-tacit-*` patterns. It then compares all five attributes. A mismatch in any attribute means the candidate fails.

The `VerificationResult.details` includes a `mismatches` dict showing which attributes differed and their expected vs. actual values.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `wrong_shape` | Shape attribute is changed to a different shape from the pool. |
| `wrong_color` | Color attribute is changed to a different color from the palette. |
| `wrong_rotation` | Rotation attribute is changed to a different rotation value. |
| `wrong_count` | Count attribute is changed to a different count value. |

Each distractor changes exactly one attribute, making it a near-miss that shares the correct values for the other attributes.

---

## 3. Cellular Automata Forward Prediction

**Task name:** `ca_forward`
**Generator:** `tacit/generators/ca_forward.py` (`CAForwardGenerator`)
**Domain:** Pattern / Sequence
**Reasoning type:** Rule application, temporal simulation

### Puzzle Description

The puzzle image shows two side-by-side panels. On the left is the initial grid ("State T") -- a square grid where each cell is colored according to its state value (white for state 0, colored for states 1+). On the right is a transition rule table showing the outer-totalistic rule: rows correspond to the current cell state, columns correspond to the Moore-neighborhood sum, and each cell shows the output state color. A title indicates how many steps to predict.

The cellular automaton uses a 2D Moore neighborhood (8 surrounding cells) with wrapping (toroidal) boundary conditions.

### Solution Requirements

The solution SVG must render the final grid ("State T+k") as colored rectangles using the same state-color mapping. The verifier parses the grid by extracting `<rect>` elements and mapping their fill colors back to state values.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `grid_size` | 4 -- 64 | 4 | Width/height of the square grid |
| `rule_complexity` | 2 -- 8 | 1 | Number of cell states (2 = binary, 8 = maximum) |
| `steps` | 1 -- 20 | 1 | Number of simulation steps to predict |

**Example difficulty configs:**

```yaml
easy:   {grid_size: 8,  rule_complexity: 2, steps: 1}
medium: {grid_size: 16, rule_complexity: 4, steps: 3}
hard:   {grid_size: 32, rule_complexity: 8, steps: 5}
```

### Verification Logic

The verifier (`CAForwardGenerator.verify()`) parses both the candidate and solution SVGs into grid arrays by extracting `<rect>` fill colors and mapping them to state values. It then performs cell-by-cell comparison using `numpy.array_equal`. On failure, it reports the number of differing cells out of the total.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `wrong_cell` | A few random cells are flipped to incorrect states. |
| `wrong_step_count` | The grid is the result of simulating a different number of steps (e.g., T+2 instead of T+3). |
| `wrong_rule` | The grid is the result of applying a different random rule to the initial grid for the correct number of steps. |

---

## 4. Cellular Automata Inverse Inference

**Task name:** `ca_inverse`
**Generator:** `tacit/generators/ca_inverse.py` (`CAInverseGenerator`)
**Domain:** Pattern / Sequence
**Reasoning type:** Rule inference, inverse reasoning

### Puzzle Description

The puzzle image shows two grids side by side: "State T" on the left and "State T+k" on the right, connected by an arrow labeled with the number of steps. Both grids use the same state-color mapping as CA Forward. The solver must infer the transition rule that, when applied to State T for k steps, produces State T+k.

This is the inverse of Task 3: given input and output, infer the function.

### Solution Requirements

The solution SVG must render the rule table as a grid of small colored rectangles. Rows represent current cell states, columns represent neighbor sums, and each cell's color encodes the output state.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `grid_size` | 4 -- 64 | 4 | Width/height of the square grid |
| `rule_space` | 2 -- 8 | 1 | Number of cell states in the rule space |
| `steps` | 1 -- 20 | 1 | Number of simulation steps between State T and State T+k |

**Example difficulty configs:**

```yaml
easy:   {grid_size: 8,  rule_space: 4,  steps: 1}
medium: {grid_size: 16, rule_space: 8,  steps: 2}
hard:   {grid_size: 32, rule_space: 16, steps: 3}
```

### Verification Logic

The verifier (`CAInverseGenerator.verify()`) parses the candidate rule table from SVG by extracting small colored rectangles and mapping them to state values. It performs strict entry-by-entry comparison against the expected rule table. The rule must match exactly -- even a functionally equivalent rule that produces the same output for the specific initial grid but differs in entries is rejected, because the task is to infer the exact rule.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `off_by_one_rule` | One entry in the rule table is changed to a wrong state. The rule is almost correct but produces different output for specific neighbor-sum configurations. |
| `transposed_rule` | Two entries in the rule table are swapped. The rule is structurally similar but semantically different. |
| `partial_rule` | Multiple entries are changed. The rule works for some cells but produces errors for others. |

---

## 5. Visual Logic Grids

**Task name:** `logic_grid`
**Generator:** `tacit/generators/logic_grid.py` (`LogicGridGenerator`)
**Domain:** Logical Constraint
**Reasoning type:** Constraint satisfaction, deductive reasoning

### Puzzle Description

The puzzle image shows an NxN grid with visual constraint clues. The grid uses colored geometric symbols (circle, square, triangle, diamond, star, hexagon) with no natural-language text. A symbol legend at the top shows which color maps to which symbol.

Constraint clues are rendered visually on the grid:

- **Row unique (arrow right):** A colored dot at a cell edge with a right-arrow in the margin indicates "this symbol is in this row at this column."
- **Column unique (arrow down):** A colored dot with a down-arrow in the margin indicates "this symbol is in this column at this row."
- **Exclusion (X mark):** A colored X mark in a cell means "this symbol cannot be in this cell."
- **Adjacency (= or not-equal):** A connector between two adjacent cells shows whether they must contain the same or different symbols.

The underlying structure is a Latin square: each symbol appears exactly once per row and once per column.

### Solution Requirements

The solution SVG renders the completed grid with each cell containing the correct colored symbol. The verifier parses text elements and their positions to reconstruct the grid values.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `grid_size` | 3 -- 8 | 1 | Side length of the NxN logic grid |
| `constraints` | 4 -- 20 | 1 | Number of constraint clues provided |
| `types` | 1 -- 4 | 1 | Number of distinct constraint types used |

**Example difficulty configs:**

```yaml
easy:   {grid_size: 4, constraints: 6,  types: 2}
medium: {grid_size: 5, constraints: 10, types: 3}
hard:   {grid_size: 6, constraints: 16, types: 4}
```

The generator ensures that the constraint set uniquely determines the solution by running a backtracking solver and verifying there is exactly one valid completion.

### Verification Logic

The verifier (`LogicGridGenerator.verify()`) extracts the grid by parsing symbol text elements from the SVG and mapping them to grid positions. It checks:

1. **Latin square property:** Each row and each column contains unique symbols.
2. **Constraint satisfaction:** All constraints from the puzzle metadata are satisfied.
3. **Exact match:** If the extracted grid matches the expected solution grid, verification passes immediately.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `constraint_violation` | A specific constraint is violated by swapping cells in the same row. |
| `symbol_swap` | Two symbols in a random row are swapped, breaking column uniqueness. |
| `non_unique` | An entirely different Latin square that happens to violate at least one constraint. |

---

## 6. Planar Graph k-Coloring

**Task name:** `graph_coloring`
**Generator:** `tacit/generators/graph_coloring.py` (`GraphColoringGenerator`)
**Domain:** Graph / Connectivity
**Reasoning type:** Constraint-based coloring, chromatic reasoning

### Puzzle Description

The puzzle image shows a planar graph with numbered nodes drawn as gray circles and edges drawn as gray lines on a 500x500 canvas. Node positions are determined by generating random 2D points and computing a Delaunay triangulation, then removing edges to achieve the target density while maintaining connectivity.

The puzzle view shows all nodes uncolored (gray). The model must produce a coloring using exactly k colors such that no two adjacent nodes share a color.

### Solution Requirements

The solution SVG must render the same graph with each node filled using one of k colors from the standard palette. Each node circle must have an `id="node-{id}"` attribute so the parser can map fill colors to node identifiers.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `nodes` | 4 -- 50 | 1 | Number of nodes in the graph |
| `edge_density` | 0.1 -- 1.0 | 0.1 | Fraction of Delaunay triangulation edges to keep |
| `k` | 2 -- 10 | 1 | Number of colors for the coloring (closer to chromatic number = harder) |

**Example difficulty configs:**

```yaml
easy:   {nodes: 6,  edge_density: 0.3, k: 4}
medium: {nodes: 12, edge_density: 0.4, k: 4}
hard:   {nodes: 20, edge_density: 0.5, k: 3}
```

### Verification Logic

The verifier (`GraphColoringGenerator.verify()`) regenerates the graph from the puzzle seed to obtain the adjacency structure. It then parses node fill colors from the candidate SVG using the `GraphColoringParser`. Three checks are performed:

1. **Completeness:** All nodes have an assigned color (not gray or white).
2. **Validity:** No two adjacent nodes share the same fill color.
3. **Exact k:** The number of distinct colors used equals k.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `adjacent_conflict` | Two adjacent nodes are assigned the same color. A random edge endpoint is recolored to match its neighbor. |
| `missing_color` | One color is eliminated and replaced with another, resulting in k-1 colors used. |
| `wrong_k` | An extra color (k+1-th) is introduced by recoloring a random node. |

---

## 7. Graph Isomorphism Detection

**Task name:** `graph_isomorphism`
**Generator:** `tacit/generators/graph_isomorphism.py` (`GraphIsomorphismGenerator`)
**Domain:** Graph / Connectivity
**Reasoning type:** Structural comparison, invariant recognition

### Puzzle Description

The puzzle image shows two graphs side by side, labeled "Graph A" and "Graph B." Each graph has the same number of nodes but different visual layouts (node positions). Nodes are blue circles with white numeric labels. Edges are gray lines.

The task is binary: determine whether the two graphs are isomorphic (structurally identical despite different layouts) or not.

- **Isomorphic pairs:** Graph B is created by permuting Graph A's node labels and computing a different layout.
- **Non-isomorphic pairs:** Graph B is created by adding or removing an edge from Graph A, then permuting labels and relaying out.

### Solution Requirements

The solution SVG is a badge-style indicator: a green checkmark with "Isomorphic" text, or a red X with "Not Isomorphic" text. The SVG contains a hidden `<rect>` element with `id="answer-isomorphic"` or `id="answer-not-isomorphic"` for reliable parsing.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `nodes` | 4 -- 30 | 1 | Number of nodes in each graph |
| `distortion` | 0.0 -- 1.0 | 0.1 | Layout distortion factor (random noise added to spring layout positions) |

**Example difficulty configs:**

```yaml
easy:   {nodes: 5,  distortion: 0.3}
medium: {nodes: 8,  distortion: 0.6}
hard:   {nodes: 12, distortion: 0.9}
```

Higher distortion makes the visual comparison harder because node positions become more random, obscuring the structural similarity or difference.

### Verification Logic

The verifier (`GraphIsomorphismGenerator.verify()`) parses the candidate SVG to determine which answer it represents by checking for `id="answer-isomorphic"` or `id="answer-not-isomorphic"`. It falls back to checking text content for "Isomorphic" or "Not Isomorphic." The extracted answer is compared against the ground truth stored in puzzle metadata.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `opposite_answer` | The distractor shows the opposite answer (isomorphic vs. not isomorphic). Since this is a binary task, there is only one distractor type. |

---

## 8. Unknot Detection

**Task name:** `unknot`
**Generator:** `tacit/generators/unknot.py` (`UnknotGenerator`)
**Domain:** Topology
**Reasoning type:** Topological classification, Reidemeister move reasoning

### Puzzle Description

The puzzle image shows a 2D knot diagram projection -- a closed curve with crossing indicators. At each crossing, the over-strand is drawn as a short thick line and the under-strand has a white gap, following standard knot diagram conventions. Positive crossings are marked with blue dots; negative crossings with red dots.

The task is binary: determine whether the diagram represents the unknot (topologically equivalent to a simple circle, deformable to a circle through Reidemeister moves) or a non-trivial knot.

**Unknots** are generated by starting with a circular path and inserting Reidemeister-I loops (kinks) that add crossings without changing the topological type. The resulting diagrams are visually complex but always deformable to a circle.

**Non-trivial knots** are generated using torus-knot parametrizations based on known knots: trefoil (3 crossings), figure-eight (4), cinquefoil (5), three-twist (5), stevedore (6), and knot 7_1 (7). Self-intersections are detected and rendered as crossings.

### Solution Requirements

The solution SVG is a badge showing either "unknot" (green background) or "knot" (red background) as text within the SVG. The verifier extracts the label by searching for `>unknot<` or `>knot<` in the SVG string.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `crossings` | 2 -- 15 | 1 | Number of crossings in the knot diagram |

**Example difficulty configs:**

```yaml
easy:   {crossings: 3}
medium: {crossings: 6}
hard:   {crossings: 10}
```

More crossings make the puzzle harder because the diagram becomes more tangled and harder to visually simplify.

### Verification Logic

The verifier (`UnknotGenerator.verify()`) extracts the answer label from the candidate SVG by searching for `>unknot<` or `>knot<` in the string content. It compares the extracted label against the ground truth `is_unknot` flag stored in puzzle metadata.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `opposite_answer` | The distractor shows the opposite classification (unknot vs. knot). As a binary task, there is only one distractor type. |

---

## 9. Orthographic Projection Identification

**Task name:** `ortho_projection`
**Generator:** `tacit/generators/ortho_projection.py` (`OrthoProjectionGenerator`)
**Domain:** Geometric / Projection
**Reasoning type:** 3D-to-2D mental projection, spatial visualization

### Puzzle Description

The puzzle image shows a 3D voxel solid rendered in isometric view, along with an axis indicator specifying which projection to compute (front, top, or side). The isometric view uses a standard axonometric projection with visible voxel faces colored to indicate depth.

The model must produce the correct 2D orthographic projection (silhouette) along the specified axis. This requires mentally collapsing the 3D shape along one dimension.

### Solution Requirements

The solution SVG renders the 2D projection as a grid of filled/empty cells. Verification is by exact SVG string comparison against the deterministically rendered ground truth projection.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `faces` | 4 -- 50 | 2 | Number of voxels in the 3D solid (controls shape complexity) |
| `concavities` | 0 -- 10 | 1 | Number of interior voxels removed to create concavities |

**Example difficulty configs:**

```yaml
easy:   {faces: 6,  concavities: 0}
medium: {faces: 10, concavities: 1}
hard:   {faces: 16, concavities: 3}
```

Concavities make the projection harder because removed interior voxels may or may not affect the silhouette depending on the projection axis.

### Verification Logic

The verifier (`OrthoProjectionGenerator.verify()`) performs exact SVG string comparison between the candidate and the ground truth solution. Both SVGs are deterministically rendered from the same projection data, so a correct solution must produce an identical string.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `wrong_axis` | The projection is computed along a different axis (e.g., top instead of front). |
| `missing_feature` | Some filled cells are removed from the correct projection, creating holes in the silhouette. |
| `extra_feature` | Phantom cells are added to the correct projection, extending the silhouette beyond the true shape. |
| `mirrored` | The correct projection is flipped left-to-right (or top-to-bottom if already horizontally symmetric). |

---

## 10. Isometric Reconstruction

**Task name:** `iso_reconstruction`
**Generator:** `tacit/generators/iso_reconstruction.py` (`IsoReconstructionGenerator`)
**Domain:** Geometric / Projection
**Reasoning type:** 2D-to-3D spatial reconstruction, multi-view reasoning

### Puzzle Description

The puzzle image shows three orthographic projections in a standard engineering drawing layout: front, top, and side views. Each projection is a 2D grid of filled/empty cells representing the silhouette of the 3D solid along that axis.

The model must reconstruct the correct isometric view of the 3D solid that produces all three given projections. This is the inverse of Task 9: given three 2D views, infer the 3D shape.

### Solution Requirements

The solution SVG renders the 3D voxel solid in isometric view. Verification is by exact SVG string comparison against the deterministically rendered ground truth.

### Difficulty Parameters

| Parameter | Range | Step | Description |
|-----------|-------|------|-------------|
| `faces` | 4 -- 50 | 2 | Number of voxels in the 3D solid (controls shape complexity) |
| `ambiguity` | 0 -- 10 | 1 | Number of redundant voxels removed to create projection ambiguity |

**Example difficulty configs:**

```yaml
easy:   {faces: 6,  ambiguity: 0}
medium: {faces: 10, ambiguity: 1}
hard:   {faces: 16, ambiguity: 2}
```

Higher ambiguity means the solid is thinned such that multiple 3D shapes could produce the same three projections, requiring the model to infer the specific intended reconstruction.

### Verification Logic

The verifier (`IsoReconstructionGenerator.verify()`) performs exact SVG string comparison between the candidate and the ground truth solution. Both are deterministically rendered from the same voxel grid.

### Distractor Violation Types

| Violation | Description |
|-----------|-------------|
| `wrong_depth` | A voxel is moved to a different depth position, altering the 3D shape while partially preserving some projections. |
| `missing_face` | Voxels are removed, creating a sparser solid that does not match the input projections. |
| `extra_volume` | Phantom voxels are added, creating a denser solid with incorrect projections. |
| `rotated` | The solid is rotated 90 degrees around a random axis, changing its relationship to the projection axes. |

---

## Summary Table

| # | Task | Domain | Binary? | Difficulty Axes | Violation Types |
|---|------|--------|---------|----------------|----------------|
| 1 | Multi-Layer Mazes | Spatial | No | grid_size, layers, portals | wall_breach, portal_skip, disconnected, wrong_exit |
| 2 | Raven's Matrices | Pattern | No | rules, complexity | wrong_shape, wrong_color, wrong_rotation, wrong_count |
| 3 | CA Forward | Pattern | No | grid_size, rule_complexity, steps | wrong_cell, wrong_step_count, wrong_rule |
| 4 | CA Inverse | Pattern | No | grid_size, rule_space, steps | off_by_one_rule, transposed_rule, partial_rule |
| 5 | Logic Grids | Logical | No | grid_size, constraints, types | constraint_violation, symbol_swap, non_unique |
| 6 | Graph k-Coloring | Graph | No | nodes, edge_density, k | adjacent_conflict, missing_color, wrong_k |
| 7 | Graph Isomorphism | Graph | Yes | nodes, distortion | opposite_answer |
| 8 | Unknot Detection | Topology | Yes | crossings | opposite_answer |
| 9 | Ortho Projection | Geometric | No | faces, concavities | wrong_axis, missing_feature, extra_feature, mirrored |
| 10 | Iso Reconstruction | Geometric | No | faces, ambiguity | wrong_depth, missing_face, extra_volume, rotated |

**Design note:** Tasks 3 and 4 (CA Forward/Inverse) and Tasks 9 and 10 (Ortho Projection/Iso Reconstruction) form forward-inverse pairs that test qualitatively different reasoning on the same domain. Tasks 7 and 8 (Graph Isomorphism and Unknot Detection) are binary classification tasks.
