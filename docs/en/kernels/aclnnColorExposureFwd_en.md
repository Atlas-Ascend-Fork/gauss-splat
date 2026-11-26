# ColorExposureFwd Design Document

[View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_fwd/op_kernel/arch35/color_exposure_fwd.cpp)

## 1. Basic Information

| Item | Content |
| --- | --- |
| Operator name | ColorExposureFwd |
| Repository / directory | gauss-splat / kernels/color_exposure_fwd |
| Operator type | Custom (per-gaussian color affine transform) |
| aclnn interface name | aclnnColorExposureFwd |
| Document version | v1.1 |
| Author | - |
| Reviewer | - |
| Last updated | 2026-11-23 |

## 2. Background and Objectives

### 2.1 Business Scenario

In multi-view 3D Gaussian Splatting (3DGS) training, exposure / white-balance differences across cameras cause systematic color drift of the same gaussian across images, hurting reconstruction-rendering consistency. Therefore, after SH color computation and before rasterization, each gaussian's color is transformed by a learnable exposure affine transform (a 3×4 row-augmented matrix [$[A\|b]$, 12 parameters in total), jointly optimized with gaussian attributes. Two granularities of exposure parameters are supported:

- **Global mode**: one set of 12 parameters shared across the whole batch, used by all gaussians of every (batch, camera);
- **Grid mode**: a $(G_X, G_Y, G_Z, 12)$ parameter field; per-gaussian parameters are obtained by 8-corner trilinear interpolation at the gaussian's projected coordinates $(x, y)$ and color grey value, modeling spatially varying exposure.

Inputs / outputs are organized as $[B, C, D, N]$: $B$ is the batch, $C$ the number of cameras (views) per batch, $D$ the channel dimension (3 for colors, 2 for means2d), and $N$ the per-camera gaussian capacity. The upstream projection / filtering operator outputs the valid gaussian count `cnt` $[B, C]$ per $(b, c)$: under the compact layout only the first `cnt[b, c]` gaussians of each camera are valid, and this operator skips the invalid tail gaussians accordingly, avoiding meaningless computation and memory traffic.

Exposure parameters are learnable and require gradients during training, so the forward operator (this one) and the backward operator (ColorExposureBwd) are delivered as a pair.

### 2.2 Reference Implementation

New.

### 2.3 Objectives

- Functional objectives:
  - Covered: global / grid modes; optional `clamp` non-negative truncation; grid mode outputs `index` / `weight`; `exposure` supports both $(G_X, G_Y, G_Z, 12)$ and $(G_X, G_Y, G_Z, 4, 3)$ layouts; multiple cameras $C \ge 1$; `cnt` filtering (only the first `cnt[b, c]` gaussians of each camera are computed; the remaining positions are not written back).
- Performance objective: memory-bound operator; GM bandwidth utilization close to same-platform copy-type operators (per valid gaussian: read 3~5 floats, write 3~9 floats; `cnt` filtering skips invalid tail gaussians, further removing useless traffic; to be measured).
- Accuracy objective: `output` / `weight` within rtol = atol = 1e-5 of the CPU golden; `index` bit-exact.
- Delivery scope: dtype FLOAT (`index` / `cnt` INT32), format ND, $B, C, N \ge 1$, $0 \le cnt[b,c] \le N$, every grid dimension ≥ 2 and $G_X G_Y G_Z \le 2048$ (tiling hard-check cap, i.e. maximum grid 16×16×8); Ascend 950PR / 950DT only.

## 3. Operator Definition

### 3.1 Functional Description

Applies an affine transform according to `exposure` to per-gaussian colors `colors` of shape $[B, C, 3, N]$, producing `output` of shape $[B, C, 3, N]$:

- Global mode (`exposure` has 12 elements): all gaussians share the same 12 parameters (3×4 row-augmented [$[A\|b]$), $out = Ac + b$;
- Grid mode (`exposure` has more than 12 elements, shape $(G_X, G_Y, G_Z, 12)$ or $(G_X, G_Y, G_Z, 4, 3)$): using the projected pixel coordinates `means2d` of shape $[B, C, 2, N]$ and the color grey value as interpolation arguments, the 12 parameters for each gaussian are first obtained by 8-corner trilinear interpolation over the $G_X \times G_Y \times G_Z$ grid, then the affine transform is applied;
- `clamp` = 1: output is clamped channel-wise to $\max(\cdot, 0)$;
- **cnt filtering**: the INT32 input `cnt` $[B, C]$ gives the valid gaussian count of each $(b, c)$; only gaussians $n < cnt[b, c]$ are computed; positions with $n \ge cnt[b,c]$ are not written back by the kernel and keep the output tensor's initial value (the upper layer allocates with zeros, i.e. 0);
- Grid mode additionally outputs the interpolation indices `index` (INT32) and trilinear weights `weight` (FLOAT), both of shape $[B, C, 3, N]$, for reuse by the backward operator (likewise written back only at valid gaussian positions); in global mode `index` / `weight` are empty tensors (numel = 0).

### 3.2 Mathematical Formulation

The 12 parameters are interpreted as a 3×4 row-augmented matrix [$[A\|b]$: $A_{ij} = exposure[4i+j]$, $b_i = exposure[4i+3]$ ($i, j \in \{0, 1, 2\}$).

For each $(b, c)$, only gaussians with $n < cnt[b, c]$ are computed (all other positions output 0):

**Global mode** (exposure has 12 elements):

```text
pre_i = Σ_j A_ij · c_j + b_i
out_i = clamp ? max(pre_i, 0) : pre_i
```

**Grid mode** (exposure has more than 12 elements; let the grid be $E$), three-stage pipeline:

```text
① Grey value:  grey = 0.299·r + 0.587·g + 0.114·b

② Grid coordinates (fl32 denotes fp32 rounding; no +0.5 pixel-center offset;
   out-of-range coordinates clamped to [0,1]):
    u = clamp01(x · fl32(1/W)) · (GX−1)
    v = clamp01(y · fl32(1/H)) · (GY−1)
    w = clamp01(grey)   · (GZ−1)
    ix = trunc(u), wx = u − ix; when ix == GX−1, endpoint closure (ix, wx) ← (GX−2, 1)
    (likewise for the y / z axes)

③ Reconstruct per-gaussian parameters m by 8-corner trilinear interpolation,
   apply the affine transform with m = [A|b] on colors, then clamp as in global mode:
    m_k = Σ_{(cx,cy,cz)∈{0,1}^3} wx^cx · wy^cy · wz^cz · E[ix+cx, iy+cy, iz+cz, k]

Outputs: index = (ix, iy, iz), weight = (wx, wy, wz)
```

Notes:

- $x / y$ are pixel coordinates (channels 0 / 1 of `means2d`); $W / H$ are the image width / height (attributes `width` / `height`);
- The grey value is clamped to [0,1] before multiplying by $(G_Z-1)$; negative / out-of-range colors are truncated to the grid endpoints;
- All intermediates are fp32 (grey coefficients, reciprocals, and weights are fp32 constants / fp32 arithmetic); no mixed precision;
- Gaussians with $n \ge cnt[b,c]$ produce no output at all (the corresponding positions of `output` / `index` / `weight` keep the initial value 0).

### 3.3 Inputs, Outputs, and Attributes

| Name | Direction | Description | dtype | shape | format | Optional |
| --- | --- | --- | --- | --- | --- | --- |
| colors | input | Gaussian colors (SH forward result) | FLOAT | [B, C, 3, N] | ND | No |
| means2d | input | Gaussian projected pixel coordinates | FLOAT | [B, C, 2, N] | ND | Required in grid mode; not used in global mode (pass a valid non-empty tensor) |
| exposure | input | Exposure parameters | FLOAT | global: [12] / (4,3) / (1,1,1,12); grid: (GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | No |
| cnt | input | Valid gaussian count per (b, c) (from the upstream projection / filtering) | INT32 | [B, C] | ND | No |
| width | attr | Image width W (pixels) | int32 | - | - | No |
| height | attr | Image height H (pixels) | int32 | - | - | No |
| clamp | attr | Channel-wise non-negative clamp of the output | bool (0/1) | - | - | No |
| output | output | Exposure-corrected colors | FLOAT | [B, C, 3, N] | ND | No |
| index | output | Grid interpolation indices (ix, iy, iz) | INT32 | [B, C, 3, N] | ND | Required in grid mode; empty tensor (numel = 0) in global mode |
| weight | output | Trilinear weights (wx, wy, wz) | FLOAT | [B, C, 3, N] | ND | Same as index |

### 3.4 Support Matrix

| Dimension | Supported range | Unsupported items and reasons |
| --- | --- | --- |
| Product | Ascend 950PR / 950DT (arch35) | Other products unsupported: the operator is adapted to arch35 only (SIMT VF / Reg vector instructions); no builds for other architectures |
| dtype | FLOAT (index / cnt INT32) | fp16 / bf16 / integer colors etc. unsupported: the color path is fp32 throughout; no mixed-precision requirement |
| format | ND | Other formats unsupported: the kernel addresses memory with a compact layout |
| shape | colors [B,C,3,N], means2d [B,C,2,N], cnt [B,C], B ≥ 1, C ≥ 1, N ≥ 1, 0 ≤ cnt[b,c] ≤ N; exposure numel = 12 or (GX,GY,GZ,12) / (GX,GY,GZ,4,3), every grid dimension ≥ 2, $G_X G_Y G_Z \le 2048$ (MAX_GRID_DIMS hard check, i.e. maximum 16×16×8, see 5.2) | Empty tensors (0-sized dims) and 0-dim inputs unsupported; cnt > N treated as full, negative values undefined (contract: not passed in); grid dimension = 1 see Section 8 |
| Broadcast | Not supported | exposure is a semantically fixed parameter shared by the whole batch / cameras, not a broadcast |

### 3.5 Boundary and Exception Cases

| Case | Expected behavior |
| --- | --- |
| cnt[b, c] = 0 | The whole camera is skipped (break at the first taskLoop tile); output / index / weight of that (b, c) all stay 0 |
| cnt[b, c] < N | Only the first cnt gaussians are written back; the invalid tail gaussians are neither read nor written; when cnt is not 8-aligned or falls inside a taskLoop tile, it is truncated by the mask (validTaskNum) |
| cnt[b, c] > N | Treated as full (= N); contract: callers do not pass this |
| Empty tensor (0-sized dim) | Not supported: required inputs / outputs non-empty; in global mode index / weight are empty tensors (numel = 0) meaning "not output"; the kernel does not allocate or write back |
| Scalar / 0-dim input | Not supported; parameter validation error (161002) |
| Non-contiguous input | Not supported: the kernel addresses a compact layout; the caller must ensure contiguity (upper layers do not convert automatically) |
| NaN / Inf in inputs | Undefined: $\max(\cdot, 0)$ with NaN is platform-dependent; the kernel performs no special checks; injection not recommended |
| Invalid shape or dtype | Error 161001 (null pointer) / 161002 (dtype or shape validation failure), see 4.2 |
| Insufficient workspace | Not a user-side concern: managed by the aclnn framework per the GetWorkspaceSize return value (system workspace only) |

## 4. Interface Design

### 4.1 aclnn Interface and Call Flow

```text
aclnnColorExposureFwdGetWorkspaceSize(colors, means2d, exposure, cnt, width, height, clamp,
                                      output, index, weight, workspaceSize, executor)
aclnnColorExposureFwd(workspace, workspaceSize, executor, stream)
```

- Workspace rule: only the system workspace (LibApiWorkSpaceSize) is used; no operator-custom workspace. Allocate per the `GetWorkspaceSize` return value; the same rule applies to empty-tensor cases.
- Executor reuse rule: each `GetWorkspaceSize` performs parameter validation and creates an executor consumed by one `aclnnColorExposureFwd` execution; re-invoking with the same inputs requires going through the two-stage flow again (standard aclnn semantics).

### 4.2 Parameter Validation and Errors

| Check | Rule | Error code / log keyword |
| --- | --- | --- |
| Null pointer | All input / output tensor pointers and workspaceSize / executor non-null | 161001, "param ... is null" |
| dtype combination | colors / means2d / exposure / output / weight are FLOAT, index / cnt are INT32 | 161002 |
| shape consistency | colors [B,C,3,N] and means2d [B,C,2,N] agree on B, C, N; cnt is [B,C]; exposure numel ≥ 12; in grid mode index / weight must be [B,C,3,N], empty tensors in global mode | 161002 |
| format | ND | 161002 |

(Validation is performed by the aclnn adapter generated at op-package build time, consistent with other operators in this repository.)

### 4.3 Upper-Layer Framework Mapping

No native torch operator corresponds. Upper-layer chain: `gauss_splat.ops.color_exposure` (torch.autograd.Function) → `gauss_splat/csrc/ColorExposureFwd.cpp` (EXEC_NPU_CMD direct call, strides preserved) → aclnnColorExposureFwd. The rendering pipeline `gauss_splat.ops.rendering` invokes this operator after projection / filtering; `cnt` is exactly the filtered count output by the projection operator.

| Upper-layer parameter | This operator's parameter / attr | Notes |
| --- | --- | --- |
| colors: Tensor [B,C,3,N] | colors | must be contiguous |
| means2d: Tensor [B,C,2,N] | means2d | used only in grid mode |
| exposure: Tensor | exposure | shared across the whole batch / cameras |
| cnt: Tensor [B,C] (INT32) | cnt | valid gaussian count from the upstream projection / filtering |
| width / height: int | attrs width / height | image size in pixels |
| clamp: bool = True | attr clamp (0/1) | on by default (rendering config clamp_color_min) |
| returns output (autograd saves index / weight for backward) | output / index / weight | - |

Handling of unsupported parameters: no gradient for `means2d` / `cnt` (not needed upstream); non-contiguous inputs must be made `.contiguous()` by the caller.

## 5. Implementation

### 5.1 Layered Structure

| Layer | Responsibility | Key files |
| --- | --- | --- |
| aclnn adapter (op_api) | Interface definition, parameter validation, workspace calculation | Generated at op-package build (no handwritten op_api in this repo) |
| op_host | Attribute parsing, global / grid mode determination, tiling | kernels/color_exposure_fwd/op_host/arch35/color_exposure_fwd.cpp, color_exposure_fwd_tiling.h |
| op_kernel | GM↔UB data movement and computation | op_kernel/arch35/color_exposure_fwd.cpp, apply_grid_exposure_vf.h |
| Framework binding | PyTorch / aten bridge | gauss_splat/csrc/ColorExposureFwd.cpp |

### 5.2 Tiling Strategy

- Split dimension and order: the gaussian dimension N is the only split dimension (the batch × camera dimensions loop inside the kernel); N is rounded up to a multiple of ALIGN_VALUE = 8 as totalTaskNum, with tail padding tailNum = totalTaskNum − N;
- Tile size determination:
  - Per-task queue budget bufferSize (floats / task): grid mode $3 \times 2 \times 2 + 8 = 20$ (colors / output queues 2×3 each, means2d queue 2×2, index 3, grey 1); global mode $3 \times 2 \times 2 = 12$;
  - `taskNumPerLoop = (ubTotalSize × 0.8 − expSize) / (bufferSize × 4)`, rounded up to a multiple of 8;
  - `expSize = (G_X·G_Y·G_Z × 12 + B·C) × 4` (global mode counts the grid as 1×1×1; B·C is the resident cnt buffer); grid mode additionally reserves DCACHE_SIZE = 32KB and calls `SetDynUBufSize(ubTotalSize − 32KB)`, a region dedicated to the SIMT VF DCache to avoid VF out-of-bounds caused by conflict with UB queues;
- Multi-core / multi-block assignment: blockDim = number of AIV cores; taskNumPerScore = (totalTaskNum / blockDim / 8) × 8 is the short-core task count, long cores get taskNumPerLcore = taskNumPerScore + 8; numScore / numLcore split the remainder, with the first numLcore blocks being long cores; degenerate cases (taskNumPerScore or taskNumPerLcore = 0) shrink blockDim;
- Grid size cap: the tiling hard check requires $G_X G_Y G_Z \times 12 \le 24576$ (MAX_GRID_DIMS), i.e. $G_X G_Y G_Z \le 2048$, maximum grid 16×16×8; exceeding it fails tiling directly; the UB budget $G_X G_Y G_Z \le 3549$ (roughly 15×15×15, Ascend 950 UB = 253952 B) is non-binding (the hard check takes effect first);
- UB / L1 / L0 planning: no L1 / L0; within UB: queue region (0.8 budget) + exposure resident region + cnt resident region ($B \times C \times 4$ B) + 32KB DCache reservation (grid mode); at 16×16×8 the exposure resident region is 96KB and taskNumPerLoop ≈ 904 (Ascend 950) — runs normally;
- Tail and misalignment handling: the tail tile uses DataCopyPad masked copies, validTaskNum = taskNumPerCurLoop − tailNum (last tile only); cnt truncation likewise converges into this masked path (see 5.3);
- TilingKey design: always 1; mode branching is decided inside the kernel from tiling fields (useGrid / clamp), no key splitting.

### 5.3 Kernel Implementation

- Data movement path: exposure and cnt ($B \times C$ INT32) are each copied GM→UB once and stay resident (DataCopyPad, V→MTE2 event sync); colors / means2d per tile GM→UB; output (and index / weight in grid mode) per tile UB→GM;
- Pipeline: CopyIn → Compute → CopyOut advance serially with event synchronization; queues are double-buffered (bufferNum = 2); outer batch × camera × taskLoop triple loop;
- **cnt filtering implementation**: for each (b, c), `cnt_ = cnt[b × C + c]` is read first; each tile is checked by ComputingTaskNum for `taskStart < cnt_`, and the check failing breaks out of the remaining tiles of this camera; when cnt falls inside a tile (taskStart < cnt < taskEnd), the tile is truncated with `validTaskNum = cnt − taskStart` and forced onto the tail-mask path (DataCopyPad masked copies + VEC / VF computing with the validTaskNum mask), implementing an "early end inside the tile";
- Compute primitives used:
  - Global mode: pure VEC path (ApplyGlobalExposure), Reg-level Muls / Add / Maxs, 64 lanes per batch;
  - Grid mode: PrepareCalculate (VEC path, 64 lanes per batch, vectorized: grey → coordinate × fp32 reciprocal → clamp01 → ×(G−1) → trunc → endpoint closure; writes wx / wy / wz / index back to UB buffers), then `asc_vf_call` launches ApplyGridExposureVF (SIMT VF, dim3{1024, 1, 1}, one gaussian per thread: 8 corners × 3 interpolations to rebuild the 12 parameters → affine → clamp);
- Loops and boundary handling: each tile gets taskNumPerCurLoop, isTailBlock, and validTaskNum (the valid task count after cnt truncation) from ComputingTaskNum; both VEC / VF mask the tail with validTaskNum;
- Special-case implementation: global mode launches no VF and allocates no means2d / index / grey buffers; special exposures such as identity have no dedicated fast path (they go through the generic path); with cnt = 0, all tiles of the camera break immediately.

### 5.4 Special-Case Handling

| Case | Solution |
| --- | --- |
| Small shape (N = 1) | totalTaskNum = 8, blockDim shrink + tail masking |
| Misaligned shape (N % 8 ≠ 0) | tailNum masking + DataCopyPad, no separate branch |
| cnt = 0 (camera-level full skip) | break at the first taskLoop tile, no reads or writes at all |
| cnt not 8-aligned / falling inside a tile | validTaskNum truncation + forced tail-mask path |
| Large shape / multi-core | taskLoop tiling (verified up to N = 100000), long/short core load balancing |
| Multiple cameras (C > 1) | the camera dimension reuses the same queues / resident buffers; cnt read per camera |
| Empty tensor | Not supported (see 3.5) |

## 6. Accuracy and Performance

### 6.1 Accuracy Standards and Verification

- Golden source: the CPU reference implementation inside tests/ops/test_color_exposure.py (fp32, bit-aligned with the kernel's multiply-by-reciprocal / trunc / endpoint-closure order; masked by cnt, invalid positions set to 0); the NPU-side C++ cases (tests/kernels/test_color_exposure_fwd.cpp) embed an equivalent CPU golden (cnt mixed mode: 0 / 1 / N/2 / N);
- Comparison method: index compared strictly with torch.equal; output / weight compared with torch.allclose after fp32 conversion;
- Error thresholds: output / weight rtol = atol = 1e-5; index zero tolerance;
- Mixed precision: none. Everything is fp32; the grey coefficients (0.299 / 0.587 / 0.114) and coordinate reciprocals are fp32 constants.

### 6.2 Performance Targets and Estimation

| Metric | Theoretical | Target | Measured | Achievement |
| --- | --- | --- | --- | --- |
| Latency | Memory-bound: per valid gaussian read 3 (global) / 5 (grid) floats, write 3 / 9 floats; invalid tail gaussians cost nothing | Close to same-platform copy-type operators | TBD | - |
| Bandwidth utilization | - | Same as above | TBD | - |

- Bottleneck analysis: GM data movement dominates; grid mode adds per-gaussian 8-corner × 12-parameter interpolation cost on the VF;
- Optimizations: double-buffered queues, VEC 64-lane vectorized preprocessing, explicit index / weight outputs to avoid backward recomputation, cnt filtering to skip invalid gaussians.

## 7. Test Design

### 7.1 Functional Cases

- NPU C++ cases (tests/kernels/test_color_exposure_fwd.cpp, 13 cases): global_basic (B=1, N=4096), global_identity_exact (identity exposure, output bit-equal to input), global_clamp_tail (clamp + tail N=123), grid_aligned (7×7×8), grid_clamp_tail (2×2×2 minimum grid + clamp), grid_tiny (N=7), grid_5d_exposure ((4,5,6,4,3) layout), grid_large (N=100000, 12×12×8), grid_max_16x16x8 (B=2, C=2, N=517, 16×16×8 hard-check cap), global_multi_camera (B=2, C=3), grid_multi_camera_cnt_mixed (B=2, C=3, mixed cnt), grid_cnt_mixed_tail (B=2, C=2, N=123, mixed cnt), grid_cnt_mixed_large (B=1, C=4, N=100000, mixed cnt); the mixed cnt mode takes 0 / 1 / N/2 / N by the (b, c) flat index, covering the four paths of whole-camera skip, single gaussian, in-tile truncation, and full;
- CPU golden comparison (tests/ops/test_color_exposure.py): test_fwd_global_random, test_fwd_grid_random (both with multiple cameras C ∈ {1, 2, 3}), test_fwd_cnt_filter (cnt all-0 / mixed / crossing taskLoop boundaries), test_fwd_grid_5d_exposure, test_fwd_grid_max_16x16x8 (16×16×8 hard-check cap).

### 7.2 Boundary Cases

- test_fwd_boundary: injects 7 boundary coordinate groups ((0,0), (W−1,H−1), (W,H), positive / negative out-of-range, axis endpoints) × 7 boundary color groups (pure black, out-of-range 2.0, negative −2.0, grey-coefficient colors, etc.), covering all branches of clamp01, trunc, endpoint closure, and grey saturation (B=2, C=2);
- Tail and misalignment: N = 7 / 13 / 123 / 257; minimum grid 2×2×2; multi batch / camera B ∈ {1, 2, 3}, C ∈ {1, 2, 3}.

### 7.3 Accuracy Cases

- All functional cases compare against golden: output / weight allclose(rtol=atol=1e-5), index torch.equal;
- Value-pinning cases: identity exposure → output bit-equal to colors; a designated element of $A$ set to k → the corresponding output channel equals $k \times$ the source channel element-wise;
- cnt filtering pinning: cameras with cnt = 0 have all-zero output; with cnt < N, the [cnt, N) range of output / index / weight is all zero.

### 7.4 Performance and Generalization Cases

- Shape sweep: N ∈ {1, 7, 13, 123, 257, 4096, 100000}, B ∈ {1, 2, 3}, C ∈ {1, 2, 3, 4}, grids 2×2×2 ~ 16×16×8 (including the hard-check cap), clamp ∈ {0, 1}, exposure 4d / 5d layouts, cnt ∈ {full, mixed, all-0};
- Performance baseline: to be added (currently focused on functional and accuracy verification).

## 8. Risks and Alternatives

| Risk | Impact | Mitigation / alternative |
| --- | --- | --- |
| cnt semantic violation (cnt > N or negative) | cnt > N treated as full (correct results); negative values are undefined through unsigned comparison | Contract $0 \le cnt[b,c] \le N$ (guaranteed by the upstream projection / filtering); add validation in the aclnn adapter if defense is needed |
| Output not pre-zeroed | Invalid gaussian positions keep stale values | Semantic contract: the output tensor must be pre-zeroed; the upper layer already allocates with zeros; bare aclnn callers must check themselves |
| Any grid dimension = 1 | Endpoint closure gives ix−1 = −1; the index falls back to the neighboring cell; unexpected results | Constrain every dimension ≥ 2 (inherited from the CUDA reference); to support it, add an ix > 0 guard in the closure branch and update golden accordingly |
| Grid size over the cap | Tiling fails directly (hard check $G_X G_Y G_Z \times 12 \le 24576$); over the UB budget, taskNumPerLoop underflows and results are wrong | Constrain $G_X G_Y G_Z \le 2048$ (i.e. maximum 16×16×8, enforced by the hard check); the UB budget cap of 3549 is non-binding; larger grids require raising MAX_GRID_DIMS plus splitting the exposure field or block-wise staging |
| 32KB DCACHE reservation mismatched with hardware | Under-reserved → SIMT VF out-of-bounds access; over-reserved → UB waste | 32KB is a conservative empirical value; calibrate on hardware |
| NaN + clamp | max(NaN, 0) is platform-dependent; undefined results | Declared unsupported; supporting it requires NaN checks (performance cost) |
| fp32 multiply-by-reciprocal vs division | ~1 ulp difference from the CUDA reference; index may differ by 1 at boundaries | Golden replicates this implementation; cross-platform comparison must tolerate the difference |

## 9. Appendix

### 9.1 Glossary

| Term | Meaning |
| --- | --- |
| 12-parameter affine / [$[A\|b]$ | 3×4 row-augmented matrix: $A_{ij} = exposure[4i+j]$, $b_i = exposure[4i+3]$, out = Ac + b |
| cnt filtering | Only the first cnt[b, c] gaussians of each (b, c) are computed (compact layout); the remaining positions are not written back |
| Endpoint closure | When a grid coordinate lands on the last cell (ix = G−1), fall back to (ix, wx) ← (G−2, 1) so 8-corner interpolation stays in bounds |
| SIMT VF | The SIMT vector engine of Ascend 950 arch35 (launched via asc_vf_call, 1024 threads) |
| Long / short cores | Multi-core load-balancing: long cores carry 8 extra tasks; the rest are short cores |
| taskNumPerLoop | Number of tasks (gaussians) processed per tile on one core, derived from the UB budget |
| validTaskNum | Number of tasks actually computed inside a tile: the tail subtracts tailNum; under cnt truncation it is the remaining valid count inside the tile |

### 9.2 References

- docs/en/kernels/aclnnColorExposureBwd_en.md: design document of the paired backward operator
- CANN Ascend C programming guide (TPipe / Tiling / DataCopyPad / asc_vf_call / Reg vector instructions)
- gsplat CUDA reference implementation

### 9.3 Changelog

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | Initial version |
| v1.1 | 2026-11-23 | - | Inputs / outputs restructured from [B, C, N] (batch×camera fused) to [B, C, D, N] (explicit camera dimension); added the INT32 input cnt (valid gaussian count per (b, c)), with the kernel skipping invalid tail gaussians per cnt; index / weight shapes changed accordingly to [B, C, 3, N]; updated tiling (cnt resident buffer, triple loop) and test design; added a maximum-grid 16×16×8 (MAX_GRID_DIMS hard-check cap) boundary case |
