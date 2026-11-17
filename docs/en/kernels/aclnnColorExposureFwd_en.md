# ColorExposureFwd Design Document

[View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_fwd/op_kernel/arch35/color_exposure_fwd.cpp)

## 1. Basic Information

| Item | Content |
| --- | --- |
| Operator name | ColorExposureFwd |
| Repository / directory | gauss-splat / kernels/color_exposure_fwd |
| Operator type | Custom (per-gaussian color affine transform) |
| aclnn interface name | aclnnColorExposureFwd |
| Document version | v1.0 |
| Author | - |
| Reviewer | - |
| Last updated | 2026-11-16 |

## 2. Background and Objectives

### 2.1 Business Scenario

In multi-view 3D Gaussian Splatting (3DGS) training, exposure / white-balance differences across cameras cause systematic color drift of the same gaussian across images, hurting reconstruction-rendering consistency. Therefore, after SH color computation and before rasterization, each gaussian's color is transformed by a learnable exposure affine transform (a 3×4 row-augmented matrix $[A|b]$, 12 parameters in total), jointly optimized with gaussian attributes. Two granularities of exposure parameters are supported:

- **Global mode**: one set of 12 parameters per view (batch), shared by all gaussians in the batch;
- **Grid mode**: a $(G_X, G_Y, G_Z, 12)$ parameter field; per-gaussian parameters are obtained by 8-corner trilinear interpolation at the gaussian's projected coordinates $(x, y)$ and color grey value, modeling spatially varying exposure.

Exposure parameters are learnable and require gradients during training, so the forward operator (this one) and the backward operator (ColorExposureBwd) are delivered as a pair.

### 2.2 Reference Implementation

The operator matches the mathematical semantics of the `color_exposure` CUDA reference implementation in the gsplat ecosystem (per-gaussian 12-parameter affine + grid exposure field). Differences of this implementation:

- Hardware path: based on Ascend 950 (arch35) AIV vector cores + SIMT VF; the coordinate-to-grid-index mapping is vectorized on the VEC path, while 8-corner interpolation and the affine application run on the SIMT VF;
- The forward explicitly outputs the interpolation intermediates `index` / `weight` for direct reuse by the backward operator, avoiding recomputation of the interpolation;
- The coordinate mapping uses fp32 multiply-by-reciprocal $x \cdot \mathrm{fl32}(1/W)$ instead of division, giving ~1 ulp difference versus the CUDA version; golden and accuracy cases replicate this implementation bit-exactly;
- The behavior that a grid dimension of 1 makes the endpoint closure produce a $-1$ index is inherited from the CUDA reference implementation (see Section 8).

### 2.3 Objectives

- Functional objectives:
  - Covered: global / grid modes; optional `clamp` non-negative truncation; grid mode outputs `index` / `weight`; `exposure` supports both $(G_X, G_Y, G_Z, 12)$ and $(G_X, G_Y, G_Z, 4, 3)$ layouts;
  - Explicitly not covered: gradients of `means2d`, per-batch independent exposure, non-contiguous inputs, low-precision dtypes.
- Performance objective: memory-bound operator; GM bandwidth utilization close to same-platform copy-type operators (per gaussian: read 3~5 floats, write 3~9 floats; to be measured).
- Accuracy objective: `output` / `weight` within rtol = atol = 1e-5 of the CPU golden; `index` bit-exact.
- Delivery scope: dtype FLOAT (`index` INT32), format ND, $B, N \ge 1$, every grid dimension ≥ 2, grid bytes bounded by the UB budget (see 5.2); Ascend 950PR / 950DT only.

## 3. Operator Definition

### 3.1 Functional Description

Applies an affine transform to per-gaussian colors `colors` of shape $[B, 3, N]$ according to `exposure`, producing `output` of shape $[B, 3, N]$:

- Global mode (`exposure` has 12 elements): all gaussians in the batch share the same 12 parameters (3×4 row-augmented $[A|b]$), $out = Ac + b$;
- Grid mode (`exposure` has more than 12 elements, shape $(G_X, G_Y, G_Z, 12)$ or $(G_X, G_Y, G_Z, 4, 3)$): using the projected pixel coordinates `means2d` of shape $[B, 2, N]$ and the color grey value as interpolation arguments, the 12 parameters for each gaussian are first obtained by 8-corner trilinear interpolation over the $G_X \times G_Y \times G_Z$ grid, then the affine transform is applied;
- `clamp` = 1: output is clamped channel-wise to $\max(\cdot, 0)$;
- Grid mode additionally outputs the interpolation indices `index` (INT32) and trilinear weights `weight` (FLOAT), both of shape $[B, 3, N]$, for reuse by the backward operator; in global mode `index` / `weight` are empty tensors (numel = 0).

### 3.2 Mathematical Formulation

The 12 parameters are interpreted as a 3×4 row-augmented matrix $[A|b]$: $A_{ij} = exposure[4i+j]$, $b_i = exposure[4i+3]$ ($i, j \in \{0, 1, 2\}$).

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
- All intermediates are fp32 (grey coefficients, reciprocals, and weights are fp32 constants / fp32 arithmetic); no mixed precision.

### 3.3 Inputs, Outputs, and Attributes

| Name | Direction | Description | dtype | shape | format | Optional |
| --- | --- | --- | --- | --- | --- | --- |
| colors | input | Gaussian colors (SH forward result) | FLOAT | [B, 3, N] | ND | No |
| means2d | input | Gaussian projected pixel coordinates | FLOAT | [B, 2, N] | ND | Required in grid mode; not used in global mode (pass a valid non-empty tensor) |
| exposure | input | Exposure parameters | FLOAT | global: [12] / (4,3) / (1,1,1,12); grid: (GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | No |
| width | attr | Image width W (pixels) | int64 | - | - | No |
| height | attr | Image height H (pixels) | int64 | - | - | No |
| clamp | attr | Channel-wise non-negative clamp of the output | int64 (0/1) | - | - | No |
| output | output | Exposure-corrected colors | FLOAT | [B, 3, N] | ND | No |
| index | output | Grid interpolation indices (ix, iy, iz) | INT32 | [B, 3, N] | ND | Required in grid mode; empty tensor (numel = 0) in global mode |
| weight | output | Trilinear weights (wx, wy, wz) | FLOAT | [B, 3, N] | ND | Same as index |

### 3.4 Support Matrix

| Dimension | Supported range | Unsupported items and reasons |
| --- | --- | --- |
| Product | Ascend 950PR / 950DT (arch35) | Other products unsupported: the operator is adapted to arch35 only (SIMT VF / Reg vector instructions); no builds for other architectures |
| dtype | FLOAT (index INT32) | fp16 / bf16 / integer etc. unsupported: the color path is fp32 throughout; no mixed-precision requirement |
| format | ND | Other formats unsupported: the kernel addresses memory with a compact layout |
| shape | colors [B,3,N], means2d [B,2,N], B ≥ 1, N ≥ 1; exposure numel = 12 or (GX,GY,GZ,12) / (GX,GY,GZ,4,3), every grid dimension ≥ 2; grid bytes $G_X G_Y G_Z \times 48\,B$ bounded by the UB budget (see 5.2) | Empty tensors (0-sized dims) and 0-dim inputs unsupported; grid dimension = 1 see Section 8 |
| Broadcast | Not supported | exposure is a semantically fixed parameter shared by the whole batch, not a broadcast |

### 3.5 Boundary and Exception Cases

| Case | Expected behavior |
| --- | --- |
| Empty tensor (0-sized dim) | Not supported: required inputs / outputs must be non-empty; in global mode index / weight are empty tensors (numel = 0) meaning "not output"; the kernel does not allocate or write back |
| Scalar / 0-dim input | Not supported; parameter validation error (161002) |
| Non-contiguous input | Not supported: the kernel addresses a compact layout; the caller must ensure contiguity (upper layers do not convert automatically) |
| NaN / Inf in inputs | Undefined: $\max(\cdot, 0)$ with NaN is platform-dependent; the kernel performs no special checks; injection not recommended |
| Invalid shape or dtype | Error 161001 (null pointer) / 161002 (dtype or shape validation failure), see 4.2 |
| Insufficient workspace | Not a user-side concern: managed by the aclnn framework per the GetWorkspaceSize return value (system workspace only) |

## 4. Interface Design

### 4.1 aclnn Interface and Call Flow

```text
aclnnColorExposureFwdGetWorkspaceSize(colors, means2d, exposure, width, height, clamp,
                                      output, index, weight, workspaceSize, executor)
aclnnColorExposureFwd(workspace, workspaceSize, executor, stream)
```

- Workspace rule: only the system workspace (LibApiWorkSpaceSize) is used; no operator-custom workspace. Allocate per the `GetWorkspaceSize` return value; the same rule applies to empty-tensor cases.
- Executor reuse rule: each `GetWorkspaceSize` performs parameter validation and creates an executor consumed by one `aclnnColorExposureFwd` execution; re-invoking with the same inputs requires going through the two-stage flow again (standard aclnn semantics).

### 4.2 Parameter Validation and Errors

| Check | Rule | Error code / log keyword |
| --- | --- | --- |
| Null pointer | All input / output tensor pointers and workspaceSize / executor non-null | 161001, "param ... is null" |
| dtype combination | colors / means2d / exposure / output / weight are FLOAT, index is INT32 | 161002 |
| shape consistency | colors [B,3,N] and means2d [B,2,N] agree on B and N; exposure numel ≥ 12; in grid mode index / weight must be [B,3,N]; empty tensors in global mode | 161002 |
| format | ND | 161002 |

(Validation is performed by the aclnn adapter generated at op-package build time, consistent with other operators in this repository.)

### 4.3 Upper-Layer Framework Mapping

No native torch operator corresponds. Upper-layer chain: `gauss_splat.ops.color_exposure` (torch.autograd.Function) → `gauss_splat/csrc/ColorExposureFwd.cpp` (EXEC_NPU_CMD direct call, strides preserved) → aclnnColorExposureFwd.

| Upper-layer parameter | This operator's parameter / attr | Notes |
| --- | --- | --- |
| colors: Tensor [B,3,N] | colors | must be contiguous |
| means2d: Tensor [B,2,N] | means2d | used only in grid mode |
| exposure: Tensor | exposure | shared across the batch |
| width / height: int | attrs width / height | image size in pixels |
| clamp: bool = False | attr clamp (0/1) | off by default |
| returns output (autograd saves index / weight for backward) | output / index / weight | - |

Handling of unsupported parameters: no gradient for `means2d` (not needed upstream); non-contiguous inputs must be made `.contiguous()` by the caller.

## 5. Implementation

### 5.1 Layered Structure

| Layer | Responsibility | Key files |
| --- | --- | --- |
| aclnn adapter (op_api) | Interface definition, parameter validation, workspace calculation | Generated at op-package build (no handwritten op_api in this repo) |
| op_host | Attribute parsing, global / grid mode determination, tiling | kernels/color_exposure_fwd/op_host/arch35/color_exposure_fwd.cpp, color_exposure_fwd_tiling.h |
| op_kernel | GM↔UB data movement and computation | op_kernel/arch35/color_exposure_fwd.cpp, apply_grid_exposure_vf.h |
| Framework binding | PyTorch / aten bridge | gauss_splat/csrc/ColorExposureFwd.cpp |

### 5.2 Tiling Strategy

- Split dimension and order: the gaussian dimension N is the only split dimension (the batch dimension loops inside the kernel); N is rounded up to a multiple of ALIGN_VALUE = 8 as totalTaskNum, with tail padding tailNum = totalTaskNum − N;
- Tile size determination:
  - Per-task queue budget bufferSize (floats / task): grid mode $3 \times 2 \times 2 + 8 = 20$ (colors / output queues 2×3 each, means2d queue 2×2, index 3, grey 1); global mode $3 \times 2 \times 2 = 12$;
  - `taskNumPerLoop = (ubTotalSize × 0.8 − expSize) / (bufferSize × 4)`, rounded up to a multiple of 8;
  - `expSize = G_X·G_Y·G_Z × 12 × 4` (global mode counts 1×1×1); grid mode additionally reserves DCACHE_SIZE = 32KB and calls `SetDynUBufSize(ubTotalSize − 32KB)`, a region dedicated to the SIMT VF DCache to avoid VF out-of-bounds caused by conflict with UB queues;
- Multi-core / multi-block assignment: blockDim = number of AIV cores; taskNumPerScore = (totalTaskNum / blockDim / 8) × 8 is the short-core task count, long cores get taskNumPerLcore = taskNumPerScore + 8; numScore / numLcore split the remainder, with the first numLcore blocks being long cores; degenerate cases (taskNumPerScore or taskNumPerLcore = 0) shrink blockDim;
- UB / L1 / L0 planning: no L1 / L0; within UB: queue region (0.8 budget) + exposure resident region + 32KB DCache reservation (grid mode); on Ascend 950 (UB = 253952 B), grid mode requires $G_X G_Y G_Z \le 3549$ (roughly 15×15×15);
- Tail and misalignment handling: the tail tile uses DataCopyPad masked copies, validTaskNum = taskNumPerCurLoop − tailNum (last tile only);
- TilingKey design: always 1; mode branching is decided inside the kernel from tiling fields (useGrid / clamp), no key splitting.

### 5.3 Kernel Implementation

- Data movement path: exposure is copied GM→UB once and stays resident (DataCopyPad, V→MTE2 event sync); colors / means2d per tile GM→UB; output (and index / weight in grid mode) per tile UB→GM;
- Pipeline: CopyIn → Compute → CopyOut advance serially with event synchronization; queues are double-buffered (bufferNum = 2); outer batch × taskLoop double loop;
- Compute primitives used:
  - Global mode: pure VEC path (ApplyGlobalExposure), Reg-level Muls / Add / Maxs, 64 lanes per batch;
  - Grid mode: PrepareCalculate (VEC path, 64 lanes per batch, vectorized: grey → coordinate × fp32 reciprocal → clamp01 → ×(G−1) → trunc → endpoint closure; writes wx / wy / wz / index back to UB buffers), then `asc_vf_call` launches ApplyGridExposureVF (SIMT VF, dim3{1024, 1, 1}, one gaussian per thread: 8 corners × 3 interpolations to rebuild the 12 parameters → affine → clamp);
- Loops and boundary handling: each tile gets taskNumPerCurLoop and isTailBlock from ComputingTaskNum; both VEC / VF mask the tail with validTaskNum;
- Special-case implementation: global mode launches no VF and allocates no means2d / index / grey buffers; special exposures such as identity have no dedicated fast path (they go through the generic path).

### 5.4 Special-Case Handling

| Case | Solution |
| --- | --- |
| Small shape (N = 1) | totalTaskNum = 8, blockDim shrink + tail masking |
| Misaligned shape (N % 8 ≠ 0) | tailNum masking + DataCopyPad, no separate branch |
| Large shape / multi-core | taskLoop tiling (verified up to N = 100000), long/short core load balancing |
| Empty tensor | Not supported (see 3.5) |

## 6. Accuracy and Performance

### 6.1 Accuracy Standards and Verification

- Golden source: the CPU reference implementation inside tests/ops/test_color_exposure.py (fp32, bit-aligned with the kernel's multiply-by-reciprocal / trunc / endpoint-closure order); the NPU-side C++ cases (tests/kernels/test_color_exposure_fwd.cpp) embed an equivalent CPU golden;
- Comparison method: index compared strictly with torch.equal; output / weight compared with torch.allclose after fp32 conversion;
- Error thresholds: output / weight rtol = atol = 1e-5; index zero tolerance;
- Mixed precision: none. Everything is fp32; the grey coefficients (0.299 / 0.587 / 0.114) and coordinate reciprocals are fp32 constants.

### 6.2 Performance Targets and Estimation

| Metric | Theoretical | Target | Measured | Achievement |
| --- | --- | --- | --- | --- |
| Latency | Memory-bound: per gaussian read 3 (global) / 5 (grid) floats, write 3 / 9 floats | Close to same-platform copy-type operators | TBD | - |
| Bandwidth utilization | - | Same as above | TBD | - |

- Bottleneck analysis: GM data movement dominates; grid mode adds per-gaussian 8-corner × 12-parameter interpolation cost on the VF;
- Optimizations: double-buffered queues, VEC 64-lane vectorized preprocessing, explicit index / weight outputs to avoid backward recomputation.

## 7. Test Design

### 7.1 Functional Cases

- NPU C++ cases (tests/kernels/test_color_exposure_fwd.cpp, 8 cases): global_basic (B=1, N=4096), global_identity_exact (identity exposure, output bit-equal to input), global_clamp_tail (clamp + tail N=123), grid_aligned (7×7×8), grid_clamp_tail (2×2×2 minimum grid + clamp), grid_tiny (N=7), grid_5d_exposure ((4,5,6,4,3) layout), grid_large (N=100000, 12×12×8);
- CPU golden comparison (tests/ops/test_color_exposure.py): test_fwd_global_random, test_fwd_grid_random, test_fwd_grid_5d_exposure.

### 7.2 Boundary Cases

- test_fwd_boundary: injects 7 boundary coordinate groups ((0,0), (W−1,H−1), (W,H), positive / negative out-of-range, axis endpoints) × 7 boundary color groups (pure black, out-of-range 2.0, negative −2.0, grey-coefficient colors, etc.), covering all branches of clamp01, trunc, endpoint closure, and grey saturation;
- Tail and misalignment: N = 7 / 13 / 123 / 257; minimum grid 2×2×2; multi-batch B = 2 / 3.

### 7.3 Accuracy Cases

- All functional cases compare against golden: output / weight allclose(rtol=atol=1e-5), index torch.equal;
- Value-pinning cases: identity exposure → output bit-equal to colors; a designated element of $A$ set to k → the corresponding output channel equals $k \times$ the source channel element-wise.

### 7.4 Performance and Generalization Cases

- Shape sweep: N ∈ {1, 7, 13, 123, 257, 4096, 100000}, B ∈ {1, 2, 3}, grids 2×2×2 ~ 12×12×8, clamp ∈ {0, 1}, exposure 4d / 5d layouts;
- Performance baseline: to be added (currently focused on functional and accuracy verification).

## 8. Risks and Alternatives

| Risk | Impact | Mitigation / alternative |
| --- | --- | --- |
| Any grid dimension = 1 | Endpoint closure gives ix−1 = −1; the index falls back to the neighboring cell; unexpected results | Constrain every dimension ≥ 2 (inherited from the CUDA reference); to support it, add an ix > 0 guard in the closure branch and update golden accordingly |
| Grid bytes exceed the UB budget | taskNumPerLoop underflows; incorrect results | Constrain $G_X G_Y G_Z \le 3549$ (Ascend 950); larger grids require splitting the exposure field or block-wise staging |
| 32KB DCACHE reservation mismatched with hardware | Under-reserved → SIMT VF out-of-bounds access; over-reserved → UB waste | 32KB is a conservative empirical value; calibrate on hardware |
| NaN + clamp | max(NaN, 0) is platform-dependent; undefined results | Declared unsupported; supporting it requires NaN checks (performance cost) |
| fp32 multiply-by-reciprocal vs division | ~1 ulp difference from the CUDA reference; index may differ by 1 at boundaries | Golden replicates this implementation; cross-platform comparison must tolerate the difference |

## 9. Appendix

### 9.1 Glossary

| Term | Meaning |
| --- | --- |
| 12-parameter affine / $[A|b]$ | 3×4 row-augmented matrix: $A_{ij} = exposure[4i+j]$, $b_i = exposure[4i+3]$, out = Ac + b |
| Endpoint closure | When a grid coordinate lands on the last cell (ix = G−1), fall back to (ix, wx) ← (G−2, 1) so 8-corner interpolation stays in bounds |
| SIMT VF | The SIMT vector engine of Ascend 950 arch35 (launched via asc_vf_call, 1024 threads) |
| Long / short cores | Multi-core load-balancing: long cores carry 8 extra tasks; the rest are short cores |
| taskNumPerLoop | Number of tasks (gaussians) processed per tile on one core, derived from the UB budget |

### 9.2 References

- docs/en/kernels/aclnnColorExposureBwd_en.md: design document of the paired backward operator
- CANN Ascend C programming guide (TPipe / Tiling / DataCopyPad / asc_vf_call / Reg vector instructions)
- gsplat CUDA reference implementation

### 9.3 Changelog

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | Initial version |
