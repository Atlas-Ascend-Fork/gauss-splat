# ColorExposureBwd Design Document

[View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_bwd/op_kernel/arch35/color_exposure_bwd.cpp)

## 1. Basic Information

| Item | Content |
| --- | --- |
| Operator name | ColorExposureBwd |
| Repository / directory | gauss-splat / kernels/color_exposure_bwd |
| Operator type | Custom (backward gradients of the per-gaussian color affine transform) |
| aclnn interface name | aclnnColorExposureBwd |
| Document version | v1.1 |
| Author | - |
| Reviewer | - |
| Last updated | 2026-11-23 |

## 2. Background and Objectives

### 2.1 Business Scenario

The backward operator paired with ColorExposureFwd. In 3DGS training, exposure parameters are jointly optimized with gaussian attributes, and backpropagation requires two gradients:

- `vColorsOut`: the gradient of the loss w.r.t. the forward input `colors`, propagated to the upstream SH color computation;
- `vExposure`: the gradient of the loss w.r.t. `exposure` (whether to compute it is controlled by the attribute `calcExpGrad`), used to optimize the exposure parameters.

Inputs / outputs are organized as $[B, C, D, N]$ (same as the forward: $B$ batch, $C$ cameras, $D$ channels, $N$ per-camera gaussian capacity). The backward likewise receives the INT32 input `cnt` $[B, C]$: only the first `cnt[b, c]` valid gaussians of each camera produce gradients; invalid tail gaussians are neither written back to `vColorsOut` nor accumulated into `vExposure`, strictly aligning with the forward's filtering semantics (gaussians the forward did not compute naturally receive no gradient).

The forward already outputs the interpolation intermediates `index` / `weight` (written back only at valid gaussian positions), which the backward reuses directly; the coordinate-to-grid interpolation is not recomputed, and `means2d` no longer participates in the backward.

### 2.2 Reference Implementation

New.

### 2.3 Objectives

- Functional objectives:
  - Covered: `vColorsOut` in both global / grid modes; `vExposure` controlled by `calcExpGrad` (=1: computed and accumulated; =0: empty tensor output); `clamp` gradient masking (ReLU backward semantics); `cnt` filtering (only the first `cnt[b, c]` gaussians produce gradients; the remaining positions are neither written back nor accumulated); multiple cameras $C \ge 1$.
- Performance objective: memory-bound operator (per valid gaussian: read 6~12 floats, write 3 floats); GM bandwidth utilization close to same-platform copy-type operators (cnt filtering skips invalid tail gaussians; to be measured).
- Accuracy objective: `vColorsOut` within rtol = atol = 1e-5 of the CPU golden; `vExposure` rtol = 2e-4, atol = 1e-5 + 5e-8·total (total = total valid gaussians $\sum_{b,c} cnt[b,c]$; error comes from fp32 atomic accumulation order).
- Delivery scope: dtype FLOAT (`index` / `cnt` INT32), format ND, shape constraints identical to the forward; the grid byte budget is half of the forward's (double grid buffers, see 5.2); Ascend 950PR / 950DT only.

## 3. Operator Definition

### 3.1 Functional Description

Given the forward inputs `colors` / `exposure` / `cnt` and the upstream color gradient `vColorsIn` (in grid mode, additionally the forward outputs `index` / `weight`), computes:

- `vColorsOut` $[B, C, 3, N]$: `vColorsIn` propagated through the transpose of the affine matrix to obtain the gradient w.r.t. colors; with `clamp` = 1, upstream gradients of channels whose forward pre-clamp output was negative are masked to zero; only valid gaussians $n < cnt[b,c]$ are written back, the remaining positions keep the output tensor's initial value (the upper layer allocates with zeros, i.e. 0);
- `vExposure`: the gradient of the loss w.r.t. `exposure`, same shape as `exposure`, accumulated over all valid gaussians ($\sum cnt$); an empty tensor when `calcExpGrad` = 0.

### 3.2 Mathematical Formulation

Let $m = [A|b]$ be the per-gaussian affine parameters ($A_{ij} = m[4i+j]$, $b_i = m[4i+3]$):

- Global mode: $m$ is taken directly from `exposure`;
- Grid mode: reconstructed by trilinear interpolation from the forward outputs `index` / `weight` (`means2d` coordinates no longer used):

```text
m_k = Σ_{(cx,cy,cz)∈{0,1}^3} wx^cx · wy^cy · wz^cz · E[ix+cx, iy+cy, iz+cz, k]
```

For each $(b, c)$, **the following gradients are computed only for gaussians $n < cnt[b, c]$** (all other positions contribute to neither output):

**Clamp mask** (when clamp = 1; $pre = Ac + b$ is the pre-clamp forward output, recomputed):

```text
v'_i = (pre_i < 0) ? 0 : v_i        (ReLU backward semantics)
```

**v_colors** (transpose propagation, $dL/dc = A^T \cdot dL/dout$):

```text
vColorsOut_j = Σ_i A_ij · v'_i
```

**v_exposure** (per element, accumulated over all valid gaussians in the batch; $W$ is the gaussian's 8-corner trilinear weight, $W \equiv 1$ in global mode):

```text
g[..., 4i+j] += W · c_j · v'_i
g[..., 4i+3] += W · v'_i
```

Notes:

- `vExposure` has accumulation semantics: the kernel writes back with fp32 atomic adds, and the caller must zero `vExposure` beforehand (the upper layer already allocates with zeros, see 4.3);
- In grid mode, both $A$ and $W$ are reconstructed from `index` / `weight`; callers must pass the forward's outputs as-is (the forward writes back only within $[0, cnt)$; tail positions are 0, so the backward never reads stale data);
- Gaussians with $n \ge cnt[b,c]$ produce no gradient at all: the corresponding `vColorsOut` positions keep the initial value 0 and nothing is accumulated into `vExposure`.

### 3.3 Inputs, Outputs, and Attributes

| Name | Direction | Description | dtype | shape | format | Optional |
| --- | --- | --- | --- | --- | --- | --- |
| colors | input | Forward colors | FLOAT | [B, C, 3, N] | ND | No |
| vColorsIn | input | Upstream color gradient | FLOAT | [B, C, 3, N] | ND | No |
| exposure | input | Forward exposure parameters | FLOAT | global: [12] / (4,3) / (1,1,1,12); grid: (GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | No |
| cnt | input | Valid gaussian count per (b, c) (same as the forward) | INT32 | [B, C] | ND | No |
| index | input | Forward output: interpolation indices | INT32 | [B, C, 3, N] | ND | Required in grid mode; empty tensor in global mode |
| weight | input | Forward output: trilinear weights | FLOAT | [B, C, 3, N] | ND | Same as index |
| clamp | attr | Same as the forward | bool (0/1) | - | - | No |
| calcExpGrad | attr | Whether to compute and accumulate v_exposure | bool (0/1) | - | - | No |
| vColorsOut | output | Gradient w.r.t. colors (invalid gaussian positions not written back; must be pre-zeroed) | FLOAT | [B, C, 3, N] | ND | No |
| vExposure | output | Gradient w.r.t. exposure (atomic-accumulate semantics; must be pre-zeroed) | FLOAT | same as exposure; empty tensor when calcExpGrad = 0 | ND | No |

### 3.4 Support Matrix

| Dimension | Supported range | Unsupported items and reasons |
| --- | --- | --- |
| Product | Ascend 950PR / 950DT (arch35) | Other products unsupported: the operator is adapted to arch35 only (SIMT VF); no builds for other architectures |
| dtype | FLOAT (index / cnt INT32) | fp16 / bf16 / integer etc. unsupported: the gradient path is fp32 throughout |
| format | ND | Other formats unsupported: the kernel addresses a compact layout |
| shape | colors / vColorsIn / vColorsOut all [B,C,3,N] and consistent; index / weight [B,C,3,N]; cnt [B,C], $0 \le cnt[b,c] \le N$; exposure as in the forward; grid byte budget $2 \times G_X G_Y G_Z \times 48\,B$ bounded by UB (see 5.2, about half of the forward) | Empty tensors (0-sized dims) and 0-dim inputs unsupported; cnt > N treated as full, negative values undefined; grid dimension = 1 see Section 8 |
| Broadcast | Not supported | exposure is a batch / camera-shared parameter |

### 3.5 Boundary and Exception Cases

| Case | Expected behavior |
| --- | --- |
| cnt[b, c] = 0 | No gradient at all for this camera: break at the first taskLoop tile; vColorsOut of that (b, c) all stays 0; no contribution to vExposure |
| cnt[b, c] < N | Only the first cnt gaussians produce and write back gradients; the invalid tail gaussians are neither read nor written; when cnt is not 8-aligned or crosses taskLoop tiles, it is truncated by the mask (validTaskNum) |
| cnt[b, c] > N | Treated as full (= N); contract: callers do not pass this |
| Empty tensor (0-sized dim) | Required inputs / outputs non-empty; empty index / weight in global mode; vExposure empty tensor when calcExpGrad = 0 (kernel does not write back) |
| Scalar / 0-dim input | Not supported; parameter validation error (161002) |
| Non-contiguous input | Not supported: the kernel addresses a compact layout; the caller must ensure contiguity |
| NaN / Inf in inputs | Undefined: the clamp-mask comparison with NaN is platform-dependent; injection not recommended |
| vExposure not pre-zeroed | SetAtomicAdd accumulates onto stale values; wrong gradients. The caller must pre-zero (guaranteed by the upper layer) |
| vColorsOut not pre-zeroed | Invalid gaussian positions ($n \ge cnt$) keep stale values; the upper layer allocates with zeros; bare aclnn callers must check themselves |
| index / weight not matching forward outputs | Wrong gradients; they must be passed as-is from the forward (including the 0 padding outside $[0, cnt)$) |
| Invalid shape or dtype | Error 161001 (null pointer) / 161002 (dtype or shape validation failure), see 4.2 |
| Insufficient workspace | Not a user-side concern: managed by the aclnn framework per the GetWorkspaceSize return value (system workspace only) |

## 4. Interface Design

### 4.1 aclnn Interface and Call Flow

```text
aclnnColorExposureBwdGetWorkspaceSize(colors, vColorsIn, exposure, cnt, index, weight,
                                      clamp, calcExpGrad, vColorsOut, vExposure,
                                      workspaceSize, executor)
aclnnColorExposureBwd(workspace, workspaceSize, executor, stream)
```

- Workspace rule: only the system workspace (LibApiWorkSpaceSize) is used; no operator-custom workspace.
- Executor reuse rule: each `GetWorkspaceSize` performs parameter validation and creates an executor consumed by one `aclnnColorExposureBwd` execution (standard aclnn semantics).

### 4.2 Parameter Validation and Errors

| Check | Rule | Error code / log keyword |
| --- | --- | --- |
| Null pointer | All input / output tensor pointers and workspaceSize / executor non-null | 161001, "param ... is null" |
| dtype combination | colors / vColorsIn / exposure / weight / vColorsOut / vExposure are FLOAT, index / cnt are INT32 | 161002 |
| shape consistency | colors / vColorsIn / vColorsOut all [B,C,3,N] and consistent; cnt is [B,C]; exposure numel ≥ 12; in grid mode index / weight must be [B,C,3,N]; when calcExpGrad = 1, vExposure must match the exposure shape | 161002 |
| format | ND | 161002 |

(Validation is performed by the aclnn adapter generated at op-package build time, consistent with other operators in this repository.)

### 4.3 Upper-Layer Framework Mapping

Upper-layer chain: `gauss_splat.ops.color_exposure` (torch.autograd.Function.backward) → `gauss_splat/csrc/ColorExposureBwd.cpp` (EXEC_NPU_CMD direct call) → aclnnColorExposureBwd.

| Upper-layer behavior | This operator's counterpart |
| --- | --- |
| vColorsOut = at::zeros({B,C,3,N}) passed in | output vColorsOut (the kernel writes back only valid positions $[0, cnt)$; zeros allocation guarantees invalid positions are 0) |
| vExposure = calcExpGrad ? at::zeros_like(exposure) : at::empty(0) | output vExposure (zeros satisfies the pre-zeroing requirement; empty tensor corresponds to calcExpGrad = 0) |
| cnt passed as-is via the forward's ctx.save_for_backward | input cnt (filtering semantics identical to the forward) |
| upstream gradient made .contiguous() first | satisfies the contiguity constraint |
| backward returns (v_colorsOut, None, v_exposure or None, None, None, None, None, None) | no gradients for means2d / cnt / width / height / clamp / calc_exp_grad |

## 5. Implementation

### 5.1 Layered Structure

| Layer | Responsibility | Key files |
| --- | --- | --- |
| aclnn adapter (op_api) | Interface definition, parameter validation, workspace calculation | Generated at op-package build (no handwritten op_api in this repo) |
| op_host | Attribute parsing, mode determination, tiling | kernels/color_exposure_bwd/op_host/arch35/color_exposure_bwd.cpp, color_exposure_bwd_tiling.h |
| op_kernel | GM↔UB data movement and gradient computation | op_kernel/arch35/color_exposure_bwd.cpp, calc_color_exp_grad_vf.h |
| Framework binding | PyTorch / aten bridge | gauss_splat/csrc/ColorExposureBwd.cpp |

### 5.2 Tiling Strategy

The split dimension, long/short-core assignment, tail handling, and TilingKey = 1 are all identical to the forward (see aclnnColorExposureFwd_en.md 5.2); the differences are in the UB budget:

- Per-task queue budget bufferSize (floats / task): global mode $3 \times 3 \times 2 = 18$ (colors / vColorsIn / vColorsOut queues, 2×3 each); grid mode adds $6 \times 2 = 12$ (index / weight queues, 2×3 each) → 30;
- The exposure region is always budgeted for double grid buffers: `expSize = (2 × G_X·G_Y·G_Z × 12 + B·C) × 4 + 32KB` (exposure resident + vExposure accumulation buffer + cnt resident buffer $B \times C \times 4$ B; DCACHE_SIZE is unconditionally included in expSize, consistent with the kernel's unconditional double buffering; a conservative budget); `SetDynUBufSize(ubTotalSize − 32KB)` is unconditional;
- Grid size cap: the tiling hard check requires $G_X G_Y G_Z \times 12 \le 24576$ (MAX_GRID_DIMS), i.e. $G_X G_Y G_Z \le 2048$, maximum grid 16×16×8; exceeding it fails tiling directly; under the UB budget, $G_X G_Y G_Z \le 1774$ at B·C = 1 (shrinking slightly with B·C, about half of the forward's); verified up to 12×12×8 = 1152, plus a new 16×16×8 hard-check boundary case — note that on Ascend 950 (UB = 253952 B), 16×16×8 gives expSize ≈ 229KB, exceeding 0.8×UB ≈ 203KB, so taskNumPerLoop underflows to negative and the case cannot pass on real hardware under the current tiling formula (fix directions: recompute the budget as UB − 32KB − double grid buffers, skip the vExp buffer when calcExpGrad = 0, or split the exposure field); the case pins down this boundary;

### 5.3 Kernel Implementation

- Data movement path: exposure and cnt ($B \times C$ INT32) are each copied GM→UB once and stay resident (DataCopyPad + event sync); colors / vColorsIn (plus index / weight in grid mode) per tile GM→UB; vColorsOut per tile UB→GM; vExposure atomically accumulated from UB back to GM in whole blocks per tile;
- Pipeline: CopyIn → CalculateGrad → CopyOut advance serially with event synchronization; queues double-buffered; the vExposure write-back (MTE3) and the next tile's buffer zeroing (V) are serialized with set_flag / wait_flag to avoid read/write races; outer batch × camera × taskLoop triple loop;
- **cnt filtering implementation**: identical to the forward — for each (b, c), `cnt_ = cnt[b × C + c]` is read first; each tile is checked by ComputingTaskNum for `taskStart < cnt_`, and the check failing breaks out of the remaining tiles of this camera; when cnt falls inside a tile, the tile is truncated with `validTaskNum = cnt − taskStart` and forced onto the tail-mask path (DataCopyPad masked copies + VF computing with the validTaskNum mask); with cnt = 0, all tiles of the camera break immediately and neither output is affected;
- Compute primitives: `asc_vf_call` launches CalcGradUseGrid (grid) / CalcGradGlobal (global), dim3{1024, 1, 1}, one gaussian per thread:
  - Grid mode: reconstructs the 8-corner weights $W$ and the 12 parameters $m$ from index / weight (no coordinate remapping);
  - Clamp mask: recomputes $pre = Ac + b$ to determine the sign; negative channels' gradients masked to zero;
  - `vColorsOut_j = Σ_i A_ij · v'_i` (column extraction of $A$, i.e. $A^T v'$);
  - `vExposure`: grid mode accumulates atomically over 8 corners ($g[4i+j] += W \cdot c_j \cdot v'_i$, $g[4i+3] += W \cdot v'_i$); global mode is the special case $W \equiv 1$; skipped when calcExpGrad = 0;
- Buffer management: vExpBuf_ is always allocated (consistent with the tiling's double budget); each tile first zero-fills it with Duplicate, the VF accumulates atomically, and CopyOut writes the whole block back to GM with SetAtomicAdd;
- Loops and boundary handling: each tile gets taskNumPerCurLoop, isTailBlock, and validTaskNum (the valid task count after cnt truncation) from ComputingTaskNum; the VF masks the tail with validTaskNum, same as the forward.

### 5.4 Special-Case Handling

| Case | Solution |
| --- | --- |
| Small shape (N = 1) | totalTaskNum = 8, blockDim shrink + tail masking |
| Misaligned shape (N % 8 ≠ 0) | tailNum masking + DataCopyPad |
| cnt = 0 (camera-level full skip) | break at the first taskLoop tile; zero reads / writes / gradients for this camera |
| cnt not 8-aligned / falling inside a tile | validTaskNum truncation + forced tail-mask path |
| Large shape / multi-core | taskLoop tiling (verified up to N = 100000) |
| Multiple cameras (C > 1) | the camera dimension reuses the same queues / resident buffers; cnt read per camera |
| calcExpGrad = 0 | a flag skips vExposure computation and the atomic write-back; empty tensor output |
| Empty tensor | Not supported (see 3.5) |

## 6. Accuracy and Performance

### 6.1 Accuracy Standards and Verification

- Golden source: the CPU reference implementation inside tests/ops/test_color_exposure.py (`vExposure` accumulated in float64, everything else fp32; masked by cnt, invalid gaussians contribute nothing); the NPU-side C++ cases (tests/kernels/test_color_exposure_bwd.cpp) embed an equivalent golden (double accumulation, cnt mixed mode: 0 / 1 / N/2 / N);
- Comparison method: vColorsOut compared with torch.allclose after fp32 conversion; vExposure likewise but with scale-dependent tolerance;
- Error thresholds: vColorsOut rtol = atol = 1e-5; vExposure rtol = 2e-4, atol = 1e-5 + 5e-8·total (total = $\sum_{b,c} cnt[b,c]$, the total valid gaussians; with cnt filtering the error grows only with valid gaussians; the difference comes entirely from NPU fp32 atomic accumulation order vs the golden's high-precision accumulation);
- Mixed precision: none. vExposure accumulation is fp32 atomic add.

### 6.2 Performance Targets and Estimation

| Metric | Theoretical | Target | Measured | Achievement |
| --- | --- | --- | --- | --- |
| Latency | Memory-bound: per valid gaussian read 6 (global) / 12 (grid) floats, write 3 floats; invalid tail gaussians cost nothing; grid mode adds a $G_X G_Y G_Z \times 48\,B$ vExposure atomic write-back per tile | Close to same-platform copy-type operators | TBD | - |
| Bandwidth utilization | - | Same as above | TBD | - |

- Bottleneck analysis: GM data movement of the color path dominates; with calcExpGrad = 1, the whole-block vExposure atomic write-back is a secondary cost;
- Optimizations: whole-block atomic write-back (not per-element atomics), double-buffered queues, reuse of forward index / weight to avoid recomputing the interpolation, cnt filtering to skip invalid gaussians.

## 7. Test Design

### 7.1 Functional Cases

- NPU C++ cases (tests/kernels/test_color_exposure_bwd.cpp, 10 cases): global_clamp_calc (B=1, N=257, clamp + calcExpGrad), global_nocalc (B=2, calcExpGrad = 0 → vExposure empty), grid_clamp_calc (B=2, N=517, 5×5×6), grid_nocalc (N=4096, 3×5×4), grid_large (N=100000, 12×12×8, clamp + calcExpGrad), grid_max_16x16x8 (B=2, C=2, N=517, 16×16×8 hard-check cap), global_multi_camera (B=2, C=3), grid_cnt_mixed (B=2, C=3, N=517, 4×4×4, mixed cnt), grid_cnt_mixed_nocalc (B=2, C=2, N=123, 3×3×4, mixed cnt + no vExp), grid_cnt_mixed_large (B=1, C=4, N=100000, 12×12×8, mixed cnt); the mixed cnt mode takes 0 / 1 / N/2 / N by the (b, c) flat index; index / weight are generated per the forward semantics (valid only within $[0, cnt)$); the golden always uses asymmetric random affine matrices (A=rand), covering the transpose-propagation path;
- CPU golden comparison (tests/ops/test_color_exposure.py): test_bwd_global_random, test_bwd_grid_random, test_bwd_cnt_filter (cnt all-0 / mixed), test_bwd_grid_5d_exposure, test_bwd_grid_max_16x16x8 (16×16×8 hard-check cap; same caveat as above).

### 7.2 Boundary Cases

- test_bwd_clamp_mask: constructs gaussians whose forward output is negative and verifies the masked channels' gradients are zero; a two-value clamp golden comparison confirms the mask actually changes the gradients;
- test_bwd_large: N = 100000, verifying large-scale fp32 atomic accumulation accuracy;
- Tail and misalignment: N = 13 / 123 / 257 / 517; minimum grid 2×2×2; multi batch / camera B ∈ {1, 2}, C ∈ {1, 2, 3, 4};
- cnt boundaries: cnt = 0 (whole camera zero gradient), cnt = 1 (single gaussian), cnt = N/2 (in-tile truncation), cnt = N (full), and combinations thereof.

### 7.3 Accuracy Cases

- test_bwd_v_colors_transpose: basis-matrix pinning case; designated elements of $A$ in exposure are set to constants (e.g. $e_1 = k$, $e_8 = k$), verifying element-wise the channel mapping of `vColorsOut = A^T·v'` ($e_1 = k \Rightarrow vOut_g = k \cdot v_r$, $e_8 = k \Rightarrow vOut_r = k \cdot v_b$), with golden self-check + NPU comparison;
- test_autograd_vs_torch_grad: compared against the analytical backward via torch.autograd.grad on random inputs;
- cnt filtering pinning: cameras with cnt = 0 have all-zero vColorsOut and zero contribution to vExposure; with cnt < N, the [cnt, N) range of vColorsOut is all zero;
- Tolerances: vColorsOut 1e-5; vExposure widened with the total valid gaussian count per 6.1.

### 7.4 Performance and Generalization Cases

- Shape sweep as in the forward (N ∈ {1, 7, 13, 123, 257, 4096, 100000}, B, C, grids, clamp, exposure layouts), additionally covering the full clamp × calcExpGrad combination and cnt ∈ {full, mixed, all-0};
- End-to-end autograd: test_autograd_function, test_autograd_global, test_autograd_cnt_filter (end-to-end gradient correctness under cnt modes), test_autograd_grad_contract (gradient contract: autograd numerical consistency); wrapper validation test_wrapper_input_validation.

## 8. Risks and Alternatives

| Risk | Impact | Mitigation / alternative |
| --- | --- | --- |
| vExposure not pre-zeroed | Atomic accumulation onto stale values; wrong gradients | Semantic contract: caller zeroes it; the upper layer uses zeros_like; bare aclnn callers must check themselves |
| vColorsOut not pre-zeroed | Invalid gaussian positions ($n \ge cnt$) keep stale values | Semantic contract: the output tensor must be pre-zeroed; the upper layer already allocates with zeros; bare aclnn callers must check themselves |
| cnt semantic violation (cnt > N or negative) | cnt > N treated as full (correct results); negative values are undefined through unsigned comparison | Contract $0 \le cnt[b,c] \le N$ (guaranteed by the upstream projection / filtering, and consistent with the forward); add validation in the aclnn adapter if defense is needed |
| fp32 atomic accumulation accuracy | vExposure error grows with the total valid gaussian count | Tolerance widened with scale (atol = 1e-5 + 5e-8·total); higher accuracy requires blocked partial sums + reduction |
| Grid budget half of the forward (double grid buffers) | 16×16×8 = 2048 cells, allowed by the hard check | A grid_max_16x16x8 boundary case has been added to pin down this boundary; |
| index / weight tampered or self-computed | Wrong gradients | Contract: pass forward outputs as-is; fixed by documentation and test cases |
| Grid dimension = 1 / NaN + clamp | Same as the forward | See aclnnColorExposureFwd_en.md Section 8 |

## 9. Appendix

### 9.1 Glossary

| Term | Meaning |
| --- | --- |
| 12-parameter affine / [$[A\|b]$ | 3×4 row-augmented matrix: $A_{ij} = m[4i+j]$, $b_i = m[4i+3]$ |
| cnt filtering | Only the first cnt[b, c] gaussians (compact layout) produce gradients; the remaining positions are neither written back nor accumulated, aligned with the forward's filtering semantics |
| Clamp mask | With clamp = 1, upstream gradients of forward-negative channels are masked to zero (ReLU backward semantics) |
| Atomic accumulation / SetAtomicAdd | The fp32 atomic-add implementation of the cross-gaussian vExposure accumulation; ordering undetermined |
| SIMT VF | The SIMT vector engine of Ascend 950 arch35 (launched via asc_vf_call, 1024 threads) |
| Long / short cores / taskNumPerLoop | Multi-core load balancing and per-core tile size, same as the forward |
| validTaskNum | Number of tasks actually computed inside a tile: the tail subtracts tailNum; under cnt truncation it is the remaining valid count inside the tile |

### 9.2 References

- docs/en/kernels/aclnnColorExposureFwd_en.md: design document of the paired forward operator
- CANN Ascend C programming guide (TPipe / Tiling / asc_vf_call / SetAtomicAdd)
- gsplat CUDA reference implementation

### 9.3 Changelog

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | Initial version (vColorsOut propagates as $A^T \cdot v'$ per the mathematical definition) |
| v1.1 | 2026-11-23 | - | Inputs / outputs restructured from [B, C, N] (batch×camera fused) to [B, C, D, N] (explicit camera dimension); added the INT32 input cnt (valid gaussian count per (b, c)), with the kernel skipping invalid tail gaussians per cnt (no vColorsOut write-back, no vExposure accumulation); index / weight shapes changed accordingly to [B, C, 3, N]; updated tiling (cnt resident buffer, triple loop, grid hard check) and test design; vExposure tolerance base changed to the total valid gaussians Σcnt; added a maximum-grid 16×16×8 (hard-check boundary) case — on A5 the current tiling formula underflows the UB budget, and passing on real hardware requires a kernel-side fix |
