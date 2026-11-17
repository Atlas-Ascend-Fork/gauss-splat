# ColorExposureBwd Design Document

[View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_bwd/op_kernel/arch35/color_exposure_bwd.cpp)

## 1. Basic Information

| Item | Content |
| --- | --- |
| Operator name | ColorExposureBwd |
| Repository / directory | gauss-splat / kernels/color_exposure_bwd |
| Operator type | Custom (backward gradients of the per-gaussian color affine transform) |
| aclnn interface name | aclnnColorExposureBwd |
| Document version | v1.0 |
| Author | - |
| Reviewer | - |
| Last updated | 2026-11-16 |

## 2. Background and Objectives

### 2.1 Business Scenario

The backward operator paired with ColorExposureFwd. In 3DGS training, exposure parameters are jointly optimized with gaussian attributes, and backpropagation requires two gradients:

- `vColorsOut`: the gradient of the loss w.r.t. the forward input `colors`, propagated to the upstream SH color computation;
- `vExposure`: the gradient of the loss w.r.t. `exposure` (whether to compute it is controlled by the attribute `calcExpGrad`), used to optimize the exposure parameters.

The forward already outputs the interpolation intermediates `index` / `weight`, which the backward reuses directly; the coordinate-to-grid interpolation is not recomputed, and `means2d` no longer participates in the backward.

### 2.2 Reference Implementation

The operator matches the mathematical semantics of the `color_exposure` backward in the gsplat CUDA reference implementation. Differences of this implementation:

- The CUDA reference recomputes the interpolation in the backward; this implementation reuses the forward outputs `index` / `weight` directly, saving one coordinate-to-grid mapping;
- Hardware path: based on Ascend 950 (arch35) AIV vector cores + SIMT VF; the entire gradient computation runs on the SIMT VF (1024 threads, one gaussian per thread);
- The cross-gaussian accumulation of `vExposure` uses fp32 atomic adds (SetAtomicAdd), which introduces ordering-dependent error versus the high-precision golden accumulation; the tolerance widens with the total gaussian count (see 6.1).

### 2.3 Objectives

- Functional objectives:
  - Covered: `vColorsOut` in both global / grid modes; `vExposure` controlled by `calcExpGrad` (=1: computed and accumulated; =0: empty tensor output); `clamp` gradient masking (ReLU backward semantics).
- Performance objective: memory-bound operator (per gaussian: read 6~12 floats, write 3 floats); GM bandwidth utilization close to same-platform copy-type operators (to be measured).
- Accuracy objective: `vColorsOut` within rtol = atol = 1e-5 of the CPU golden; `vExposure` rtol = 2e-4, atol = 1e-5 + 5e-8·N (N = total gaussians; error comes from fp32 atomic accumulation order).
- Delivery scope: dtype FLOAT (`index` INT32), format ND, shape constraints identical to the forward; the grid byte budget is half of the forward's (double grid buffers, see 5.2); Ascend 950PR / 950DT only.

## 3. Operator Definition

### 3.1 Functional Description

Given the forward inputs `colors` / `exposure` and the upstream color gradient `vColorsIn` (in grid mode, additionally the forward outputs `index` / `weight`), computes:

- `vColorsOut` $[B, 3, N]$: `vColorsIn` propagated through the transpose of the affine matrix to obtain the gradient w.r.t. colors; with `clamp` = 1, upstream gradients of channels whose forward pre-clamp output was negative are masked to zero;
- `vExposure`: the gradient of the loss w.r.t. `exposure`, same shape as `exposure`, accumulated over all gaussians in the batch; an empty tensor when `calcExpGrad` = 0.

### 3.2 Mathematical Formulation

Let $m = [A|b]$ be the per-gaussian affine parameters ($A_{ij} = m[4i+j]$, $b_i = m[4i+3]$):

- Global mode: $m$ is taken directly from `exposure`;
- Grid mode: reconstructed by trilinear interpolation from the forward outputs `index` / `weight` (`means2d` coordinates no longer used):

```text
m_k = Σ_{(cx,cy,cz)∈{0,1}^3} wx^cx · wy^cy · wz^cz · E[ix+cx, iy+cy, iz+cz, k]
```

**Clamp mask** (when clamp = 1; $pre = Ac + b$ is the pre-clamp forward output):

```text
v'_i = (pre_i < 0) ? 0 : v_i        (ReLU backward semantics)
```

**v_colors** (transpose propagation, $dL/dc = A^T \cdot dL/dout$):

```text
vColorsOut_j = Σ_i A_ij · v'_i
```

**v_exposure** (per element, accumulated over all gaussians in the batch; $W$ is the gaussian's 8-corner trilinear weight, $W \equiv 1$ in global mode):

```text
g[..., 4i+j] += W · c_j · v'_i
g[..., 4i+3] += W · v'_i
```

Notes:

- `vExposure` has accumulation semantics: the kernel writes back with fp32 atomic adds, and the caller must zero `vExposure` beforehand (the upper layer already allocates with zeros, see 4.3);
- In grid mode, both $A$ and $W$ are reconstructed from `index` / `weight`; callers must pass the forward's outputs as-is.

### 3.3 Inputs, Outputs, and Attributes

| Name | Direction | Description | dtype | shape | format | Optional |
| --- | --- | --- | --- | --- | --- | --- |
| colors | input | Forward colors | FLOAT | [B, 3, N] | ND | No |
| vColorsIn | input | Upstream color gradient | FLOAT | [B, 3, N] | ND | No |
| exposure | input | Forward exposure parameters | FLOAT | global: [12] / (4,3) / (1,1,1,12); grid: (GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | No |
| index | input | Forward output: interpolation indices | INT32 | [B, 3, N] | ND | Required in grid mode; empty tensor in global mode |
| weight | input | Forward output: trilinear weights | FLOAT | [B, 3, N] | ND | Same as index |
| clamp | attr | Same as the forward | int64 (0/1) | - | - | No |
| calcExpGrad | attr | Whether to compute and accumulate v_exposure | int64 (0/1) | - | - | No |
| vColorsOut | output | Gradient w.r.t. colors | FLOAT | [B, 3, N] | ND | No |
| vExposure | output | Gradient w.r.t. exposure (atomic-accumulate semantics; must be pre-zeroed) | FLOAT | same as exposure; empty tensor when calcExpGrad = 0 | ND | No |

### 3.4 Support Matrix

| Dimension | Supported range | Unsupported items and reasons |
| --- | --- | --- |
| Product | Ascend 950PR / 950DT (arch35) | Other products unsupported: the operator is adapted to arch35 only (SIMT VF); no builds for other architectures |
| dtype | FLOAT (index INT32) | fp16 / bf16 / integer etc. unsupported: the gradient path is fp32 throughout |
| format | ND | Other formats unsupported: the kernel addresses a compact layout |
| shape | colors / vColorsIn / vColorsOut all [B,3,N] and consistent; exposure as in the forward; grid byte budget $2 \times G_X G_Y G_Z \times 48\,B$ bounded by UB (see 5.2, about half of the forward) | Empty tensors (0-sized dims) and 0-dim inputs unsupported; grid dimension = 1 see Section 8 |
| Broadcast | Not supported | exposure is a batch-shared parameter |

### 3.5 Boundary and Exception Cases

| Case | Expected behavior |
| --- | --- |
| Empty tensor (0-sized dim) | Required inputs / outputs non-empty; empty index / weight in global mode; vExposure empty tensor when calcExpGrad = 0 (kernel does not write back) |
| Scalar / 0-dim input | Not supported; parameter validation error (161002) |
| Non-contiguous input | Not supported: the kernel addresses a compact layout; the caller must ensure contiguity |
| NaN / Inf in inputs | Undefined: the clamp-mask comparison with NaN is platform-dependent; injection not recommended |
| vExposure not pre-zeroed | SetAtomicAdd accumulates onto stale values; wrong gradients. The caller must pre-zero (guaranteed by the upper layer) |
| index / weight not matching forward outputs | Wrong gradients; they must be passed as-is from the forward |
| Invalid shape or dtype | Error 161001 (null pointer) / 161002 (dtype or shape validation failure), see 4.2 |
| Insufficient workspace | Not a user-side concern: managed by the aclnn framework per the GetWorkspaceSize return value (system workspace only) |

## 4. Interface Design

### 4.1 aclnn Interface and Call Flow

```text
aclnnColorExposureBwdGetWorkspaceSize(colors, vColorsIn, exposure, index, weight,
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
| dtype combination | colors / vColorsIn / exposure / weight / vColorsOut / vExposure are FLOAT, index is INT32 | 161002 |
| shape consistency | colors / vColorsIn / vColorsOut all [B,3,N] and consistent; exposure numel ≥ 12; in grid mode index / weight must be [B,3,N]; when calcExpGrad = 1, vExposure must match the exposure shape | 161002 |
| format | ND | 161002 |

(Validation is performed by the aclnn adapter generated at op-package build time, consistent with other operators in this repository.)

### 4.3 Upper-Layer Framework Mapping

Upper-layer chain: `gauss_splat.ops.color_exposure` (torch.autograd.Function.backward) → `gauss_splat/csrc/ColorExposureBwd.cpp` (EXEC_NPU_CMD direct call) → aclnnColorExposureBwd.

| Upper-layer behavior | This operator's counterpart |
| --- | --- |
| vColorsOut = at::zeros({B,3,N}) passed in | output vColorsOut (overwritten by the kernel) |
| vExposure = calcExpGrad ? at::zeros_like(exposure) : at::empty(0) | output vExposure (zeros satisfies the pre-zeroing requirement; empty tensor corresponds to calcExpGrad = 0) |
| upstream gradient made .contiguous() first | satisfies the contiguity constraint |
| backward returns (v_colorsOut, None, v_exposure or None, ...) | no gradients for means2d / width / height |

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
- The exposure region is always budgeted for double grid buffers: `expSize = 2 × G_X·G_Y·G_Z × 12 × 4 + 32KB` (exposure resident + vExposure accumulation buffer, consistent with the kernel's unconditional double buffering; a conservative budget); `SetDynUBufSize(ubTotalSize − 32KB)` is unconditional;
- Consequently the grid size cap is about half of the forward's: on Ascend 950 (UB = 253952 B), $G_X G_Y G_Z \le 1779$; verified up to 12×12×8 = 1152; 16×16×8 = 2048 exceeds the budget and is unsupported.

### 5.3 Kernel Implementation

- Data movement path: exposure GM→UB once, resident; colors / vColorsIn (plus index / weight in grid mode) per tile GM→UB; vColorsOut per tile UB→GM; vExposure atomically accumulated from UB back to GM in whole blocks per tile;
- Pipeline: CopyIn → CalculateGrad → CopyOut advance serially with event synchronization; queues double-buffered; the vExposure write-back (MTE3) and the next tile's buffer zeroing (V) are serialized with set_flag / wait_flag to avoid read/write races;
- Compute primitives: `asc_vf_call` launches CalcGradUseGrid (grid) / CalcGradGlobal (global), dim3{1024, 1, 1}, one gaussian per thread:
  - Grid mode: reconstructs the 8-corner weights $W$ and the 12 parameters $m$ from index / weight (no coordinate remapping);
  - Clamp mask: recomputes $pre = Ac + b$ to determine the sign; negative channels' gradients masked to zero;
  - `vColorsOut_j = Σ_i A_ij · v'_i` (column extraction of $A$, i.e. $A^T v'$);
  - `vExposure`: grid mode accumulates atomically over 8 corners ($g[4i+j] += W \cdot c_j \cdot v'_i$, $g[4i+3] += W \cdot v'_i$); global mode is the special case $W \equiv 1$; skipped when calcExpGrad = 0;
- Buffer management: vExpBuf_ is always allocated (consistent with the tiling's double budget); each tile first zero-fills it with Duplicate, the VF accumulates atomically, and CopyOut writes the whole block back to GM with SetAtomicAdd;
- Loops and boundary handling: outer batch × taskLoop double loop; tail masked with validTaskNum, same as the forward.

### 5.4 Special-Case Handling

| Case | Solution |
| --- | --- |
| Small shape (N = 1) | totalTaskNum = 8, blockDim shrink + tail masking |
| Misaligned shape (N % 8 ≠ 0) | tailNum masking + DataCopyPad |
| Large shape / multi-core | taskLoop tiling (verified up to N = 100000) |
| calcExpGrad = 0 | a flag skips vExposure computation and the atomic write-back; empty tensor output |
| Empty tensor | Not supported (see 3.5) |

## 6. Accuracy and Performance

### 6.1 Accuracy Standards and Verification

- Golden source: the CPU reference implementation inside tests/ops/test_color_exposure.py (`vExposure` accumulated in float64, everything else fp32); the NPU-side C++ cases (tests/kernels/test_color_exposure_bwd.cpp) embed an equivalent golden (double accumulation);
- Comparison method: vColorsOut compared with torch.allclose after fp32 conversion; vExposure likewise but with scale-dependent tolerance;
- Error thresholds: vColorsOut rtol = atol = 1e-5; vExposure rtol = 2e-4, atol = 1e-5 + 5e-8·N (N = total gaussians; the difference comes entirely from NPU fp32 atomic accumulation order vs the golden's high-precision accumulation);
- Mixed precision: none. vExposure accumulation is fp32 atomic add.

### 6.2 Performance Targets and Estimation

| Metric | Theoretical | Target | Measured | Achievement |
| --- | --- | --- | --- | --- |
| Latency | Memory-bound: per gaussian read 6 (global) / 12 (grid) floats, write 3 floats; grid mode adds a $G_X G_Y G_Z \times 48\,B$ vExposure atomic write-back per tile | Close to same-platform copy-type operators | TBD | - |
| Bandwidth utilization | - | Same as above | TBD | - |

- Bottleneck analysis: GM data movement of the color path dominates; with calcExpGrad = 1, the whole-block vExposure atomic write-back is a secondary cost;
- Optimizations: whole-block atomic write-back (not per-element atomics), double-buffered queues, reuse of forward index / weight to avoid recomputing the interpolation.

## 7. Test Design

### 7.1 Functional Cases

- NPU C++ cases (tests/kernels/test_color_exposure_bwd.cpp, 5 cases): global_clamp_calc (clamp + calcExpGrad), global_nocalc (calcExpGrad = 0 → vExposure empty), grid_clamp_calc (5×5×6), grid_nocalc (3×5×4), grid_large (N=100000, 12×12×8, clamp + calcExpGrad); the golden always uses asymmetric random affine matrices (A=rand), covering the transpose-propagation path;
- CPU golden comparison (tests/ops/test_color_exposure.py): test_bwd_global_random, test_bwd_grid_random, test_bwd_grid_5d_exposure.

### 7.2 Boundary Cases

- test_bwd_clamp_mask: constructs gaussians whose forward output is negative and verifies the masked channels' gradients are zero; a two-value clamp golden comparison confirms the mask actually changes the gradients;
- test_bwd_large: N = 100000, verifying large-scale fp32 atomic accumulation accuracy;
- Tail and misalignment: N = 13 / 257 / 517; minimum grid 2×2×2; multi-batch B = 2.

### 7.3 Accuracy Cases

- test_bwd_v_colors_transpose: basis-matrix pinning case; designated elements of $A$ in exposure are set to constants (e.g. $e_1 = k$, $e_8 = k$), verifying element-wise the channel mapping of `vColorsOut = A^T·v'` ($e_1 = k \Rightarrow vOut_g = k \cdot v_r$, $e_8 = k \Rightarrow vOut_r = k \cdot v_b$), with golden self-check + NPU comparison;
- test_autograd_vs_torch_grad: compared against the analytical backward via torch.autograd.grad on random inputs;
- Tolerances: vColorsOut 1e-5; vExposure widened with scale per 6.1.

### 7.4 Performance and Generalization Cases

- Shape sweep as in the forward, additionally covering the full clamp × calcExpGrad combination;
- End-to-end autograd: test_autograd_function, test_autograd_global, test_autograd_grad_contract (gradient contract: autograd numerical consistency); wrapper validation test_wrapper_input_validation.

## 8. Risks and Alternatives

| Risk | Impact | Mitigation / alternative |
| --- | --- | --- |
| vExposure not pre-zeroed | Atomic accumulation onto stale values; wrong gradients | Semantic contract: caller zeroes it; the upper layer uses zeros_like; bare aclnn callers must check themselves |
| fp32 atomic accumulation accuracy | vExposure error grows with total gaussians N | Tolerance widened with scale (atol = 1e-5 + 5e-8·N); higher accuracy requires blocked partial sums + reduction |
| Grid budget half of the forward | Large grids (e.g. 16×16×8) break tiling | Constrain $G_X G_Y G_Z \le 1779$ (Ascend 950); larger grids require splitting the exposure field, or not allocating the vExp buffer when calcExpGrad = 0 |
| index / weight tampered or self-computed | Wrong gradients | Contract: pass forward outputs as-is; fixed by documentation and test cases |
| Grid dimension = 1 / NaN + clamp | Same as the forward | See aclnnColorExposureFwd_en.md Section 8 |

## 9. Appendix

### 9.1 Glossary

| Term | Meaning |
| --- | --- |
| 12-parameter affine / $[A|b]$ | 3×4 row-augmented matrix: $A_{ij} = m[4i+j]$, $b_i = m[4i+3]$ |
| Clamp mask | With clamp = 1, upstream gradients of forward-negative channels are masked to zero (ReLU backward semantics) |
| Atomic accumulation / SetAtomicAdd | The fp32 atomic-add implementation of the cross-gaussian vExposure accumulation; ordering undetermined |
| SIMT VF | The SIMT vector engine of Ascend 950 arch35 (launched via asc_vf_call, 1024 threads) |
| Long / short cores / taskNumPerLoop | Multi-core load balancing and per-core tile size, same as the forward |

### 9.2 References

- docs/en/kernels/aclnnColorExposureFwd_en.md: design document of the paired forward operator
- CANN Ascend C programming guide (TPipe / Tiling / asc_vf_call / SetAtomicAdd)
- gsplat CUDA reference implementation

### 9.3 Changelog

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | Initial version (vColorsOut propagates as $A^T \cdot v'$ per the mathematical definition) |
