# ColorExposureBwd 设计说明

[📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_bwd/op_kernel/arch35/color_exposure_bwd.cpp)

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | ColorExposureBwd |
| 所属仓 / 目录 | gauss-splat / kernels/color_exposure_bwd |
| 算子类型 | 自定义（逐高斯颜色仿射变换的反向梯度） |
| aclnn 接口名 | aclnnColorExposureBwd |
| 文档版本 | v1.1 |
| 作者 | - |
| 评审人 | - |
| 最后更新 | 2026-11-23 |

## 2. 需求背景与目标

### 2.1 业务场景

与 ColorExposureFwd 配套的反向算子。3DGS 训练中曝光参数与高斯属性联合优化，反向传播需要两条梯度：

- `vColorsOut`：损失对前向输入 `colors` 的梯度，回传给上游 SH 颜色计算；
- `vExposure`：损失对 `exposure` 的梯度（由属性 `calcExpGrad` 控制是否计算），用于曝光参数的优化。

输入 / 输出按 $[B, C, D, N]$ 组织（同前向：$B$ 为 batch，$C$ 为相机数，$D$ 为通道，$N$ 为每相机高斯容量）。反向同样接收 INT32 输入 `cnt` $[B, C]$：仅每相机前 `cnt[b, c]` 个有效高斯产生梯度，无效尾部高斯既不写回 `vColorsOut`，也不累加进 `vExposure`，与前向的过滤语义严格对齐（前向未计算的高斯，反向自然无梯度）。

前向已显式输出插值中间量 `index` / `weight`（仅在有效高素位置写回），反向直接复用，无需重算坐标→网格插值，`means2d` 不再参与反向。

### 2.2 对标算子

新增算子。

### 2.3 目标

- 功能目标：
  - 覆盖：全局 / 网格两种模式的 `vColorsOut`；`vExposure` 由 `calcExpGrad` 控制（=1 输出并累加，=0 输出空 tensor）；`clamp` 梯度掩码（ReLU 反传语义）；`cnt` 过滤（仅前 `cnt[b, c]` 个高斯产生梯度，其余位置不写回、不累加）；多相机维 $C \ge 1$。
- 性能目标：访存受限型算子（每有效高斯读 6~12 float、写 3 float），GM 带宽利用率贴近同平台拷贝类算子水平（cnt 过滤跳过无效尾部高斯；实测待标定）。
- 精度目标：`vColorsOut` 对 CPU golden 的误差 rtol = atol = 1e-5；`vExposure` rtol = 2e-4、atol = 1e-5 + 5e-8·total（total = 有效高斯总数 $\sum_{b,c} cnt[b,c]$，误差来自 fp32 原子累加次序）。
- 交付范围：dtype FLOAT（`index` / `cnt` INT32）、format ND、shape 约束同前向；网格字节预算为前向的一半（双份网格缓冲，见 5.2）；仅支持 Ascend 950PR / 950DT。

## 3. 算子定义

### 3.1 功能描述

给定前向输入 `colors` / `exposure` / `cnt`、上游颜色梯度 `vColorsIn`（网格模式另需前向输出的 `index` / `weight`），计算：

- `vColorsOut` $[B, C, 3, N]$：`vColorsIn` 经仿射矩阵转置反传并对 colors 求梯度，`clamp` = 1 时前向负输出通道的上游梯度被掩零；仅有效高素 $n < cnt[b,c]$ 写回，其余位置保持输出 tensor 初值（上层封装以 zeros 分配，即 0）；
- `vExposure`：损失对 `exposure` 的梯度，形状同 `exposure`，为全部有效高斯（$\sum cnt$）的累加和；`calcExpGrad` = 0 时输出空 tensor。

### 3.2 数学表达式

记逐高斯仿射参数 $m = [A|b]$（$A_{ij} = m[4i+j]$，$b_i = m[4i+3]$）：

- 全局模式：$m$ 直接取 `exposure`；
- 网格模式：由前向输出的 `index` / `weight` 重建三线性插值（不再使用 `means2d` 坐标）：

```text
m_k = Σ_{(cx,cy,cz)∈{0,1}^3} wx^cx · wy^cy · wz^cz · E[ix+cx, iy+cy, iz+cz, k]
```

对每个 $(b, c)$，**仅对 $n < cnt[b, c]$ 的高斯计算以下梯度**（其余位置对两个输出均无贡献）：

**clamp 掩码**（clamp = 1 时，$pre = Ac + b$ 为前向截断前输出，需重算）：

```text
v'_i = (pre_i < 0) ? 0 : v_i        （ReLU 反传语义）
```

**v_colors**（$A$ 的转置反传，$dL/dc = A^T \cdot dL/dout$）：

```text
vColorsOut_j = Σ_i A_ij · v'_i
```

**v_exposure**（逐元素，跨批内全部有效高斯累加；$W$ 为该高斯的 8 角三线权重，全局模式 $W \equiv 1$）：

```text
g[..., 4i+j] += W · c_j · v'_i
g[..., 4i+3] += W · v'_i
```

补充说明：

- `vExposure` 为累加语义：kernel 以 fp32 原子加回写，调用前必须将 `vExposure` 清零（上层封装已用 zeros 分配，见 4.3）；
- 网格模式的 $A$、$W$ 均由 `index` / `weight` 重建，调用方必须传入前向的原样输出（前向仅在 $[0, cnt)$ 位置写回，尾部位置为 0，反向不会读到脏数据）；
- $n \ge cnt[b,c]$ 的高斯不产生任何梯度：`vColorsOut` 对应位置保持初值 0，也不累加进 `vExposure`。

### 3.3 输入、输出与属性

| 名称 | 方向 | 含义 | dtype | shape | format | 是否可选 |
| --- | --- | --- | --- | --- | --- | --- |
| colors | 输入 | 前向颜色 | FLOAT | [B, C, 3, N] | ND | 否 |
| vColorsIn | 输入 | 上游颜色梯度 | FLOAT | [B, C, 3, N] | ND | 否 |
| exposure | 输入 | 前向曝光参数 | FLOAT | 全局：[12] / (4,3) / (1,1,1,12)；网格：(GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | 否 |
| cnt | 输入 | 逐 (b, c) 有效高斯计数（与前向一致） | INT32 | [B, C] | ND | 否 |
| index | 输入 | 前向输出的插值下标 | INT32 | [B, C, 3, N] | ND | 网格模式必选；全局模式传空 tensor |
| weight | 输入 | 前向输出的三线权重 | FLOAT | [B, C, 3, N] | ND | 同 index |
| clamp | 属性 | 与前向一致 | bool（0/1） | - | - | 否 |
| calcExpGrad | 属性 | 是否计算并累加 v_exposure | bool（0/1） | - | - | 否 |
| vColorsOut | 输出 | 对 colors 的梯度（无效高斯位置不写回，需预清零） | FLOAT | [B, C, 3, N] | ND | 否 |
| vExposure | 输出 | 对 exposure 的梯度（原子累加语义，需预清零） | FLOAT | 同 exposure；calcExpGrad = 0 时为空 tensor | ND | 否 |

### 3.4 支持矩阵

| 维度 | 支持范围 | 不支持项及原因 |
| --- | --- | --- |
| 产品 | Ascend 950PR / 950DT（arch35） | 其他产品不支持：算子仅适配 arch35（SIMT VF），未编译其他架构版本 |
| dtype | FLOAT（index / cnt 为 INT32） | fp16 / bf16 / 整型等不支持：梯度通路全程 fp32 |
| format | ND | 其他 format 不支持：kernel 按紧凑布局直接寻址 |
| shape | colors / vColorsIn / vColorsOut 均为 [B,C,3,N] 且一致；index / weight 为 [B,C,3,N]；cnt 为 [B,C]，$0 \le cnt[b,c] \le N$；exposure 同前向；网格字节预算 $2 \times G_X G_Y G_Z \times 48\,B$ 受 UB 约束（见 5.2，约为前向一半） | 空 tensor（shape 含 0）、0 维输入不支持；cnt > N 按全量处理、负值未定义；网格维 = 1 见第 8 节 |
| 广播 | 不支持 | exposure 为全 batch / 相机共享参数 |

### 3.5 边界与异常场景

| 场景 | 预期行为 |
| --- | --- |
| cnt[b, c] = 0 | 该相机无任何梯度：taskLoop 首块即 break，vColorsOut 对应 (b, c) 全部保持 0，对 vExposure 无贡献 |
| cnt[b, c] < N | 仅前 cnt 个高斯产生并写回梯度；尾部无效高斯不读不写；cnt 非 8 对齐 / 跨 taskLoop 分块时按掩码截断（validTaskNum） |
| cnt[b, c] > N | 按全量（= N）处理；约定调用方不传入 |
| 空 tensor（shape 含 0） | 必选输入 / 输出非空；全局模式 index / weight 传空 tensor；calcExpGrad = 0 时 vExposure 传空 tensor（kernel 不回写） |
| 标量 / 0 维输入 | 不支持，参数校验报错（161002） |
| 非连续输入 | 不支持：kernel 按紧凑布局寻址，需调用方保证连续 |
| 输入含 NaN / Inf | 未定义：clamp 掩码的比较运算对 NaN 行为平台相关，不建议注入 |
| vExposure 未预清零 | SetAtomicAdd 累加到旧值，梯度错误；调用方必须预清零（上层封装已保证） |
| vColorsOut 未预清零 | 无效高斯位置（$n \ge cnt$）残留旧值；上层封装已用 zeros 分配，aclnn 裸调用者需自查 |
| index / weight 与前向输出不匹配 | 梯度错误；约定必须原样传入前向输出（含 $[0, cnt)$ 之外的 0 填充） |
| shape 或 dtype 不合法 | 报错 161001（空指针）/ 161002（dtype、shape 校验失败），见 4.2 |
| workspace 不足 | 不涉及用户侧 workspace：由 aclnn 框架按 GetWorkspaceSize 返回值管理（仅系统 workspace） |

## 4. 接口设计

### 4.1 aclnn 接口与调用流程

```text
aclnnColorExposureBwdGetWorkspaceSize(colors, vColorsIn, exposure, cnt, index, weight,
                                      clamp, calcExpGrad, vColorsOut, vExposure,
                                      workspaceSize, executor)
aclnnColorExposureBwd(workspace, workspaceSize, executor, stream)
```

- workspace 计算规则：仅使用系统 workspace（LibApiWorkSpaceSize），无算子自定义 workspace。
- 执行器复用规则：每次 `GetWorkspaceSize` 完成参数校验并创建执行器，`aclnnColorExposureBwd` 执行一次（标准 aclnn 语义）。

### 4.2 参数校验与报错

| 校验项 | 规则 | 报错码 / 日志关键字 |
| --- | --- | --- |
| 空指针 | 所有输入 / 输出 tensor 指针及 workspaceSize / executor 非空 | 161001，"param ... is null" |
| dtype 组合 | colors / vColorsIn / exposure / weight / vColorsOut / vExposure 为 FLOAT，index / cnt 为 INT32 | 161002 |
| shape 一致性 | colors / vColorsIn / vColorsOut 均为 [B,C,3,N] 且一致；cnt 为 [B,C]；exposure numel ≥ 12；网格模式 index / weight 必须为 [B,C,3,N]；calcExpGrad = 1 时 vExposure 形状必须与 exposure 一致 | 161002 |
| format | ND | 161002 |

（校验逻辑由算子包构建生成的 aclnn 适配层完成，与仓内其他算子一致。）

### 4.3 上层框架映射

上层链路：`gauss_splat.ops.color_exposure`（torch.autograd.Function.backward）→ `gauss_splat/csrc/ColorExposureBwd.cpp`（EXEC_NPU_CMD 直调）→ aclnnColorExposureBwd。

| 上层行为 | 本算子对应 |
| --- | --- |
| vColorsOut = at::zeros({B,C,3,N}) 后传入 | 输出 vColorsOut（kernel 仅写回 $[0, cnt)$ 有效位置，zeros 分配保证无效位置为 0） |
| vExposure = calcExpGrad ? at::zeros_like(exposure) : at::empty(0) | 输出 vExposure（zeros 满足预清零要求；空 tensor 对应 calcExpGrad = 0） |
| cnt 随前向 ctx.save_for_backward 原样传入 | 输入 cnt（过滤语义与前向一致） |
| 上游梯度先 .contiguous() | 满足连续性约束 |
| backward 返回 (v_colorsOut, None, v_exposure or None, None, None, None, None, None) | means2d / cnt / width / height / clamp / calc_exp_grad 无梯度 |

## 5. 实现方案

### 5.1 分层结构

| 层 | 职责 | 关键文件 |
| --- | --- | --- |
| aclnn 适配（op_api） | 接口定义、参数校验、workspace 计算 | 算子包构建生成（本仓无手写 op_api 目录） |
| op_host | 属性解析、模式判定、Tiling 计算 | kernels/color_exposure_bwd/op_host/arch35/color_exposure_bwd.cpp、color_exposure_bwd_tiling.h |
| op_kernel | GM↔UB 搬运与梯度计算 | op_kernel/arch35/color_exposure_bwd.cpp、calc_color_exp_grad_vf.h |
| 框架绑定 | PyTorch / aten 桥接 | gauss_splat/csrc/ColorExposureBwd.cpp |

### 5.2 Tiling 策略

切分维度、长短核分配、尾块处理、TilingKey = 1 均与前向完全一致（见 aclnnColorExposureFwd.md 5.2），差异在 UB 预算：

- 每任务队列预算 bufferSize（float 数 / 任务）：全局模式 $3 \times 3 \times 2 = 18$（colors / vColorsIn / vColorsOut 队列各 2×3）；网格模式再 + $6 \times 2 = 12$（index / weight 队列各 2×3）→ 30；
- exposure 区恒按双份网格缓冲预算：`expSize = (2 × G_X·G_Y·G_Z × 12 + B·C) × 4 + 32KB`（exposure 常驻 + vExposure 累加缓冲 + cnt 常驻缓冲 $B \times C \times 4$ B；DCACHE_SIZE 无条件计入 expSize，与 kernel 无条件双缓冲一致，保守预算）；`SetDynUBufSize(ubTotalSize − 32KB)` 无条件执行；
- 网格规模上限：tiling 硬校验 $G_X G_Y G_Z \times 12 \le 24576$（MAX_GRID_DIMS），即 $G_X G_Y G_Z \le 2048$、最大网格 16×16×8，超出直接 tiling 失败；UB 预算约束下 B·C = 1 时 $G_X G_Y G_Z \le 1774$（随 B·C 略减，约为前向的一半）；测试验证至 12×12×8 = 1152，并新增 16×16×8 硬校验边界用例；

### 5.3 Kernel 实现

- 数据搬运路径：exposure 与 cnt（$B \times C$ INT32）各一次 GM→UB 常驻（DataCopyPad + 事件同步）；colors / vColorsIn（网格模式另含 index / weight）按 tile GM→UB；vColorsOut 按 tile UB→GM；vExposure 每 tile 从 UB 整块原子累加回 GM；
- 流水编排：CopyIn → CalculateGrad → CopyOut 事件同步串行推进，队列双缓冲；vExposure 回写（MTE3）与下一 tile 的缓冲清零（V）之间用 set_flag / wait_flag 串行化，避免读写竞争；外层 batch × camera × taskLoop 三层循环；
- **cnt 过滤实现**：与前向完全一致——每个 (b, c) 先读取 `cnt_ = cnt[b × C + c]`；每 tile 由 ComputingTaskNum 判定 `taskStart < cnt_`，不满足则 break 跳过该相机剩余 tile；cnt 落在 tile 内部时按 `validTaskNum = cnt − taskStart` 截断并强制走尾块掩码路径（DataCopyPad 掩码搬运 + VF 以 validTaskNum 掩码计算）；cnt = 0 时该相机所有 tile 直接 break，两个输出均不受影响；
- 使用的计算原语：`asc_vf_call` 启动 CalcGradUseGrid（网格）/ CalcGradGlobal（全局），dim3{1024, 1, 1}，每线程处理一个高斯：
  - 网格模式：由 index / weight 重建 8 角权重 $W$ 与 12 参数 $m$（不重算坐标映射）；
  - clamp 掩码：重算 $pre = Ac + b$ 判定符号，负通道梯度掩零；
  - `vColorsOut_j = Σ_i A_ij · v'_i`（按 $A$ 的列提取，即 $A^T v'$）；
  - `vExposure`：网格模式按 8 角原子累加（$g[4i+j] += W \cdot c_j \cdot v'_i$，$g[4i+3] += W \cdot v'_i$），全局模式等价 $W \equiv 1$；calcExpGrad = 0 时跳过；
- 缓冲管理：vExpBuf_ 恒分配（与 tiling 双份预算一致）；每 tile 先 Duplicate 清零，VF 原子累加后，CopyOut 以 SetAtomicAdd 整块回写 GM；
- 循环与边界处理：每 tile 由 ComputingTaskNum 得 taskNumPerCurLoop、isTailBlock 与 validTaskNum（cnt 截断后的有效任务数）；VF 以 validTaskNum 掩码处理尾部，同前向。

### 5.4 特殊场景处理

| 场景 | 处理方案 |
| --- | --- |
| 小 shape（N = 1） | totalTaskNum = 8，blockDim 收缩 + 尾块掩码覆盖 |
| 非对齐 shape（N % 8 ≠ 0） | tailNum 掩码 + DataCopyPad |
| cnt = 0（相机级全跳过） | taskLoop 首块 break，该相机零读写、零梯度 |
| cnt 非 8 对齐 / 落在 tile 中间 | validTaskNum 截断 + 强制尾块掩码路径 |
| 大 shape / 多核 | taskLoop 分块循环（已验证 N = 100000） |
| 多相机（C > 1） | camera 维循环复用同一套队列 / 常驻缓冲，cnt 逐相机读取 |
| calcExpGrad = 0 | VF 传标志跳过 vExposure 计算与原子回写；输出空 tensor |
| 空 tensor | 不支持（见 3.5） |

## 6. 精度与性能

### 6.1 精度标准与验证方法

- golden 来源：tests/ops/test_color_exposure.py 内置 CPU 参考实现（`vExposure` 用 float64 累加，其余 fp32；按 cnt 掩码，无效高斯零贡献）；NPU 侧 C++ 用例（tests/kernels/test_color_exposure_bwd.cpp）内置同构 golden（double 累加，cnt 混合模式：0 / 1 / N/2 / N）；
- 对比方法：vColorsOut 转 fp32 后 torch.allclose；vExposure 同法但容差随规模放宽；
- 误差阈值：vColorsOut rtol = atol = 1e-5；vExposure rtol = 2e-4、atol = 1e-5 + 5e-8·total（total = $\sum_{b,c} cnt[b,c]$ 有效高斯总数；cnt 过滤后误差只随有效高斯增长；差异全部来自 NPU fp32 原子累加次序 vs golden 高精度累加）；
- 混合精度说明：无。vExposure 累加为 fp32 原子加。

### 6.2 性能目标与测算

| 指标 | 理论值 | 目标值 | 实测值 | 达成率 |
| --- | --- | --- | --- | --- |
| 耗时 | 访存受限：每有效高斯读 6（全局）/ 12（网格）float、写 3 float；无效尾部高斯零开销；网格模式每 tile 另有 $G_X G_Y G_Z \times 48\,B$ 的 vExposure 原子回写 | 贴近同平台拷贝类算子 | 待标定 | - |
| 带宽利用率 | - | 同上 | 待标定 | - |

- 瓶颈分析：颜色通路 GM 搬运为主；calcExpGrad = 1 时 vExposure 整块原子回写为次要开销；
- 优化手段：整块原子回写（非逐元素原子）、队列双缓冲、复用前向 index / weight 免重算插值、cnt 过滤跳过无效高斯。

## 7. 测试设计

### 7.1 功能用例

- NPU C++ 用例（tests/kernels/test_color_exposure_bwd.cpp，10 例）：global_clamp_calc（B=1, N=257，clamp + calcExpGrad）、global_nocalc（B=2, calcExpGrad = 0 → vExposure 为空）、grid_clamp_calc（B=2, N=517, 5×5×6）、grid_nocalc（N=4096, 3×5×4）、grid_large（N=100000, 12×12×8, clamp + calcExpGrad）、grid_max_16x16x8（B=2, C=2, N=517, 16×16×8 硬校验上限）、global_multi_camera（B=2, C=3）、grid_cnt_mixed（B=2, C=3, N=517, 4×4×4, cnt 混合）、grid_cnt_mixed_nocalc（B=2, C=2, N=123, 3×3×4, cnt 混合 + 不计算 vExp）、grid_cnt_mixed_large（B=1, C=4, N=100000, 12×12×8, cnt 混合）；cnt 混合模式按 (b, c) 平坦下标取 0 / 1 / N/2 / N；index / weight 按前向语义生成（仅 $[0, cnt)$ 有效）；golden 一律用非对称随机仿射矩阵（A=rand），覆盖转置反传路径；
- CPU golden 对拍（tests/ops/test_color_exposure.py）：test_bwd_global_random、test_bwd_grid_random、test_bwd_cnt_filter（cnt 全 0 / 混合）、test_bwd_grid_5d_exposure、test_bwd_grid_max_16x16x8（16×16×8 硬校验上限）。

### 7.2 边界用例

- test_bwd_clamp_mask：构造前向输出为负的高斯，验证负通道梯度掩零；并以 clamp 双取值 golden 对照，确认掩码确实改变梯度；
- test_bwd_large：N = 100000，验证大规模 fp32 原子累加精度；
- 尾块与非对齐：N = 13 / 123 / 257 / 517；最小网格 2×2×2；多 batch / camera B ∈ {1, 2}、C ∈ {1, 2, 3, 4}；
- cnt 边界：cnt = 0（整相机零梯度）、cnt = 1（单高斯）、cnt = N/2（块内截断）、cnt = N（全量），及其组合。

### 7.3 精度用例

- test_bwd_v_colors_transpose：基矩阵钉死用例，将 exposure 的 $A$ 部分指定元素置常数（如 $e_1 = k$、$e_8 = k$），逐元素验证 `vColorsOut = A^T·v'` 的通道映射（$e_1 = k \Rightarrow vOut_g = k \cdot v_r$，$e_8 = k \Rightarrow vOut_r = k \cdot v_b$），golden 自检 + NPU 对拍；
- test_autograd_vs_torch_grad：随机输入下与 torch.autograd.grad 的解析反传对拍；
- cnt 过滤钉死：cnt = 0 的相机 vColorsOut 全 0 且对 vExposure 零贡献；cnt < N 时 [cnt, N) 区间 vColorsOut 全 0；
- 容差：vColorsOut 1e-5；vExposure 按 6.1 随有效高斯总数放宽。

### 7.4 性能与泛化用例

- shape 扫描同前向（N ∈ {1, 7, 13, 123, 257, 4096, 100000}，B、C、网格、clamp、exposure 排布），另覆盖 clamp × calcExpGrad 全组合与 cnt ∈ {全量, 混合, 全 0}；
- autograd 端到端：test_autograd_function、test_autograd_global、test_autograd_cnt_filter（cnt 模式下端到端梯度正确性）、test_autograd_grad_contract（梯度契约：autograd 数值一致性）；封装校验 test_wrapper_input_validation。

## 8. 风险与备选方案

| 风险 | 影响 | 应对措施 / 备选方案 |
| --- | --- | --- |
| vExposure 未预清零 | 原子累加到旧值，梯度错误 | 语义约定调用方清零；上层封装已用 zeros_like；aclnn 裸调用者需自查 |
| vColorsOut 未预清零 | 无效高斯位置（$n \ge cnt$）残留旧值 | 语义约定输出 tensor 需预清零；上层封装已用 zeros 分配；aclnn 裸调用者需自查 |
| cnt 语义越界（cnt > N 或负值） | cnt > N 按全量处理（结果正确）；负值经无符号比较行为未定义 | 约定 $0 \le cnt[b,c] \le N$（上游投影 / 过滤保证，且与前向一致）；如需防御可在 aclnn 适配层加校验 |
| fp32 原子累加精度 | vExposure 误差随有效高斯总数增长 | 容差随规模放宽（atol = 1e-5 + 5e-8·total）；更高精度需改为分块部分和 + 归约 |
| 网格预算为前向一半（双份网格缓冲） | 硬校验放行的 16×16×8 = 2048 单元 | 已新增 grid_max_16x16x8 边界用例固化该边界；|
| index / weight 被篡改或自行计算 | 梯度错误 | 约定原样传前向输出；文档与用例均按此约定固化 |
| 网格任一维 = 1 / NaN + clamp | 同前向 | 见 aclnnColorExposureFwd.md 第 8 节 |

## 9. 附录

### 9.1 术语表

| 术语 | 含义 |
| --- | --- |
| 12 参数仿射 / $[A\|b]$ | 3×4 行增广矩阵：$A_{ij} = m[4i+j]$，$b_i = m[4i+3]$ |
| cnt 过滤 | 仅前 cnt[b, c] 个高斯（紧凑布局）产生梯度，其余位置不写回、不累加，与前向过滤语义对齐 |
| clamp 掩码 | clamp = 1 时前向负输出通道的上游梯度掩零（ReLU 反传语义） |
| 原子累加 / SetAtomicAdd | vExposure 跨高斯累加的 fp32 原子加实现，次序不定 |
| SIMT VF | Ascend 950 arch35 的 SIMT 向量引擎（asc_vf_call 启动，1024 线程） |
| 长短核 / taskNumPerLoop | 多核负载均衡策略与单核 tile 大小，同前向 |
| validTaskNum | tile 内实际参与计算的任务数：尾块扣除 tailNum，cnt 截断时取块内剩余有效数 |

### 9.2 参考资料

- docs/zh/kernels/aclnnColorExposureFwd.md：配套前向算子设计说明
- CANN Ascend C 编程指南（TPipe / Tiling / asc_vf_call / SetAtomicAdd）
- gsplat CUDA 参考实现

### 9.3 变更记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | 初稿（vColorsOut 按数学定义 $A^T \cdot v'$ 反传） |
| v1.1 | 2026-11-23 | - | 输入 / 输出从 [B, C, N]（batch×camera 融合）重构为 [B, C, D, N]（显式相机维）；新增 INT32 输入 cnt（逐 (b, c) 有效高斯计数），kernel 按 cnt 跳过无效尾部高斯（不写回 vColorsOut、不累加 vExposure）；index / weight 形状同步变为 [B, C, 3, N]；更新 Tiling（cnt 常驻缓冲、三层循环、网格硬校验）与测试设计；vExposure 容差基数改为有效高斯总数 Σcnt；新增最大网格 16×16×8（硬校验边界）用例——A5 上当前 tiling 公式 UB 预算下溢，真机通过需内核侧修复 |
