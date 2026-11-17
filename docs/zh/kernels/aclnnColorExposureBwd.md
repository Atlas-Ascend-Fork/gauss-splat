# ColorExposureBwd 设计说明

[📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_bwd/op_kernel/arch35/color_exposure_bwd.cpp)

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | ColorExposureBwd |
| 所属仓 / 目录 | gauss-splat / kernels/color_exposure_bwd |
| 算子类型 | 自定义（逐高斯颜色仿射变换的反向梯度） |
| aclnn 接口名 | aclnnColorExposureBwd |
| 文档版本 | v1.0 |
| 作者 | - |
| 评审人 | - |
| 最后更新 | 2026-11-16 |

## 2. 需求背景与目标

### 2.1 业务场景

与 ColorExposureFwd 配套的反向算子。3DGS 训练中曝光参数与高斯属性联合优化，反向传播需要两条梯度：

- `vColorsOut`：损失对前向输入 `colors` 的梯度，回传给上游 SH 颜色计算；
- `vExposure`：损失对 `exposure` 的梯度（由属性 `calcExpGrad` 控制是否计算），用于曝光参数的优化。

前向已显式输出插值中间量 `index` / `weight`，反向直接复用，无需重算坐标→网格插值，`means2d` 不再参与反向。

### 2.2 对标算子

对标 gsplat 生态 CUDA 参考实现的 `color_exposure` 反向，数学语义一致。与本实现的差异：

- CUDA 参考实现在反向重算插值；本实现直接复用前向输出的 `index` / `weight`，省一次坐标→网格映射；
- 硬件通路：基于 Ascend 950（arch35）AIV 向量核 + SIMT VF，梯度计算整体在 SIMT VF 完成（1024 线程，每线程一个高斯）；
- `vExposure` 的跨高斯累加用 fp32 原子加（SetAtomicAdd）实现，与 golden 的高精度累加存在次序性误差，容差随高斯总数放宽（见 6.1）。

### 2.3 目标

- 功能目标：
  - 覆盖：全局 / 网格两种模式的 `vColorsOut`；`vExposure` 由 `calcExpGrad` 控制（=1 输出并累加，=0 输出空 tensor）；`clamp` 梯度掩码（ReLU 反传语义）。
- 性能目标：访存受限型算子（每高斯读 6~12 float、写 3 float），GM 带宽利用率贴近同平台拷贝类算子水平（实测待标定）。
- 精度目标：`vColorsOut` 对 CPU golden 的误差 rtol = atol = 1e-5；`vExposure` rtol = 2e-4、atol = 1e-5 + 5e-8·N（N 为高斯总数，误差来自 fp32 原子累加次序）。
- 交付范围：dtype FLOAT（`index` INT32）、format ND、shape 约束同前向；网格字节预算为前向的一半（双份网格缓冲，见 5.2）；仅支持 Ascend 950PR / 950DT。

## 3. 算子定义

### 3.1 功能描述

给定前向输入 `colors` / `exposure`、上游颜色梯度 `vColorsIn`（网格模式另需前向输出的 `index` / `weight`），计算：

- `vColorsOut` $[B, 3, N]$：`vColorsIn` 经仿射矩阵转置反传并对 colors 求梯度，`clamp` = 1 时前向负输出通道的上游梯度被掩零；
- `vExposure`：损失对 `exposure` 的梯度，形状同 `exposure`，为批内全部高斯的累加和；`calcExpGrad` = 0 时输出空 tensor。

### 3.2 数学表达式

记逐高斯仿射参数 $m = [A|b]$（$A_{ij} = m[4i+j]$，$b_i = m[4i+3]$）：

- 全局模式：$m$ 直接取 `exposure`；
- 网格模式：由前向输出的 `index` / `weight` 重建三线性插值（不再使用 `means2d` 坐标）：

```text
m_k = Σ_{(cx,cy,cz)∈{0,1}^3} wx^cx · wy^cy · wz^cz · E[ix+cx, iy+cy, iz+cz, k]
```

**clamp 掩码**（clamp = 1 时，$pre = Ac + b$ 为前向截断前输出）：

```text
v'_i = (pre_i < 0) ? 0 : v_i        （ReLU 反传语义）
```

**v_colors**（$A$ 的转置反传，$dL/dc = A^T \cdot dL/dout$）：

```text
vColorsOut_j = Σ_i A_ij · v'_i
```

**v_exposure**（逐元素，跨批内全部高斯累加；$W$ 为该高斯的 8 角三线权重，全局模式 $W \equiv 1$）：

```text
g[..., 4i+j] += W · c_j · v'_i
g[..., 4i+3] += W · v'_i
```

补充说明：

- `vExposure` 为累加语义：kernel 以 fp32 原子加回写，调用前必须将 `vExposure` 清零（上层封装已用 zeros 分配，见 4.3）；
- 网格模式的 $A$、$W$ 均由 `index` / `weight` 重建，调用方必须传入前向的原样输出。

### 3.3 输入、输出与属性

| 名称 | 方向 | 含义 | dtype | shape | format | 是否可选 |
| --- | --- | --- | --- | --- | --- | --- |
| colors | 输入 | 前向颜色 | FLOAT | [B, 3, N] | ND | 否 |
| vColorsIn | 输入 | 上游颜色梯度 | FLOAT | [B, 3, N] | ND | 否 |
| exposure | 输入 | 前向曝光参数 | FLOAT | 全局：[12] / (4,3) / (1,1,1,12)；网格：(GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | 否 |
| index | 输入 | 前向输出的插值下标 | INT32 | [B, 3, N] | ND | 网格模式必选；全局模式传空 tensor |
| weight | 输入 | 前向输出的三线权重 | FLOAT | [B, 3, N] | ND | 同 index |
| clamp | 属性 | 与前向一致 | bool（0/1） | - | - | 否 |
| calcExpGrad | 属性 | 是否计算并累加 v_exposure | bool（0/1） | - | - | 否 |
| vColorsOut | 输出 | 对 colors 的梯度 | FLOAT | [B, 3, N] | ND | 否 |
| vExposure | 输出 | 对 exposure 的梯度（原子累加语义，需预清零） | FLOAT | 同 exposure；calcExpGrad = 0 时为空 tensor | ND | 否 |

### 3.4 支持矩阵

| 维度 | 支持范围 | 不支持项及原因 |
| --- | --- | --- |
| 产品 | Ascend 950PR / 950DT（arch35） | 其他产品不支持：算子仅适配 arch35（SIMT VF），未编译其他架构版本 |
| dtype | FLOAT（index 为 INT32） | fp16 / bf16 / 整型等不支持：梯度通路全程 fp32 |
| format | ND | 其他 format 不支持：kernel 按紧凑布局直接寻址 |
| shape | colors / vColorsIn / vColorsOut 均为 [B,3,N] 且一致；exposure 同前向；网格字节预算 $2 \times G_X G_Y G_Z \times 48\,B$ 受 UB 约束（见 5.2，约为前向一半） | 空 tensor（shape 含 0）、0 维输入不支持；网格维 = 1 见第 8 节 |
| 广播 | 不支持 | exposure 为全 batch 共享参数 |

### 3.5 边界与异常场景

| 场景 | 预期行为 |
| --- | --- |
| 空 tensor（shape 含 0） | 必选输入 / 输出非空；全局模式 index / weight 传空 tensor；calcExpGrad = 0 时 vExposure 传空 tensor（kernel 不回写） |
| 标量 / 0 维输入 | 不支持，参数校验报错（161002） |
| 非连续输入 | 不支持：kernel 按紧凑布局寻址，需调用方保证连续 |
| 输入含 NaN / Inf | 未定义：clamp 掩码的比较运算对 NaN 行为平台相关，不建议注入 |
| vExposure 未预清零 | SetAtomicAdd 累加到旧值，梯度错误；调用方必须预清零（上层封装已保证） |
| index / weight 与前向输出不匹配 | 梯度错误；约定必须原样传入前向输出 |
| shape 或 dtype 不合法 | 报错 161001（空指针）/ 161002（dtype、shape 校验失败），见 4.2 |
| workspace 不足 | 不涉及用户侧 workspace：由 aclnn 框架按 GetWorkspaceSize 返回值管理（仅系统 workspace） |

## 4. 接口设计

### 4.1 aclnn 接口与调用流程

```text
aclnnColorExposureBwdGetWorkspaceSize(colors, vColorsIn, exposure, index, weight,
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
| dtype 组合 | colors / vColorsIn / exposure / weight / vColorsOut / vExposure 为 FLOAT，index 为 INT32 | 161002 |
| shape 一致性 | colors / vColorsIn / vColorsOut 均为 [B,3,N] 且一致；exposure numel ≥ 12；网格模式 index / weight 必须为 [B,3,N]；calcExpGrad = 1 时 vExposure 形状必须与 exposure 一致 | 161002 |
| format | ND | 161002 |

（校验逻辑由算子包构建生成的 aclnn 适配层完成，与仓内其他算子一致。）

### 4.3 上层框架映射

上层链路：`gauss_splat.ops.color_exposure`（torch.autograd.Function.backward）→ `gauss_splat/csrc/ColorExposureBwd.cpp`（EXEC_NPU_CMD 直调）→ aclnnColorExposureBwd。

| 上层行为 | 本算子对应 |
| --- | --- |
| vColorsOut = at::zeros({B,3,N}) 后传入 | 输出 vColorsOut（kernel 覆写） |
| vExposure = calcExpGrad ? at::zeros_like(exposure) : at::empty(0) | 输出 vExposure（zeros 满足预清零要求；空 tensor 对应 calcExpGrad = 0） |
| 上游梯度先 .contiguous() | 满足连续性约束 |
| backward 返回 (v_colorsOut, None, v_exposure or None, ...) | means2d / width / height 无梯度 |

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
- exposure 区恒按双份网格缓冲预算：`expSize = 2 × G_X·G_Y·G_Z × 12 × 4 + 32KB`（exposure 常驻 + vExposure 累加缓冲，与 kernel 无条件双缓冲一致，保守预算）；`SetDynUBufSize(ubTotalSize − 32KB)` 无条件执行；
- 由此网格规模上限约为前向的一半：Ascend 950（UB = 253952 B）上 $G_X G_Y G_Z \le 1779$，测试验证至 12×12×8 = 1152；16×16×8 = 2048 超预算，不支持。

### 5.3 Kernel 实现

- 数据搬运路径：exposure 一次 GM→UB 常驻；colors / vColorsIn（网格模式另含 index / weight）按 tile GM→UB；vColorsOut 按 tile UB→GM；vExposure 每 tile 从 UB 整块原子累加回 GM；
- 流水编排：CopyIn → CalculateGrad → CopyOut 事件同步串行推进，队列双缓冲；vExposure 回写（MTE3）与下一 tile 的缓冲清零（V）之间用 set_flag / wait_flag 串行化，避免读写竞争；
- 使用的计算原语：`asc_vf_call` 启动 CalcGradUseGrid（网格）/ CalcGradGlobal（全局），dim3{1024, 1, 1}，每线程处理一个高斯：
  - 网格模式：由 index / weight 重建 8 角权重 $W$ 与 12 参数 $m$（不重算坐标映射）；
  - clamp 掩码：重算 $pre = Ac + b$ 判定符号，负通道梯度掩零；
  - `vColorsOut_j = Σ_i A_ij · v'_i`（按 $A$ 的列提取，即 $A^T v'$）；
  - `vExposure`：网格模式按 8 角原子累加（$g[4i+j] += W \cdot c_j \cdot v'_i$，$g[4i+3] += W \cdot v'_i$），全局模式等价 $W \equiv 1$；calcExpGrad = 0 时跳过；
- 缓冲管理：vExpBuf_ 恒分配（与 tiling 双份预算一致）；每 tile 先 Duplicate 清零，VF 原子累加后，CopyOut 以 SetAtomicAdd 整块回写 GM；
- 循环与边界处理：外层 batch × taskLoop 双循环；validTaskNum 掩码处理尾块，同前向。

### 5.4 特殊场景处理

| 场景 | 处理方案 |
| --- | --- |
| 小 shape（N = 1） | totalTaskNum = 8，blockDim 收缩 + 尾块掩码覆盖 |
| 非对齐 shape（N % 8 ≠ 0） | tailNum 掩码 + DataCopyPad |
| 大 shape / 多核 | taskLoop 分块循环（已验证 N = 100000） |
| calcExpGrad = 0 | VF 传标志跳过 vExposure 计算与原子回写；输出空 tensor |
| 空 tensor | 不支持（见 3.5） |

## 6. 精度与性能

### 6.1 精度标准与验证方法

- golden 来源：tests/ops/test_color_exposure.py 内置 CPU 参考实现（`vExposure` 用 float64 累加，其余 fp32）；NPU 侧 C++ 用例（tests/kernels/test_color_exposure_bwd.cpp）内置同构 golden（double 累加）；
- 对比方法：vColorsOut 转 fp32 后 torch.allclose；vExposure 同法但容差随规模放宽；
- 误差阈值：vColorsOut rtol = atol = 1e-5；vExposure rtol = 2e-4、atol = 1e-5 + 5e-8·N（N 为高斯总数；差异全部来自 NPU fp32 原子累加次序 vs golden 高精度累加）；
- 混合精度说明：无。vExposure 累加为 fp32 原子加。

### 6.2 性能目标与测算

| 指标 | 理论值 | 目标值 | 实测值 | 达成率 |
| --- | --- | --- | --- | --- |
| 耗时 | 访存受限：每高斯读 6（全局）/ 12（网格）float、写 3 float；网格模式每 tile 另有 $G_X G_Y G_Z \times 48\,B$ 的 vExposure 原子回写 | 贴近同平台拷贝类算子 | 待标定 | - |
| 带宽利用率 | - | 同上 | 待标定 | - |

- 瓶颈分析：颜色通路 GM 搬运为主；calcExpGrad = 1 时 vExposure 整块原子回写为次要开销；
- 优化手段：整块原子回写（非逐元素原子）、队列双缓冲、复用前向 index / weight 免重算插值。

## 7. 测试设计

### 7.1 功能用例

- NPU C++ 用例（tests/kernels/test_color_exposure_bwd.cpp，5 例）：global_clamp_calc（clamp + calcExpGrad）、global_nocalc（calcExpGrad = 0 → vExposure 为空）、grid_clamp_calc（5×5×6）、grid_nocalc（3×5×4）、grid_large（N=100000, 12×12×8, clamp + calcExpGrad）；golden 一律用非对称随机仿射矩阵（A=rand），覆盖转置反传路径；
- CPU golden 对拍（tests/ops/test_color_exposure.py）：test_bwd_global_random、test_bwd_grid_random、test_bwd_grid_5d_exposure。

### 7.2 边界用例

- test_bwd_clamp_mask：构造前向输出为负的高斯，验证负通道梯度掩零；并以 clamp 双取值 golden 对照，确认掩码确实改变梯度；
- test_bwd_large：N = 100000，验证大规模 fp32 原子累加精度；
- 尾块与非对齐：N = 13 / 257 / 517；最小网格 2×2×2；多 batch B = 2。

### 7.3 精度用例

- test_bwd_v_colors_transpose：基矩阵钉死用例，将 exposure 的 $A$ 部分指定元素置常数（如 $e_1 = k$、$e_8 = k$），逐元素验证 `vColorsOut = A^T·v'` 的通道映射（$e_1 = k \Rightarrow vOut_g = k \cdot v_r$，$e_8 = k \Rightarrow vOut_r = k \cdot v_b$），golden 自检 + NPU 对拍；
- test_autograd_vs_torch_grad：随机输入下与 torch.autograd.grad 的解析反传对拍；
- 容差：vColorsOut 1e-5；vExposure 按 6.1 随规模放宽。

### 7.4 性能与泛化用例

- shape 扫描同前向，另覆盖 clamp × calcExpGrad 全组合；
- autograd 端到端：test_autograd_function、test_autograd_global、test_autograd_grad_contract（梯度契约：autograd 数值一致性）；封装校验 test_wrapper_input_validation。

## 8. 风险与备选方案

| 风险 | 影响 | 应对措施 / 备选方案 |
| --- | --- | --- |
| vExposure 未预清零 | 原子累加到旧值，梯度错误 | 语义约定调用方清零；上层封装已用 zeros_like；aclnn 裸调用者需自查 |
| fp32 原子累加精度 | vExposure 误差随高斯总数 N 增长 | 容差随规模放宽（atol = 1e-5 + 5e-8·N）；更高精度需改为分块部分和 + 归约 |
| 网格预算为前向一半 | 大网格（如 16×16×8）tiling 异常 | 约束 $G_X G_Y G_Z \le 1779$（Ascend 950）；更大网格需拆分曝光场，或改造为 calcExpGrad = 0 时不分配 vExp 缓冲 |
| index / weight 被篡改或自行计算 | 梯度错误 | 约定原样传前向输出；文档与用例均按此约定固化 |
| 网格任一维 = 1 / NaN + clamp | 同前向 | 见 aclnnColorExposureFwd.md 第 8 节 |

## 9. 附录

### 9.1 术语表

| 术语 | 含义 |
| --- | --- |
| 12 参数仿射 / $[A|b]$ | 3×4 行增广矩阵：$A_{ij} = m[4i+j]$，$b_i = m[4i+3]$ |
| clamp 掩码 | clamp = 1 时前向负输出通道的上游梯度掩零（ReLU 反传语义） |
| 原子累加 / SetAtomicAdd | vExposure 跨高斯累加的 fp32 原子加实现，次序不定 |
| SIMT VF | Ascend 950 arch35 的 SIMT 向量引擎（asc_vf_call 启动，1024 线程） |
| 长短核 / taskNumPerLoop | 多核负载均衡策略与单核 tile 大小，同前向 |

### 9.2 参考资料

- docs/zh/kernels/aclnnColorExposureFwd.md：配套前向算子设计说明
- CANN Ascend C 编程指南（TPipe / Tiling / asc_vf_call / SetAtomicAdd）
- gsplat CUDA 参考实现

### 9.3 变更记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | 初稿（vColorsOut 按数学定义 $A^T \cdot v'$ 反传） |
