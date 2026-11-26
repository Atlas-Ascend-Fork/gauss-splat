# ColorExposureFwd 设计说明

[📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_fwd/op_kernel/arch35/color_exposure_fwd.cpp)

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | ColorExposureFwd |
| 所属仓 / 目录 | gauss-splat / kernels/color_exposure_fwd |
| 算子类型 | 自定义（逐高斯颜色仿射变换） |
| aclnn 接口名 | aclnnColorExposureFwd |
| 文档版本 | v1.1 |
| 作者 | - |
| 评审人 | - |
| 最后更新 | 2026-11-23 |

## 2. 需求背景与目标

### 2.1 业务场景

3D Gaussian Splatting（3DGS）多视角训练中，不同视角相机的曝光 / 白平衡差异会使同一高斯球在各图像中呈现系统性颜色漂移，影响重建-渲染一致性。为此在 SH 颜色计算之后、光栅化之前，对逐高斯颜色施加一组可学习的曝光仿射变换（3×4 行增广矩阵 $[A\|b]$，共 12 参数），并与高斯属性联合优化。曝光参数支持两种粒度：

- **全局模式**：全 batch 共享一组 12 参数，所有 (batch, camera) 内的高斯共用；
- **网格模式**：$(G_X, G_Y, G_Z, 12)$ 参数场，按高斯投影坐标 $(x, y)$ 与颜色灰度做 8 角三线性插值，建模空间变化的曝光。

输入 / 输出按 $[B, C, D, N]$ 组织：$B$ 为 batch，$C$ 为每 batch 的相机（视角）数，$D$ 为通道维度（colors 为 3、means2d 为 2），$N$ 为每相机的高斯容量。上游投影 / 过滤算子输出逐 $(b, c)$ 的有效高斯计数 `cnt` $[B, C]$：紧凑布局下只有每相机前 `cnt[b, c]` 个高斯有效，本算子据此跳过无效尾部高斯，避免无意义计算与访存。

曝光参数为可学习参数，训练时需要梯度，因此前向（本算子）与反向（ColorExposureBwd）成对交付。

### 2.2 对标算子

新增算子。

### 2.3 目标

- 功能目标：
  - 覆盖：全局 / 网格两种模式；`clamp` 非负截断（可选）；网格模式输出 `index` / `weight`；`exposure` 支持 $(G_X, G_Y, G_Z, 12)$ 与 $(G_X, G_Y, G_Z, 4, 3)$ 两种排布；多相机维 $C \ge 1$；`cnt` 过滤（仅前 `cnt[b, c]` 个高斯参与计算，其余位置不写回）。
- 性能目标：访存受限型算子，GM 带宽利用率贴近同平台拷贝类算子水平（每有效高斯读 3~5 float、写 3~9 float；`cnt` 过滤跳过无效尾部高斯，进一步减少无效访存；实测待标定）。
- 精度目标：`output` / `weight` 对 CPU golden 的误差 rtol = atol = 1e-5；`index` 逐位一致。
- 交付范围：dtype FLOAT（`index` / `cnt` INT32）、format ND、$B, C, N \ge 1$、$0 \le cnt[b,c] \le N$、网格各维 ≥ 2 且 $G_X G_Y G_Z \le 2048$（tiling 硬校验上限，即最大网格 16×16×8）；仅支持 Ascend 950PR / 950DT。

## 3. 算子定义

### 3.1 功能描述

对 $[B, C, 3, N]$ 的逐高斯颜色 `colors` 按 `exposure` 做仿射变换，输出 $[B, C, 3, N]$ 的 `output`：

- 全局模式（`exposure` 元素数 = 12）：所有高斯共用同一组 12 参数（3×4 行增广 $[A\|b]$），$out = Ac + b$；
- 网格模式（`exposure` 元素数 > 12，形状 $(G_X, G_Y, G_Z, 12)$ 或 $(G_X, G_Y, G_Z, 4, 3)$）：以 $[B, C, 2, N]$ 的投影像素坐标 `means2d` 与颜色灰度为插值自变量，先在 $G_X \times G_Y \times G_Z$ 网格上 8 角三线性插值出该高斯的 12 参数，再做仿射变换；
- `clamp` = 1 时输出逐通道 $\max(\cdot, 0)$；
- **cnt 过滤**：INT32 输入 `cnt` $[B, C]$ 给出每个 $(b, c)$ 的有效高斯数，仅高斯 $n < cnt[b, c]$ 参与计算；$n \ge cnt[b,c]$ 的位置 kernel 不写回，保持输出 tensor 的初值（上层封装以 zeros 分配，即 0）；
- 网格模式额外输出插值下标 `index`（INT32）与三线权重 `weight`（FLOAT），形状均为 $[B, C, 3, N]$，供反向算子复用（同样只在有效高素位置写回）；全局模式下 `index` / `weight` 传空 tensor（numel = 0）。

### 3.2 数学表达式

12 个参数按 3×4 行增广矩阵 $[A\|b]$ 解释：$A_{ij} = exposure[4i+j]$，$b_i = exposure[4i+3]$（$i, j \in \{0, 1, 2\}$）。

对每个 $(b, c)$，仅对 $n < cnt[b, c]$ 的高斯计算（其余位置输出 0）：

**全局模式**（exposure 元素数 = 12）：

```text
pre_i = Σ_j A_ij · c_j + b_i
out_i = clamp ? max(pre_i, 0) : pre_i
```

**网格模式**（exposure 元素数 > 12，记网格为 $E$），三段流水：

```text
① 灰度：      grey = 0.299·r + 0.587·g + 0.114·b

② 网格坐标（fl32 为 fp32 舍入；无 +0.5 像素中心偏移；出界钳位到 [0,1]）：
    u = clamp01(x · fl32(1/W)) · (GX−1)
    v = clamp01(y · fl32(1/H)) · (GY−1)
    w = clamp01(grey)   · (GZ−1)
    ix = trunc(u)，wx = u − ix；当 ix == GX−1 时端点闭包 (ix, wx) ← (GX−2, 1)
    （y / z 方向同理）

③ 8 角三线性插值出逐高斯参数 m，按 m = [A|b] 仿射 colors，再按 clamp 截断（同全局模式）：
    m_k = Σ_{(cx,cy,cz)∈{0,1}^3} wx^cx · wy^cy · wz^cz · E[ix+cx, iy+cy, iz+cz, k]

输出：index = (ix, iy, iz)，weight = (wx, wy, wz)
```

补充说明：

- $x / y$ 为像素坐标（`means2d` 第 0 / 1 通道），$W / H$ 为图像宽高（属性 `width` / `height`）；
- 灰度先 clamp01 再乘 $(G_Z-1)$，负色 / 超界色被截到网格端点；
- 全部中间量为 fp32（灰度系数、倒数、权重均为 fp32 常量 / fp32 运算），无混合精度；
- $n \ge cnt[b,c]$ 的高斯不产生任何输出（`output` / `index` / `weight` 对应位置保持初值 0）。

### 3.3 输入、输出与属性

| 名称 | 方向 | 含义 | dtype | shape | format | 是否可选 |
| --- | --- | --- | --- | --- | --- | --- |
| colors | 输入 | 高斯颜色（SH 前向结果） | FLOAT | [B, C, 3, N] | ND | 否 |
| means2d | 输入 | 高斯投影像素坐标 | FLOAT | [B, C, 2, N] | ND | 网格模式必选；全局模式不参与计算（传合法非空 tensor） |
| exposure | 输入 | 曝光参数 | FLOAT | 全局：[12] / (4,3) / (1,1,1,12)；网格：(GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | 否 |
| cnt | 输入 | 逐 (b, c) 有效高斯计数（来自上游投影 / 过滤） | INT32 | [B, C] | ND | 否 |
| width | 属性 | 图像宽 W（像素） | int32 | - | - | 否 |
| height | 属性 | 图像高 H（像素） | int32 | - | - | 否 |
| clamp | 属性 | 输出是否逐通道非负截断 | bool（0/1） | - | - | 否 |
| output | 输出 | 曝光校正后颜色 | FLOAT | [B, C, 3, N] | ND | 否 |
| index | 输出 | 网格插值下标 (ix, iy, iz) | INT32 | [B, C, 3, N] | ND | 网格模式必选；全局模式传空 tensor（numel = 0） |
| weight | 输出 | 三线插值权重 (wx, wy, wz) | FLOAT | [B, C, 3, N] | ND | 同 index |

### 3.4 支持矩阵

| 维度 | 支持范围 | 不支持项及原因 |
| --- | --- | --- |
| 产品 | Ascend 950PR / 950DT（arch35） | 其他产品不支持：算子仅适配 arch35（SIMT VF / Reg 向量指令），未编译其他架构版本 |
| dtype | FLOAT（index / cnt 为 INT32） | fp16 / bf16 / 整型颜色等不支持：颜色通路全程 fp32，无混合精度需求 |
| format | ND | 其他 format 不支持：kernel 按紧凑布局直接寻址 |
| shape | colors [B,C,3,N]、means2d [B,C,2,N]、cnt [B,C]，B ≥ 1、C ≥ 1、N ≥ 1、0 ≤ cnt[b,c] ≤ N；exposure numel = 12 或 (GX,GY,GZ,12) / (GX,GY,GZ,4,3)，网格各维 ≥ 2、$G_X G_Y G_Z \le 2048$（MAX_GRID_DIMS 硬校验，即最大 16×16×8，见 5.2） | 空 tensor（shape 含 0）、0 维输入不支持；cnt > N 按全量处理、负值未定义（约定不传入）；网格维 = 1 见第 8 节 |
| 广播 | 不支持 | exposure 为全 batch / 相机共享的固定语义参数，非广播 |

### 3.5 边界与异常场景

| 场景 | 预期行为 |
| --- | --- |
| cnt[b, c] = 0 | 该相机整个跳过（taskLoop 首块即 break），output / index / weight 对应 (b, c) 全部保持 0 |
| cnt[b, c] < N | 仅前 cnt 个高素写回；尾部无效高斯不读不写；cnt 非 8 对齐 / 跨 taskLoop 分块时按掩码截断（validTaskNum） |
| cnt[b, c] > N | 按全量（= N）处理；约定调用方不传入 |
| 空 tensor（shape 含 0） | 不支持：必选输入 / 输出非空；全局模式 index / weight 传空 tensor（numel = 0）表示不输出，kernel 不分配、不回写 |
| 标量 / 0 维输入 | 不支持，参数校验报错（161002） |
| 非连续输入 | 不支持：kernel 按紧凑布局寻址，需调用方保证连续（上层封装不自动转连续） |
| 输入含 NaN / Inf | 未定义：clamp 的 $\max(\cdot, 0)$ 对 NaN 行为平台相关，kernel 不做特殊检查，不建议注入 |
| shape 或 dtype 不合法 | 报错 161001（空指针）/ 161002（dtype、shape 校验失败），见 4.2 |
| workspace 不足 | 不涉及用户侧 workspace：由 aclnn 框架按 GetWorkspaceSize 返回值管理（本算子仅申请系统 workspace） |

## 4. 接口设计

### 4.1 aclnn 接口与调用流程

```text
aclnnColorExposureFwdGetWorkspaceSize(colors, means2d, exposure, cnt, width, height, clamp,
                                      output, index, weight, workspaceSize, executor)
aclnnColorExposureFwd(workspace, workspaceSize, executor, stream)
```

- workspace 计算规则：仅使用系统 workspace（LibApiWorkSpaceSize），无算子自定义 workspace；用户按 `GetWorkspaceSize` 返回值分配即可，空 tensor 场景同规则。
- 执行器复用规则：每次 `GetWorkspaceSize` 完成参数校验并创建执行器，`aclnnColorExposureFwd` 执行一次；同一输入重新调用需重新走两段式流程（标准 aclnn 语义）。

### 4.2 参数校验与报错

| 校验项 | 规则 | 报错码 / 日志关键字 |
| --- | --- | --- |
| 空指针 | 所有输入 / 输出 tensor 指针及 workspaceSize / executor 非空 | 161001，"param ... is null" |
| dtype 组合 | colors / means2d / exposure / output / weight 为 FLOAT，index / cnt 为 INT32 | 161002 |
| shape 一致性 | colors [B,C,3,N] 与 means2d [B,C,2,N] 的 B、C、N 一致；cnt 为 [B,C]；exposure numel ≥ 12；网格模式 index / weight 必须为 [B,C,3,N]，全局模式传空 tensor | 161002 |
| format | ND | 161002 |

（校验逻辑由算子包构建生成的 aclnn 适配层完成，与仓内其他算子一致。）

### 4.3 上层框架映射

无 torch 原生算子对应。上层链路：`gauss_splat.ops.color_exposure`（torch.autograd.Function）→ `gauss_splat/csrc/ColorExposureFwd.cpp`（EXEC_NPU_CMD 直调，不改变 stride）→ aclnnColorExposureFwd。渲染管线 `gauss_splat.ops.rendering` 在投影 / 过滤之后调用本算子，`cnt` 即投影算子输出的过滤计数。

| 上层参数 | 本算子参数 / 属性 | 说明 |
| --- | --- | --- |
| colors: Tensor [B,C,3,N] | colors | 需连续 |
| means2d: Tensor [B,C,2,N] | means2d | 仅网格模式参与计算 |
| exposure: Tensor | exposure | 全 batch / 相机共享 |
| cnt: Tensor [B,C] (INT32) | cnt | 上游投影 / 过滤输出的有效高斯计数 |
| width / height: int | 属性 width / height | 图像宽高（像素） |
| clamp: bool = True | 属性 clamp（0/1） | 默认开（渲染配置 clamp_color_min） |
| 返回 output（autograd 保存 index / weight 供 backward） | output / index / weight | - |

不支持参数的处理：`means2d` / `cnt` 不反传梯度（上游无需求）；非连续输入由调用方负责 `.contiguous()`。

## 5. 实现方案

### 5.1 分层结构

| 层 | 职责 | 关键文件 |
| --- | --- | --- |
| aclnn 适配（op_api） | 接口定义、参数校验、workspace 计算 | 算子包构建生成（本仓无手写 op_api 目录） |
| op_host | 属性解析、全局 / 网格模式判定、Tiling 计算 | kernels/color_exposure_fwd/op_host/arch35/color_exposure_fwd.cpp、color_exposure_fwd_tiling.h |
| op_kernel | GM↔UB 搬运与计算 | op_kernel/arch35/color_exposure_fwd.cpp、apply_grid_exposure_vf.h |
| 框架绑定 | PyTorch / aten 桥接 | gauss_splat/csrc/ColorExposureFwd.cpp |

### 5.2 Tiling 策略

- 切分维度与顺序：以高斯维 N 为唯一切分维度（batch × camera 维在 kernel 内双层循环）；N 按 ALIGN_VALUE = 8 向上取整为 totalTaskNum，尾部填充数 tailNum = totalTaskNum − N；
- tile 大小确定方式：
  - 每任务队列预算 bufferSize（float 数 / 任务）：网格模式 $3 \times 2 \times 2 + 8 = 20$（colors / output 队列各 2×3，means2d 队列 2×2，index 3，grey 1）；全局模式 $3 \times 2 \times 2 = 12$；
  - `taskNumPerLoop = (ubTotalSize × 0.8 − expSize) / (bufferSize × 4)`，向上按 8 对齐；
  - `expSize = (G_X·G_Y·G_Z × 12 + B·C) × 4`（全局模式网格按 1×1×1 计；B·C 为 cnt 常驻缓冲）；网格模式额外预留 DCACHE_SIZE = 32KB 并调用 `SetDynUBufSize(ubTotalSize − 32KB)`，该区域专供 SIMT VF DCache 使用，避免与 UB 队列冲突导致 VF 越界；
- 多核 / 多 block 分配：blockDim = AIV 核数；taskNumPerScore = (totalTaskNum / blockDim / 8) × 8 为短核任务数，长核 taskNumPerLcore = taskNumPerScore + 8；numScore / numLcore 按余数分配，前 numLcore 个 block 为长核；退化场景（taskNumPerScore 或 taskNumPerLcore 为 0）时收缩 blockDim；
- 网格规模上限：tiling 硬校验 $G_X G_Y G_Z \times 12 \le 24576$（MAX_GRID_DIMS），即 $G_X G_Y G_Z \le 2048$、最大网格 16×16×8，超出直接 tiling 失败；UB 预算 $G_X G_Y G_Z \le 3549$（约 15×15×15，Ascend 950 UB = 253952 B）为非绑定约束（硬校验先行生效）；
- UB / L1 / L0 空间规划：不使用 L1 / L0；UB 内 = 队列区（0.8 预算）+ exposure 常驻区 + cnt 常驻区（$B \times C \times 4$ B）+ 32KB DCache 预留（网格模式）；16×16×8 时 exposure 常驻区 96KB，taskNumPerLoop ≈ 904（Ascend 950），运行正常；
- 尾块与非对齐处理：尾块 tile 用 DataCopyPad 掩码搬运，validTaskNum = taskNumPerCurLoop − tailNum（仅末 tile）；cnt 截断同样收敛到该掩码路径（见 5.3）；
- TilingKey 设计：恒为 1；模式分支在 kernel 内按 tiling 字段（useGrid / clamp）判断，不做 key 分裂。

### 5.3 Kernel 实现

- 数据搬运路径：exposure 与 cnt（$B \times C$ INT32）各一次 GM→UB 常驻（DataCopyPad，V→MTE2 事件同步）；colors / means2d 按 tile GM→UB；output（及网格模式 index / weight）按 tile UB→GM；
- 流水编排：CopyIn → Compute → CopyOut 事件同步串行推进，队列双缓冲（bufferNum = 2）；外层 batch × camera × taskLoop 三层循环；
- **cnt 过滤实现**：每个 (b, c) 先读取 `cnt_ = cnt[b × C + c]`；每 tile 由 ComputingTaskNum 判定 `taskStart < cnt_`，不满足则 break 跳过该相机剩余 tile；cnt 落在 tile 内部（taskStart < cnt < taskEnd）时按 `validTaskNum = cnt − taskStart` 截断并强制走尾块掩码路径（DataCopyPad 掩码搬运 + VEC / VF 以 validTaskNum 掩码计算），实现"块内提前结束"；
- 使用的计算原语：
  - 全局模式：纯 VEC 通路（ApplyGlobalExposure），Reg 级 Muls / Add / Maxs，64 lane 一批；
  - 网格模式：PrepareCalculate（VEC 通路，64 lane 一批向量化计算灰度 → 坐标 × fp32 倒数 → clamp01 → ×(G−1) → trunc 取整 → 端点闭包，将 wx / wy / wz / index 写回 UB 缓冲），随后 `asc_vf_call` 启动 ApplyGridExposureVF（SIMT VF，dim3{1024, 1, 1}，每线程处理一个高斯：8 角 × 3 次插值重建 12 参数 → 仿射 → clamp）；
- 循环与边界处理：每 tile 由 ComputingTaskNum 得 taskNumPerCurLoop、isTailBlock 与 validTaskNum（cnt 截断后的有效任务数）；VEC / VF 均以 validTaskNum 掩码处理尾部；
- 特殊场景实现：全局模式不启动 VF、不分配 means2d / index / grey 缓冲；identity 等特殊 exposure 无专门快路径（统一走通用通路）；cnt = 0 时该相机所有 tile 直接 break。

### 5.4 特殊场景处理

| 场景 | 处理方案 |
| --- | --- |
| 小 shape（N = 1） | totalTaskNum = 8，blockDim 收缩 + 尾块掩码覆盖 |
| 非对齐 shape（N % 8 ≠ 0） | tailNum 掩码 + DataCopyPad，无单独分支 |
| cnt = 0（相机级全跳过） | taskLoop 首块 break，无任何读写 |
| cnt 非 8 对齐 / 落在 tile 中间 | validTaskNum 截断 + 强制尾块掩码路径 |
| 大 shape / 多核 | taskLoop 分块循环（已验证 N = 100000），长短核均衡负载 |
| 多相机（C > 1） | camera 维循环复用同一套队列 / 常驻缓冲，cnt 逐相机读取 |
| 空 tensor | 不支持（见 3.5） |

## 6. 精度与性能

### 6.1 精度标准与验证方法

- golden 来源：tests/ops/test_color_exposure.py 内置 CPU 参考实现（fp32，逐位对齐 kernel 的乘倒数 / trunc / 端点闭包次序；按 cnt 掩码，无效位置置 0）；NPU 侧 C++ 用例（tests/kernels/test_color_exposure_fwd.cpp）内置同构 CPU golden（cnt 混合模式：0 / 1 / N/2 / N）；
- 对比方法：index 用 torch.equal 严格逐位对比；output / weight 转 fp32 后 torch.allclose；
- 误差阈值：output / weight rtol = atol = 1e-5；index 零容差；
- 混合精度说明：无。全程 fp32，灰度系数（0.299 / 0.587 / 0.114）与坐标倒数均为 fp32 常量。

### 6.2 性能目标与测算

| 指标 | 理论值 | 目标值 | 实测值 | 达成率 |
| --- | --- | --- | --- | --- |
| 耗时 | 访存受限：每有效高斯读 3（全局）/ 5（网格）float、写 3 / 9 float；无效尾部高斯零开销 | 贴近同平台拷贝类算子 | 待标定 | - |
| 带宽利用率 | - | 同上 | 待标定 | - |

- 瓶颈分析：GM 搬运为主；网格模式另有每高斯 8 角 × 12 参数插值的 VF 计算开销；
- 优化手段：队列双缓冲、VEC 64-lane 向量化预处理、index / weight 显式输出避免反向重算插值、cnt 过滤跳过无效高斯。

## 7. 测试设计

### 7.1 功能用例

- NPU C++ 用例（tests/kernels/test_color_exposure_fwd.cpp，13 例）：global_basic（B=1, N=4096）、global_identity_exact（identity exposure，输出逐位等于输入）、global_clamp_tail（clamp + 尾块 N=123）、grid_aligned（7×7×8）、grid_clamp_tail（2×2×2 最小网格 + clamp）、grid_tiny（N=7）、grid_5d_exposure（(4,5,6,4,3) 排布）、grid_large（N=100000, 12×12×8）、grid_max_16x16x8（B=2, C=2, N=517, 16×16×8 硬校验上限）、global_multi_camera（B=2, C=3）、grid_multi_camera_cnt_mixed（B=2, C=3, cnt 混合）、grid_cnt_mixed_tail（B=2, C=2, N=123, cnt 混合）、grid_cnt_mixed_large（B=1, C=4, N=100000, cnt 混合）；cnt 混合模式按 (b, c) 平坦下标取 0 / 1 / N/2 / N，覆盖整相机跳过、单高斯、块内截断、全量四种路径；
- CPU golden 对拍（tests/ops/test_color_exposure.py）：test_fwd_global_random、test_fwd_grid_random（均含多相机 C ∈ {1, 2, 3}）、test_fwd_cnt_filter（cnt 全 0 / 混合 / 跨 taskLoop 边界）、test_fwd_grid_5d_exposure、test_fwd_grid_max_16x16x8（16×16×8 硬校验上限）。

### 7.2 边界用例

- test_fwd_boundary：注入 7 组边界坐标（(0,0)、(W−1,H−1)、(W,H)、出界正 / 负、轴端点）× 7 组边界颜色（纯黑、超界 2.0、负 −2.0、灰度系数色等），覆盖 clamp01、trunc、端点闭包、灰度饱和全部分支（B=2, C=2）；
- 尾块与非对齐：N = 7 / 13 / 123 / 257；最小网格 2×2×2；多 batch / camera B ∈ {1, 2, 3}、C ∈ {1, 2, 3}。

### 7.3 精度用例

- 全部功能用例均做 golden 对拍：output / weight allclose(rtol=atol=1e-5)、index torch.equal；
- 数值钉死用例：identity exposure → output 逐位等于 colors；$A$ 的指定元素置 k → 对应输出通道逐元素等于 $k \times$ 源通道；
- cnt 过滤钉死：cnt = 0 的相机 output 全 0；cnt < N 时 [cnt, N) 区间 output / index / weight 全 0。

### 7.4 性能与泛化用例

- shape 扫描：N ∈ {1, 7, 13, 123, 257, 4096, 100000}，B ∈ {1, 2, 3}，C ∈ {1, 2, 3, 4}，网格 2×2×2 ~ 16×16×8（含硬校验上限），clamp ∈ {0, 1}，exposure 4d / 5d 排布，cnt ∈ {全量, 混合, 全 0}；
- 性能基线：待补充（当前以功能与精度验证为主）。

## 8. 风险与备选方案

| 风险 | 影响 | 应对措施 / 备选方案 |
| --- | --- | --- |
| cnt 语义越界（cnt > N 或负值） | cnt > N 按全量处理（结果正确）；负值经无符号比较行为未定义 | 约定 $0 \le cnt[b,c] \le N$（上游投影 / 过滤保证）；如需防御可在 aclnn 适配层加校验 |
| 输出未预清零 | 无效高斯位置残留旧值 | 语义约定输出 tensor 需预清零；上层封装已用 zeros 分配；aclnn 裸调用者需自查 |
| 网格任一维 = 1 | 端点闭包 ix−1 = −1，下标回退到相邻单元，结果非预期 | 约束各维 ≥ 2（继承 CUDA 参考实现）；如需支持须在闭包分支加 ix > 0 保护并同步 golden |
| 网格规模超限 | tiling 直接失败（硬校验 $G_X G_Y G_Z \times 12 \le 24576$）；UB 预算超限时 taskNumPerLoop 下溢、结果错误 | 约束 $G_X G_Y G_Z \le 2048$（即最大 16×16×8，硬校验兜底）；UB 预算上限 3549 非绑定；更大网格需调大 MAX_GRID_DIMS 并拆分曝光场或改造为分块搬运 |
| DCACHE 预留 32KB 与硬件实际不符 | 预留不足 → SIMT VF 访存越界；过大 → UB 浪费 | 32KB 为保守经验值，需硬件实测校准 |
| NaN + clamp | max(NaN, 0) 平台相关，结果未定义 | 声明不支持；如需支持须加 NaN 检查（有性能代价） |
| fp32 乘倒数 vs 除法 | 与 CUDA 参考存在 ~1 ulp 差异，index 边界处可能差 1 | golden 按本实现复刻；跨平台对拍需容忍该差异 |

## 9. 附录

### 9.1 术语表

| 术语 | 含义 |
| --- | --- |
| 12 参数仿射 / $[A\|b]$ | 3×4 行增广矩阵：$A_{ij} = exposure[4i+j]$，$b_i = exposure[4i+3]$，out = Ac + b |
| cnt 过滤 | 仅计算每 (b, c) 前 cnt[b, c] 个高斯（紧凑布局），其余位置不写回 |
| 端点闭包 | 网格坐标落在最后单元（ix = G−1）时回退 (ix, wx) ← (G−2, 1)，保证 8 角插值不越界 |
| SIMT VF | Ascend 950 arch35 的 SIMT 向量引擎（asc_vf_call 启动，1024 线程） |
| 长短核 | 多核负载均衡策略：长核多带 8 个任务，其余为短核 |
| taskNumPerLoop | 单核内单次 tile 处理的任务数（高斯数），由 UB 预算推出 |
| validTaskNum | tile 内实际参与计算的任务数：尾块扣除 tailNum，cnt 截断时取块内剩余有效数 |

### 9.2 参考资料

- docs/zh/kernels/aclnnColorExposureBwd.md：配套反向算子设计说明
- CANN Ascend C 编程指南（TPipe / Tiling / DataCopyPad / asc_vf_call / Reg 向量指令）
- gsplat CUDA 参考实现

### 9.3 变更记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | 初稿 |
| v1.1 | 2026-11-23 | - | 输入 / 输出从 [B, C, N]（batch×camera 融合）重构为 [B, C, D, N]（显式相机维）；新增 INT32 输入 cnt（逐 (b, c) 有效高斯计数），kernel 按 cnt 跳过无效尾部高斯；index / weight 形状同步变为 [B, C, 3, N]；更新 Tiling（cnt 常驻缓冲、三层循环）与测试设计；新增最大网格 16×16×8（MAX_GRID_DIMS 硬校验上限）边界用例 |
