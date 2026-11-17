# ColorExposureFwd 设计说明

[📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/color_exposure_fwd/op_kernel/arch35/color_exposure_fwd.cpp)

## 1. 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | ColorExposureFwd |
| 所属仓 / 目录 | gauss-splat / kernels/color_exposure_fwd |
| 算子类型 | 自定义（逐高斯颜色仿射变换） |
| aclnn 接口名 | aclnnColorExposureFwd |
| 文档版本 | v1.0 |
| 作者 | - |
| 评审人 | - |
| 最后更新 | 2026-11-16 |

## 2. 需求背景与目标

### 2.1 业务场景

3D Gaussian Splatting（3DGS）多视角训练中，不同视角相机的曝光 / 白平衡差异会使同一高斯球在各图像中呈现系统性颜色漂移，影响重建-渲染一致性。为此在 SH 颜色计算之后、光栅化之前，对逐高斯颜色施加一组可学习的曝光仿射变换（3×4 行增广矩阵 $[A|b]$，共 12 参数），并与高斯属性联合优化。曝光参数支持两种粒度：

- **全局模式**：每个视角（batch）一组 12 参数，批内所有高斯共享；
- **网格模式**：$(G_X, G_Y, G_Z, 12)$ 参数场，按高斯投影坐标 $(x, y)$ 与颜色灰度做 8 角三线性插值，建模空间变化的曝光。

曝光参数为可学习参数，训练时需要梯度，因此前向（本算子）与反向（ColorExposureBwd）成对交付。

### 2.2 对标算子

对标 gsplat 生态 CUDA 参考实现的 `color_exposure`（逐高斯 12 参数仿射 + 网格曝光场），数学语义一致。与本实现的差异：

- 硬件通路：本实现基于 Ascend 950（arch35）AIV 向量核 + SIMT VF；坐标→网格索引的映射在 VEC 通路向量化完成，8 角插值与仿射应用在 SIMT VF 完成；
- 本实现前向显式输出插值中间量 `index` / `weight`，供反向算子直接复用，避免反向重算插值；
- 坐标映射按 fp32 乘倒数 $x \cdot \mathrm{fl32}(1/W)$ 实现而非除法，与 CUDA 版存在约 1 ulp 差异，golden 与精度用例按本实现逐位复刻；
- 网格维数 = 1 时端点闭包产生 $-1$ 下标的行为继承自 CUDA 参考实现（见第 8 节）。

### 2.3 目标

- 功能目标：
  - 覆盖：全局 / 网格两种模式；`clamp` 非负截断（可选）；网格模式输出 `index` / `weight`；`exposure` 支持 $(G_X, G_Y, G_Z, 12)$ 与 $(G_X, G_Y, G_Z, 4, 3)$ 两种排布；
  - 明确不做：`means2d` 的梯度、逐 batch 独立 exposure、非连续输入、低精度 dtype。
- 性能目标：访存受限型算子，GM 带宽利用率贴近同平台拷贝类算子水平（每高斯读 3~5 float、写 3~9 float；实测待标定）。
- 精度目标：`output` / `weight` 对 CPU golden 的误差 rtol = atol = 1e-5；`index` 逐位一致。
- 交付范围：dtype FLOAT（`index` INT32）、format ND、$B, N \ge 1$、网格各维 ≥ 2 且网格字节受 UB 预算约束（见 5.2）；仅支持 Ascend 950PR / 950DT。

## 3. 算子定义

### 3.1 功能描述

对 $[B, 3, N]$ 的逐高斯颜色 `colors` 按 `exposure` 做仿射变换，输出 $[B, 3, N]$ 的 `output`：

- 全局模式（`exposure` 元素数 = 12）：批内所有高斯共用同一组 12 参数（3×4 行增广 $[A|b]$），$out = Ac + b$；
- 网格模式（`exposure` 元素数 > 12，形状 $(G_X, G_Y, G_Z, 12)$ 或 $(G_X, G_Y, G_Z, 4, 3)$）：以 $[B, 2, N]$ 的投影像素坐标 `means2d` 与颜色灰度为插值自变量，先在 $G_X \times G_Y \times G_Z$ 网格上 8 角三线性插值出该高斯的 12 参数，再做仿射变换；
- `clamp` = 1 时输出逐通道 $\max(\cdot, 0)$；
- 网格模式额外输出插值下标 `index`（INT32）与三线权重 `weight`（FLOAT），形状均为 $[B, 3, N]$，供反向算子复用；全局模式下 `index` / `weight` 传空 tensor（numel = 0）。

### 3.2 数学表达式

12 个参数按 3×4 行增广矩阵 $[A|b]$ 解释：$A_{ij} = exposure[4i+j]$，$b_i = exposure[4i+3]$（$i, j \in \{0, 1, 2\}$）。

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
- 全部中间量为 fp32（灰度系数、倒数、权重均为 fp32 常量 / fp32 运算），无混合精度。

### 3.3 输入、输出与属性

| 名称 | 方向 | 含义 | dtype | shape | format | 是否可选 |
| --- | --- | --- | --- | --- | --- | --- |
| colors | 输入 | 高斯颜色（SH 前向结果） | FLOAT | [B, 3, N] | ND | 否 |
| means2d | 输入 | 高斯投影像素坐标 | FLOAT | [B, 2, N] | ND | 网格模式必选；全局模式不参与计算（传合法非空 tensor） |
| exposure | 输入 | 曝光参数 | FLOAT | 全局：[12] / (4,3) / (1,1,1,12)；网格：(GX,GY,GZ,12) / (GX,GY,GZ,4,3) | ND | 否 |
| width | 属性 | 图像宽 W（像素） | int32 | - | - | 否 |
| height | 属性 | 图像高 H（像素） | int32 | - | - | 否 |
| clamp | 属性 | 输出是否逐通道非负截断 | bool（0/1） | - | - | 否 |
| output | 输出 | 曝光校正后颜色 | FLOAT | [B, 3, N] | ND | 否 |
| index | 输出 | 网格插值下标 (ix, iy, iz) | INT32 | [B, 3, N] | ND | 网格模式必选；全局模式传空 tensor（numel = 0） |
| weight | 输出 | 三线插值权重 (wx, wy, wz) | FLOAT | [B, 3, N] | ND | 同 index |

### 3.4 支持矩阵

| 维度 | 支持范围 | 不支持项及原因 |
| --- | --- | --- |
| 产品 | Ascend 950PR / 950DT（arch35） | 其他产品不支持：算子仅适配 arch35（SIMT VF / Reg 向量指令），未编译其他架构版本 |
| dtype | FLOAT（index 为 INT32） | fp16 / bf16 / 整型等不支持：颜色通路全程 fp32，无混合精度需求 |
| format | ND | 其他 format 不支持：kernel 按紧凑布局直接寻址 |
| shape | colors [B,3,N]、means2d [B,2,N]，B ≥ 1、N ≥ 1；exposure numel = 12 或 (GX,GY,GZ,12) / (GX,GY,GZ,4,3)，网格各维 ≥ 2；网格字节数 $G_X G_Y G_Z \times 48\,B$ 受 UB 预算约束（见 5.2） | 空 tensor（shape 含 0）、0 维输入不支持；网格维 = 1 见第 8 节 |
| 广播 | 不支持 | exposure 为全 batch 共享的固定语义参数，非广播 |

### 3.5 边界与异常场景

| 场景 | 预期行为 |
| --- | --- |
| 空 tensor（shape 含 0） | 不支持：必选输入 / 输出非空；全局模式 index / weight 传空 tensor（numel = 0）表示不输出，kernel 不分配、不回写 |
| 标量 / 0 维输入 | 不支持，参数校验报错（161002） |
| 非连续输入 | 不支持：kernel 按紧凑布局寻址，需调用方保证连续（上层封装不自动转连续） |
| 输入含 NaN / Inf | 未定义：clamp 的 $\max(\cdot, 0)$ 对 NaN 行为平台相关，kernel 不做特殊检查，不建议注入 |
| shape 或 dtype 不合法 | 报错 161001（空指针）/ 161002（dtype、shape 校验失败），见 4.2 |
| workspace 不足 | 不涉及用户侧 workspace：由 aclnn 框架按 GetWorkspaceSize 返回值管理（本算子仅申请系统 workspace） |

## 4. 接口设计

### 4.1 aclnn 接口与调用流程

```text
aclnnColorExposureFwdGetWorkspaceSize(colors, means2d, exposure, width, height, clamp,
                                      output, index, weight, workspaceSize, executor)
aclnnColorExposureFwd(workspace, workspaceSize, executor, stream)
```

- workspace 计算规则：仅使用系统 workspace（LibApiWorkSpaceSize），无算子自定义 workspace；用户按 `GetWorkspaceSize` 返回值分配即可，空 tensor 场景同规则。
- 执行器复用规则：每次 `GetWorkspaceSize` 完成参数校验并创建执行器，`aclnnColorExposureFwd` 执行一次；同一输入重新调用需重新走两段式流程（标准 aclnn 语义）。

### 4.2 参数校验与报错

| 校验项 | 规则 | 报错码 / 日志关键字 |
| --- | --- | --- |
| 空指针 | 所有输入 / 输出 tensor 指针及 workspaceSize / executor 非空 | 161001，"param ... is null" |
| dtype 组合 | colors / means2d / exposure / output / weight 为 FLOAT，index 为 INT32 | 161002 |
| shape 一致性 | colors [B,3,N] 与 means2d [B,2,N] 的 B、N 一致；exposure numel ≥ 12；网格模式 index / weight 必须为 [B,3,N]，全局模式传空 tensor | 161002 |
| format | ND | 161002 |

（校验逻辑由算子包构建生成的 aclnn 适配层完成，与仓内其他算子一致。）

### 4.3 上层框架映射

无 torch 原生算子对应。上层链路：`gauss_splat.ops.color_exposure`（torch.autograd.Function）→ `gauss_splat/csrc/ColorExposureFwd.cpp`（EXEC_NPU_CMD 直调，不改变 stride）→ aclnnColorExposureFwd。

| 上层参数 | 本算子参数 / 属性 | 说明 |
| --- | --- | --- |
| colors: Tensor [B,3,N] | colors | 需连续 |
| means2d: Tensor [B,2,N] | means2d | 仅网格模式参与计算 |
| exposure: Tensor | exposure | 全 batch 共享 |
| width / height: int | 属性 width / height | 图像宽高（像素） |
| clamp: bool = False | 属性 clamp（0/1） | 默认关 |
| 返回 output（autograd 保存 index / weight 供 backward） | output / index / weight | - |

不支持参数的处理：`means2d` 不反传梯度（上游无需求）；非连续输入由调用方负责 `.contiguous()`。

## 5. 实现方案

### 5.1 分层结构

| 层 | 职责 | 关键文件 |
| --- | --- | --- |
| aclnn 适配（op_api） | 接口定义、参数校验、workspace 计算 | 算子包构建生成（本仓无手写 op_api 目录） |
| op_host | 属性解析、全局 / 网格模式判定、Tiling 计算 | kernels/color_exposure_fwd/op_host/arch35/color_exposure_fwd.cpp、color_exposure_fwd_tiling.h |
| op_kernel | GM↔UB 搬运与计算 | op_kernel/arch35/color_exposure_fwd.cpp、apply_grid_exposure_vf.h |
| 框架绑定 | PyTorch / aten 桥接 | gauss_splat/csrc/ColorExposureFwd.cpp |

### 5.2 Tiling 策略

- 切分维度与顺序：以高斯维 N 为唯一切分维度（batch 维在 kernel 内循环）；N 按 ALIGN_VALUE = 8 向上取整为 totalTaskNum，尾部填充数 tailNum = totalTaskNum − N；
- tile 大小确定方式：
  - 每任务队列预算 bufferSize（float 数 / 任务）：网格模式 $3 \times 2 \times 2 + 8 = 20$（colors / output 队列各 2×3，means2d 队列 2×2，index 3，grey 1）；全局模式 $3 \times 2 \times 2 = 12$；
  - `taskNumPerLoop = (ubTotalSize × 0.8 − expSize) / (bufferSize × 4)`，向上按 8 对齐；
  - `expSize = G_X·G_Y·G_Z × 12 × 4`（全局模式按 1×1×1 计）；网格模式额外预留 DCACHE_SIZE = 32KB 并调用 `SetDynUBufSize(ubTotalSize − 32KB)`，该区域专供 SIMT VF DCache 使用，避免与 UB 队列冲突导致 VF 越界；
- 多核 / 多 block 分配：blockDim = AIV 核数；taskNumPerScore = (totalTaskNum / blockDim / 8) × 8 为短核任务数，长核 taskNumPerLcore = taskNumPerScore + 8；numScore / numLcore 按余数分配，前 numLcore 个 block 为长核；退化场景（taskNumPerScore 或 taskNumPerLcore 为 0）时收缩 blockDim；
- UB / L1 / L0 空间规划：不使用 L1 / L0；UB 内 = 队列区（0.8 预算）+ exposure 常驻区 + 32KB DCache 预留（网格模式）；Ascend 950 UB = 253952 B，网格模式下需 $G_X G_Y G_Z \le 3549$（约 15×15×15）；
- 尾块与非对齐处理：尾块 tile 用 DataCopyPad 掩码搬运，validTaskNum = taskNumPerCurLoop − tailNum（仅末 tile）；
- TilingKey 设计：恒为 1；模式分支在 kernel 内按 tiling 字段（useGrid / clamp）判断，不做 key 分裂。

### 5.3 Kernel 实现

- 数据搬运路径：exposure 一次 GM→UB 常驻（DataCopyPad，V→MTE2 事件同步）；colors / means2d 按 tile GM→UB；output（及网格模式 index / weight）按 tile UB→GM；
- 流水编排：CopyIn → Compute → CopyOut 事件同步串行推进，队列双缓冲（bufferNum = 2）；外层 batch × taskLoop 双循环；
- 使用的计算原语：
  - 全局模式：纯 VEC 通路（ApplyGlobalExposure），Reg 级 Muls / Add / Maxs，64 lane 一批；
  - 网格模式：PrepareCalculate（VEC 通路，64 lane 一批向量化计算灰度 → 坐标 × fp32 倒数 → clamp01 → ×(G−1) → trunc 取整 → 端点闭包，将 wx / wy / wz / index 写回 UB 缓冲），随后 `asc_vf_call` 启动 ApplyGridExposureVF（SIMT VF，dim3{1024, 1, 1}，每线程处理一个高斯：8 角 × 3 次插值重建 12 参数 → 仿射 → clamp）；
- 循环与边界处理：每 tile 由 ComputingTaskNum 得 taskNumPerCurLoop 与 isTailBlock；VEC / VF 均以 validTaskNum 掩码处理尾部；
- 特殊场景实现：全局模式不启动 VF、不分配 means2d / index / grey 缓冲；identity 等特殊 exposure 无专门快路径（统一走通用通路）。

### 5.4 特殊场景处理

| 场景 | 处理方案 |
| --- | --- |
| 小 shape（N = 1） | totalTaskNum = 8，blockDim 收缩 + 尾块掩码覆盖 |
| 非对齐 shape（N % 8 ≠ 0） | tailNum 掩码 + DataCopyPad，无单独分支 |
| 大 shape / 多核 | taskLoop 分块循环（已验证 N = 100000），长短核均衡负载 |
| 空 tensor | 不支持（见 3.5） |

## 6. 精度与性能

### 6.1 精度标准与验证方法

- golden 来源：tests/ops/test_color_exposure.py 内置 CPU 参考实现（fp32，逐位对齐 kernel 的乘倒数 / trunc / 端点闭包次序）；NPU 侧 C++ 用例（tests/kernels/test_color_exposure_fwd.cpp）内置同构 CPU golden；
- 对比方法：index 用 torch.equal 严格逐位对比；output / weight 转 fp32 后 torch.allclose；
- 误差阈值：output / weight rtol = atol = 1e-5；index 零容差；
- 混合精度说明：无。全程 fp32，灰度系数（0.299 / 0.587 / 0.114）与坐标倒数均为 fp32 常量。

### 6.2 性能目标与测算

| 指标 | 理论值 | 目标值 | 实测值 | 达成率 |
| --- | --- | --- | --- | --- |
| 耗时 | 访存受限：每高斯读 3（全局）/ 5（网格）float、写 3 / 9 float | 贴近同平台拷贝类算子 | 待标定 | - |
| 带宽利用率 | - | 同上 | 待标定 | - |

- 瓶颈分析：GM 搬运为主；网格模式另有每高斯 8 角 × 12 参数插值的 VF 计算开销；
- 优化手段：队列双缓冲、VEC 64-lane 向量化预处理、index / weight 显式输出避免反向重算插值。

## 7. 测试设计

### 7.1 功能用例

- NPU C++ 用例（tests/kernels/test_color_exposure_fwd.cpp，8 例）：global_basic（B=1, N=4096）、global_identity_exact（identity exposure，输出逐位等于输入）、global_clamp_tail（clamp + 尾块 N=123）、grid_aligned（7×7×8）、grid_clamp_tail（2×2×2 最小网格 + clamp）、grid_tiny（N=7）、grid_5d_exposure（(4,5,6,4,3) 排布）、grid_large（N=100000, 12×12×8）；
- CPU golden 对拍（tests/ops/test_color_exposure.py）：test_fwd_global_random、test_fwd_grid_random、test_fwd_grid_5d_exposure。

### 7.2 边界用例

- test_fwd_boundary：注入 7 组边界坐标（(0,0)、(W−1,H−1)、(W,H)、出界正 / 负、轴端点）× 7 组边界颜色（纯黑、超界 2.0、负 −2.0、灰度系数色等），覆盖 clamp01、trunc、端点闭包、灰度饱和全部分支；
- 尾块与非对齐：N = 7 / 13 / 123 / 257；最小网格 2×2×2；多 batch B = 2 / 3。

### 7.3 精度用例

- 全部功能用例均做 golden 对拍：output / weight allclose(rtol=atol=1e-5)、index torch.equal；
- 数值钉死用例：identity exposure → output 逐位等于 colors；$A$ 的指定元素置 k → 对应输出通道逐元素等于 $k \times$ 源通道。

### 7.4 性能与泛化用例

- shape 扫描：N ∈ {1, 7, 13, 123, 257, 4096, 100000}，B ∈ {1, 2, 3}，网格 2×2×2 ~ 12×12×8，clamp ∈ {0, 1}，exposure 4d / 5d 排布；
- 性能基线：待补充（当前以功能与精度验证为主）。

## 8. 风险与备选方案

| 风险 | 影响 | 应对措施 / 备选方案 |
| --- | --- | --- |
| 网格任一维 = 1 | 端点闭包 ix−1 = −1，下标回退到相邻单元，结果非预期 | 约束各维 ≥ 2（继承 CUDA 参考实现）；如需支持须在闭包分支加 ix > 0 保护并同步 golden |
| 网格字节超 UB 预算 | taskNumPerLoop 下溢，结果错误 | 约束 $G_X G_Y G_Z \le 3549$（Ascend 950）；更大网格需拆分曝光场或改造为分块搬运 |
| DCACHE 预留 32KB 与硬件实际不符 | 预留不足 → SIMT VF 访存越界；过大 → UB 浪费 | 32KB 为保守经验值，需硬件实测校准 |
| NaN + clamp | max(NaN, 0) 平台相关，结果未定义 | 声明不支持；如需支持须加 NaN 检查（有性能代价） |
| fp32 乘倒数 vs 除法 | 与 CUDA 参考存在 ~1 ulp 差异，index 边界处可能差 1 | golden 按本实现复刻；跨平台对拍需容忍该差异 |

## 9. 附录

### 9.1 术语表

| 术语 | 含义 |
| --- | --- |
| 12 参数仿射 / $[A|b]$ | 3×4 行增广矩阵：$A_{ij} = exposure[4i+j]$，$b_i = exposure[4i+3]$，out = Ac + b |
| 端点闭包 | 网格坐标落在最后单元（ix = G−1）时回退 (ix, wx) ← (G−2, 1)，保证 8 角插值不越界 |
| SIMT VF | Ascend 950 arch35 的 SIMT 向量引擎（asc_vf_call 启动，1024 线程） |
| 长短核 | 多核负载均衡策略：长核多带 8 个任务，其余为短核 |
| taskNumPerLoop | 单核内单次 tile 处理的任务数（高斯数），由 UB 预算推出 |

### 9.2 参考资料

- docs/zh/kernels/aclnnColorExposureBwd.md：配套反向算子设计说明
- CANN Ascend C 编程指南（TPipe / Tiling / DataCopyPad / asc_vf_call / Reg 向量指令）
- gsplat CUDA 参考实现

### 9.3 变更记录

| 版本 | 日期 | 修改人 | 修改内容 |
| --- | --- | --- | --- |
| v1.0 | 2026-11-16 | - | 初稿 |
