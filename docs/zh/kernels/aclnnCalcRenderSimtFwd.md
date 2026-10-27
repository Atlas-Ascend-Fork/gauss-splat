# aclnnCalcRenderSimtFwd

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_simt_fwd/op_kernel/arch35/calc_render_simt_fwd_apt.cpp)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    ×    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    ×    |
| Atlas 200I/500 A2 推理产品                  |    ×    |
| Atlas 推理系列产品                          |    ×    |
| Atlas 训练系列产品                          |    ×    |

## 功能说明

- 接口功能：`calc_render_simt_fwd`计算SIMT渲染的前向过程，对每个像素按深度排序逐高斯球执行alpha blending累积，输出颜色、深度，并保存反向传播所需的中间结果（`lastCumsum`、`error`、`lastIds`）。该算子与`calc_render_simt_bwd`成对使用。
- 计算公式：

逐像素按排序顺序遍历高斯球，记像素坐标为 $(p_x, p_y)$，当前高斯球参数为 $(\mu_x, \mu_y, c_0, c_1, c_2, o, c_r, c_g, c_b, d)$，定义位移（二次型对 $dx, dy$ 同时变号不变，故符号约定与反向一致）：

$$
dx = \mu_x - p_x, \quad dy = \mu_y - p_y
$$

$$
w = \exp\left(-\frac{1}{2}\left(c_0 dx^2 + 2 c_1 dx\,dy + c_2 dy^2\right)\right)
$$

$$
\alpha = \min(o \cdot w,\ 0.999)
$$

当 $\alpha \ge \frac{1}{255}$ 时，令 $T_{i-1}$ 为累加前的透射率，计算 $T_i = T_{i-1}(1-\alpha)$：

- 若 $T_i > 10^{-4}$，将该高斯球的颜色/深度贡献加入累加：

$$
color = \sum_{i} \alpha_i T_{i-1} c_i, \quad depth = \sum_{i} \alpha_i T_{i-1} d_i
$$

- 否则该像素达到 per-pixel done 条件，记录 `lastIds` 后结束遍历（该高斯球及其后高斯球不再累加）。

两种情况均以 Kahan 补偿更新对数域累积透射率（`lastCumsum`）：

$$
s \mathrel{+}= \ln(1-\alpha), \quad T_{i-1} = e^{s}
$$

前向同时输出：

- `lastCumsum`：对数域累积透射率 $s = \sum \ln(1-\alpha)$；
- `error`：Kahan补偿误差项（数学真值恒为0）；
- `lastIds`：逐像素最后一个贡献高斯球在排序列表中的位置（无贡献像素为0）。

这些中间输出与输入 `gs`、`tileCoords`、`offsets`、`gsIds` 一同作为 `calc_render_simt_bwd` 的输入；其中 `depth` 为 (tileNum, 1024)，反向时需扩展为 (1, tileNum, 1024) 作为 `vDepth`。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnCalcRenderSimtFwdGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnCalcRenderSimtFwd"接口执行计算。

```Cpp
aclnnStatus aclnnCalcRenderSimtFwdGetWorkspaceSize(
    const aclTensor *gs,
    const aclTensor *tileCoords,
    const aclTensor *offsets,
    const aclTensor *gsIds,
    const aclTensor *color,
    const aclTensor *depth,
    const aclTensor *lastCumsum,
    const aclTensor *error,
    const aclTensor *lastIds,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCalcRenderSimtFwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnCalcRenderSimtFwdGetWorkspaceSize

- **参数说明：**
  
  <table style="width: 100%">
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gs</td>
      <td>输入</td>
      <td>高斯球参数，每行包含10个字段（位置、锥面、透明度、颜色、深度）。</td>
      <td>不支持空tensor。第二维必须为10。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileCoords</td>
      <td>输入</td>
      <td>逐tile像素坐标信息。</td>
      <td>不支持空tensor。第二维必须为2，第三维固定为1024。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 2, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>核间调度偏移表，三段布局：[向量核累积调度数 | tile调度序 | 累计高斯偏移]。</td>
      <td>不支持空tensor。必须为1维，长度等于向量核数 + 2*tileNum。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(vectorNum + 2*tileNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gsIds</td>
      <td>输入</td>
      <td>深度排序后的高斯球全局ID列表。</td>
      <td>不支持空tensor。必须为1维。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(totalGauss)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>color</td>
      <td>输出</td>
      <td>渲染输出颜色。</td>
      <td>不支持空tensor。shape与tileCoords像素网格一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depth</td>
      <td>输出</td>
      <td>渲染输出深度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>输出</td>
      <td>对数域累积透射率。</td>
      <td>不支持空tensor。作为反向传播输入。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>error</td>
      <td>输出</td>
      <td>Kahan补偿误差项（数学真值恒为0）。</td>
      <td>不支持空tensor。作为反向传播输入。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>lastIds</td>
      <td>输出</td>
      <td>逐像素最后一个贡献高斯球在排序列表中的位置。</td>
      <td>不支持空tensor。作为反向传播输入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/aclnn_return_code.md)。
  
  第一段接口会完成入参校验，出现以下场景时报错：
  
  <table style="width: 100%">
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入输出tensor的数据类型和数据格式不在支持的范围内，或输入输出shape不符合约束。</td>
            </tr>
        </tbody>
    </table>

## aclnnCalcRenderSimtFwd

- **参数说明：**
  
  <table style="width: 100%">
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCalcRenderSimtFwdGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/aclnn_return_code.md)。

## 约束说明

- gs、tileCoords、color、depth、lastCumsum、error的数据类型为FLOAT。
- offsets、gsIds的数据类型为INT64。
- lastIds的数据类型为INT32。
- gs的第二维固定为10；tileCoords的第二维必须为2，第三维（nPixel）固定为1024。
- offsets必须为1维，且长度等于向量核数（blockDim）+ 2*tileNum，否则第一段接口报错。
- color必须为 (3, tileNum, 1024)；depth、lastCumsum、error、lastIds必须为 (tileNum, 1024)，且必须由调用方预分配。
- 空tensor（tileNum为0）场景下算子直接返回，不写出输出。
- 该算子需与calc_render_simt_bwd配套使用，`lastCumsum`、`error`、`lastIds` 输出需传给反向算子。

**维度变量说明：**

- nPixel：每个tile内的像素数量，固定为1024（32×32 tile）。
- tileNum：tile分区数量，表示图像划分的tile区块个数。
- nGauss：高斯球参数数量，表示参与渲染的高斯球参数组数。
- totalGauss：gsIds长度，表示所有tile中累计的高斯球ID总数。
- vectorNum：向量核数（blockDim），由平台动态查询得到。

## 调用示例

见tests/kernel_tests/test_calc_render_simt_fwd.cpp（C++ 直接调用 aclnn 接口）；端到端调用见tests/test_calc_render_fwd.py（通过 `gauss_splat.calc_render` 调用，底层接口见gauss_splat/csrc/CalcRenderSimt.cpp）。
