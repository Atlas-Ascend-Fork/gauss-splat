# aclnnCalcRenderSimtBwd

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_simt_bwd/op_kernel/arch35/calc_render_simt_bwd_apt.cpp)

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

- 接口功能：`calc_render_simt_bwd`计算SIMT渲染的反向传播，使用前向保存的中间结果（`lastCumsum`、`error`、`lastIds`）对高斯球参数计算梯度，用于3D Gaussian Splatting训练。该算子与`calc_render_simt_fwd`成对使用，反向逐像素逆序遍历高斯球列表（遍历上界由前向 `lastIds` 决定）并累加梯度。
- 计算公式：

反向传播计算高斯球参数的梯度：

$$
v\_Gs = \frac{\partial L}{\partial gs}
$$

逐像素逆序遍历高斯球列表。记像素坐标为 $(p_x, p_y)$，当前高斯球参数为 $(\mu_x, \mu_y, c_0, c_1, c_2, o, c_r, c_g, c_b, d)$，定义：

$$
dx = \mu_x - p_x, \quad dy = \mu_y - p_y
$$

$$
w = \exp\left(-\frac{1}{2}\left(c_0 dx^2 + 2 c_1 dx\,dy + c_2 dy^2\right)\right), \quad \alpha = \min(0.999,\ o \cdot w)
$$

记 $T_{i-1}$ 为当前高斯球累加前的累积透射率（前向以对数域保存于 `lastCumsum`，逆序遍历时逐高斯回退），$C_r, C_g, C_b, C_d = \sum_{j>i} \alpha_j T_{j-1} c_{j,\cdot}$ 为逆序遍历中已处理的、位于当前高斯球之后的高斯球的颜色/深度累加量（与前向 $color = \sum_i \alpha_i T_{i-1} c_i$ 的累加量一致）。颜色/深度反向贡献聚合为：

$$
g_\alpha = \left(T_{i-1} c_r - \frac{C_r}{1-\alpha}\right)v_r + \left(T_{i-1} c_g - \frac{C_g}{1-\alpha}\right)v_g + \left(T_{i-1} c_b - \frac{C_b}{1-\alpha}\right)v_b + \left(T_{i-1} c_d - \frac{C_d}{1-\alpha}\right)v_d
$$

其中 $v_r, v_g, v_b, v_d$ 为对应像素的颜色/深度梯度输入。进一步得到不透明度与位置/锥面（conic）参数梯度：

$$
v_o = g_\alpha \cdot w, \quad g_w = -o \cdot v_o
$$

$$
v_x = (c_0 dx + c_1 dy)\, g_w, \quad v_y = (c_1 dx + c_2 dy)\, g_w
$$

$$
v_{c_0} = \frac{1}{2} g_w dx^2, \quad v_{c_1} = g_w\, dx\, dy, \quad v_{c_2} = \frac{1}{2} g_w dy^2
$$

$$
v_r^{gs} = \alpha T_{i-1}\, v_r, \quad v_g^{gs} = \alpha T_{i-1}\, v_g, \quad v_b^{gs} = \alpha T_{i-1}\, v_b, \quad v_d^{gs} = \alpha T_{i-1}\, v_d
$$

当 $o \cdot w$ 超过上界 0.999（$\alpha$ 被裁剪）时，不透明度、位置与锥面梯度按无效处理（置0），颜色/深度梯度仍按 $\alpha T_{i-1} v$ 正常累加；当 $\alpha$ 低于有效阈值 $\frac{1}{255}$ 时，该高斯球的全部梯度按无效处理（置0）。各像素的结果按高斯球ID原子累加，得到最终 $v\_Gs$。输出 $v\_Gs$ 由算子内部协作清零，调用方无需预先清零。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnCalcRenderSimtBwdGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnCalcRenderSimtBwd"接口执行计算。

```Cpp
aclnnStatus aclnnCalcRenderSimtBwdGetWorkspaceSize(
    const aclTensor *vColor,
    const aclTensor *vDepth,
    const aclTensor *lastCumsum,
    const aclTensor *error,
    const aclTensor *gs,
    const aclTensor *tileCoords,
    const aclTensor *offsets,
    const aclTensor *gsIds,
    const aclTensor *lastIds,
    const aclTensor *vGs,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCalcRenderSimtBwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnCalcRenderSimtBwdGetWorkspaceSize

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
      <td>vColor</td>
      <td>输入</td>
      <td>颜色的梯度输入。</td>
      <td>不支持空tensor。与tileCoords共享像素网格(tileNum, 1024)。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vDepth</td>
      <td>输入</td>
      <td>深度的梯度输入，对应前向`depth`输出扩展一个通道维。</td>
      <td>不支持空tensor。与tileCoords共享像素网格(tileNum, 1024)。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1, tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>输入</td>
      <td>前向计算保存的逐像素累积透射率（对数域）。</td>
      <td>不支持空tensor。与tileCoords共享像素网格(tileNum, 1024)。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>error</td>
      <td>输入</td>
      <td>前向计算保存的Kahan补偿误差项（前向输出恒为0）。</td>
      <td>不支持空tensor。与tileCoords共享像素网格(tileNum, 1024)。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gs</td>
      <td>输入</td>
      <td>高斯球参数，每行包含10个字段（位置、锥面、透明度、颜色、深度）。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileCoords</td>
      <td>输入</td>
      <td>逐tile像素坐标信息。</td>
      <td>不支持空tensor。第二维必须为2，第三维（nPixel）固定为1024。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 2, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>核间调度偏移表，三段布局：[向量核累积调度数 | tile调度序 | 累计高斯偏移]。</td>
      <td>不支持空tensor。必须为1维。</td>
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
      <td>lastIds</td>
      <td>输入</td>
      <td>前向计算保存的逐像素最后一个贡献高斯球索引。</td>
      <td>不支持空tensor。与tileCoords共享像素网格(tileNum, 1024)。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vGs</td>
      <td>输出</td>
      <td>高斯球参数的梯度输出。</td>
      <td>不支持空tensor。shape必须与gs一致。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
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

## aclnnCalcRenderSimtBwd

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCalcRenderSimtBwdGetWorkspaceSize获取。</td>
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

- vColor、vDepth、lastCumsum、error、gs、tileCoords、vGs的数据类型为FLOAT。
- offsets、gsIds的数据类型为INT64。
- lastIds的数据类型为INT32。
- gs与vGs的shape必须一致，均为 (nGauss, 10)。
- tileCoords必须为3维且第二维为2，第三维（nPixel）固定为1024。
- offsets、gsIds必须为1维；非空tile下offsets长度（vectorNum + 2*tileNum）须大于2*tileNum。
- vColor、vDepth、lastCumsum、error、lastIds必须与tileCoords共享同一像素网格 (tileNum, 1024)，否则第一段接口返回 shape_mismatch 错误。
- 该算子需与calc_render_simt_fwd配套使用，`gs`、`tileCoords`、`offsets`、`gsIds` 与前向输入一致，`lastCumsum`、`error`、`lastIds` 必须为前向对应输出，`vDepth` 由前向 `depth` 扩展通道维得到。
- 输出vGs由算子内部协作清零后再累加，调用方无需预先清零。
- 空tensor（tileNum为0或nGauss为0）场景下输出vGs全为0。

**维度变量说明：**

- nPixel：每个tile内的像素数量，固定为1024（32×32 tile）。
- tileNum：tile分区数量，表示图像划分的tile区块个数。
- nGauss：高斯球参数数量，表示参与反向传播的高斯球参数组数。
- totalGauss：gsIds长度，表示所有tile中累计的高斯球ID总数。
- vectorNum：向量核数（blockDim），由平台动态查询得到。

## 调用示例

见tests/test_calc_render_bwd.py（通过 `gauss_splat.calc_render` 自动反向调用；底层接口见gauss_splat/csrc/CalcRenderSimt.cpp）。
