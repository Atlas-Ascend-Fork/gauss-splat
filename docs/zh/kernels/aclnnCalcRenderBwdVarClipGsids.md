# aclnnCalcRenderBwdVarClipGsids

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_bwd_var_clip_gsids/op_kernel/calc_render_bwd_var_clip_gsids.cpp)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    ×    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
| Atlas 200I/500 A2 推理产品                  |    ×    |
| Atlas 推理系列产品                          |    ×    |
| Atlas 训练系列产品                          |    ×    |

## 功能说明

- 接口功能：`calc_render_bwd_var_clip_gsids`计算渲染反向传播，使用裁剪策略计算高斯球参数的梯度。
- 计算公式：

反向传播计算高斯球参数的梯度：

$$
v\_Gs = \frac{\partial L}{\partial gs}
$$

基于前向渲染公式进行反向梯度计算，考虑裁剪后的高斯球ID序列。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnCalcRenderBwdVarClipGsids"接口执行计算。

```Cpp
aclnnStatus aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize(
    const aclTensor *vColor,
    const aclTensor *vDepth,
    const aclTensor *lastCumsum,
    const aclTensor *error,
    const aclTensor *gs,
    const aclTensor *tileCoords,
    const aclTensor *offsets,
    const aclTensor *gsClipIndexGsIds,
    const aclTensor *alphaClipIndex,
    const aclTensor *vGs,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCalcRenderBwdVarClipGsids(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize

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
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vDepth</td>
      <td>输入</td>
      <td>深度的梯度输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1, tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>输入</td>
      <td>前向计算的累积透明度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>error</td>
      <td>输入</td>
      <td>前向计算的误差值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gs</td>
      <td>输入</td>
      <td>高斯球参数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileCoords</td>
      <td>输入</td>
      <td>Tile坐标信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 2, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>偏移量数组。</td>
      <td>不支持空tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(tileNum*2+vectorCnt)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gsClipIndexGsIds</td>
      <td>输入</td>
      <td>裁剪索引与高斯球ID组合。</td>
      <td>不支持空tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(totalGauss)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alphaClipIndex</td>
      <td>输入</td>
      <td>透明度裁剪索引。</td>
      <td>不支持空tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vGs</td>
      <td>输出</td>
      <td>高斯球参数的梯度输出。</td>
      <td>不支持空tensor。</td>
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
                <td>输入输出tensor的数据类型和数据格式不在支持的范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnCalcRenderBwdVarClipGsids

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize获取。</td>
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
- offsets、gsClipIndexGsIds的数据类型为INT64。
- alphaClipIndex的数据类型为UINT8。

**维度变量说明：**

- nPixel：像素数量，表示需要计算梯度的像素点个数。
- nGauss：高斯球参数数量，表示参与反向传播的高斯球参数组数。
- tileNum：tile分区数量，表示图像划分的tile区块个数。
- totalGauss：总高斯球数量，表示所有tile中累计的高斯球总数。

## 调用示例

见tests/kernel_tests/test_calc_render_bwd_var_clip_gsids.cpp
