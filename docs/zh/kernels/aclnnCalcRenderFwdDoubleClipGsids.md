# aclnnCalcRenderFwdDoubleClipGsids

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_fwd_double_clip_gsids/op_kernel/arch22/calc_render_fwd_double_clip_gsids.cpp)

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

- 接口功能：`calc_render_fwd_double_clip_gsids`计算渲染前向过程，使用双裁剪策略处理高斯球ID，用于3D Gaussian Splatting渲染。
- 计算公式：

渲染前向计算输出颜色和深度：

$$
color = \sum_{i} \alpha_i \cdot c_i \cdot T_i
$$

$$
depth = \sum_{i} \alpha_i \cdot d_i \cdot T_i
$$

其中$\alpha_i$是透明度，$c_i$是颜色，$d_i$是深度，$T_i = \prod_{j<i}(1-\alpha_j)$是累积透明度。

双裁剪策略通过gsClipIndex和alphaClipIndex进一步优化渲染效率。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnCalcRenderFwdDoubleClipGsidsGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnCalcRenderFwdDoubleClipGsids"接口执行计算。

```Cpp
aclnnStatus aclnnCalcRenderFwdDoubleClipGsidsGetWorkspaceSize(
    const aclTensor *gs,
    const aclTensor *tileCoords,
    const aclTensor *offsets,
    const aclTensor *gsIds,
    const aclTensor *color,
    const aclTensor *depth,
    const aclTensor *lastCumsum,
    const aclTensor *error,
    const aclTensor *gsClipIndex,
    const aclTensor *alphaClipIndex,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCalcRenderFwdDoubleClipGsids(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnCalcRenderFwdDoubleClipGsidsGetWorkspaceSize

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
      <td>高斯球参数（包含位置、颜色、透明度等）。</td>
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
      <td>gsIds</td>
      <td>输入</td>
      <td>高斯球ID列表。</td>
      <td>不支持空tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(totalGauss)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>color</td>
      <td>输出</td>
      <td>渲染输出颜色。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depth</td>
      <td>输出</td>
      <td>渲染输出深度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1, tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>输出</td>
      <td>最后累积透明度值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>error</td>
      <td>输出</td>
      <td>渲染误差值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gsClipIndex</td>
      <td>输出</td>
      <td>高斯球裁剪索引。</td>
      <td>不支持空tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(tileNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alphaClipIndex</td>
      <td>输出</td>
      <td>透明度裁剪索引。</td>
      <td>不支持空tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(totalGauss, 2)</td>
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

## aclnnCalcRenderFwdDoubleClipGsids

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCalcRenderFwdDoubleClipGsidsGetWorkspaceSize获取。</td>
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

- gs、tileCoords的数据类型为FLOAT。
- offsets、gsIds、gsClipIndex的数据类型为INT64。
- color、depth、lastCumsum、error的数据类型为FLOAT。
- alphaClipIndex的数据类型为UINT8。

**维度变量说明：**

- nGauss：高斯球参数数量，表示参与渲染的高斯球参数组数。
- tileNum：tile分区数量，表示图像划分的tile区块个数。
- nPixel：像素数量，表示每个tile内需要渲染的像素点个数。
- totalGauss：总高斯球数量，表示所有tile中累计的高斯球总数。

## 调用示例

见tests/kernel_tests/test_calc_render_fwd_double_clip_gsids.cpp
