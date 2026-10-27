# aclnnFusedSsimForward

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_forward/op_kernel/arch22/fused_ssim_forward.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_forward/op_kernel/arch35/fused_ssim_forward.cpp)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
| Atlas 200I/500 A2 推理产品                  |    ×    |
| Atlas 推理系列产品                          |    ×    |
| Atlas 训练系列产品                          |    ×    |

## 功能说明

- 接口功能：`fused_ssim_forward`计算结构相似性（SSIM）的前向结果。输入为两张图像 `img1`、`img2` 及一维高斯窗 `weight`，输出为 SSIM 图 `ssim_map`，以及反向传播所需的窗口滑动中间结果 `mu1`、`mu2`、`S11`、`S22`、`S12`。中间结果在反向时直接复用，避免重复计算多次卷积。
- 计算公式：

以 $W \circledast \cdot$ 表示使用一维高斯窗 $W$ 的二维可分离卷积，$\odot$ 表示逐元素相乘，窗口滑动均值与二阶矩为：

$$
\mu_1 = W \circledast img1, \quad \mu_2 = W \circledast img2
$$

$$
S_{11} = W \circledast (img1 \odot img1), \quad S_{22} = W \circledast (img2 \odot img2), \quad S_{12} = W \circledast (img1 \odot img2)
$$

记 $m_{11} = \mu_1^2$，$m_{22} = \mu_2^2$，$m_{12} = \mu_1\mu_2$，$\sigma_1 = \max(S_{11} - m_{11}, 0)$，$\sigma_2 = \max(S_{22} - m_{22}, 0)$，$\sigma_{12} = S_{12} - m_{12}$，则：

$$
ssim\_map = \frac{(2m_{12} + C_1)(2\sigma_{12} + C_2)}{(m_{11} + m_{22} + C_1)(\sigma_1 + \sigma_2 + C_2)}
$$

其中 $C_1 = 0.0001$，$C_2 = 0.0009$。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnFusedSsimForwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnFusedSsimForward"接口执行计算。

```Cpp
aclnnStatus aclnnFusedSsimForwardGetWorkspaceSize(
    const aclTensor *img1,
    const aclTensor *img2,
    const aclTensor *weight,
    int64_t          windowSize,
    const aclTensor *ssimMap,
    const aclTensor *mu1,
    const aclTensor *mu2,
    const aclTensor *S11,
    const aclTensor *S22,
    const aclTensor *S12,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedSsimForward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnFusedSsimForwardGetWorkspaceSize

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
      <td>img1</td>
      <td>输入</td>
      <td>第一张输入图像。</td>
      <td>不支持空tensor。与 img2 同 shape。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>img2</td>
      <td>输入</td>
      <td>第二张输入图像。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>一维高斯窗，零填充到固定长度16。</td>
      <td>不支持空tensor。长度必须为16，有效窗长由 window_size 决定。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(16,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>windowSize</td>
      <td>输入</td>
      <td>一维高斯窗的有效长度。</td>
      <td>正整数且为奇数，取值范围 (0, 16]；典型值为7、11、15。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ssimMap</td>
      <td>输出</td>
      <td>SSIM 图。</td>
      <td>不支持空tensor。与 img1 同 shape。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mu1</td>
      <td>输出</td>
      <td>img1 的窗口滑动均值。</td>
      <td>不支持空tensor。供反向复用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mu2</td>
      <td>输出</td>
      <td>img2 的窗口滑动均值。</td>
      <td>不支持空tensor。供反向复用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>S11</td>
      <td>输出</td>
      <td>img1 的窗口滑动二阶矩。</td>
      <td>不支持空tensor。供反向复用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>S22</td>
      <td>输出</td>
      <td>img2 的窗口滑动二阶矩。</td>
      <td>不支持空tensor。供反向复用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>S12</td>
      <td>输出</td>
      <td>img1 与 img2 的窗口滑动互相关。</td>
      <td>不支持空tensor。供反向复用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
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
                <td>输入输出tensor的数据类型、数据格式或shape不在支持的范围内，或参数值/属性值不合法。</td>
            </tr>
        </tbody>
    </table>

## aclnnFusedSsimForward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedSsimForwardGetWorkspaceSize获取。</td>
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

- 数据类型：`img1`、`img2`、`weight` 及全部输出 `ssimMap`、`mu1`、`mu2`、`S11`、`S22`、`S12` 均为 FLOAT；数据格式均为 ND。
- `img1`、`img2`、`ssimMap`、`mu1`、`mu2`、`S11`、`S22`、`S12` 的 shape 必须一致，均为4维 (batch_size, channel_num, image_height, image_width)。
- `weight` 必须为1维且长度固定为16（一维高斯窗零填充到16），有效窗长由 `windowSize` 决定。
- `windowSize` 必须为正奇数且不超过16，典型值为7、11、15。
- `mu1`、`mu2`、`S11`、`S22`、`S12` 为反向传播所需中间结果，应与 `fused_ssim_backward` 配合使用，并保持 `weight`、`windowSize` 一致。
- 全部输出均为必选输出，第一段接口需全部传入有效tensor。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的图像批次数量。
- channel_num：图像通道数。
- image_height：图像高度。
- image_width：图像宽度。

## 调用示例

见tests/test_fused_ssim.py
