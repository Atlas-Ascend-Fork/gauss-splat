# aclnnFusedSsimBackward

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_backward/op_kernel/arch22/fused_ssim_backward.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_backward/op_kernel/arch35/fused_ssim_backward.cpp)

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

- 接口功能：`fused_ssim_backward`计算结构相似性（SSIM）损失对输入图像的反向梯度。输入为 SSIM 损失在前向输出的梯度 `grad_ssim_map`，以及 `fused_ssim_forward` 保存的窗口滑动和中间结果（`mu1`、`mu2`、`S11`、`S22`、`S12`），输出为 `img1` 的梯度 `grad_img1` 和（可选的）`img2` 的梯度 `grad_img2`。
- 计算公式：

记 $g = grad\_ssim\_map$，$m_{11} = \mu_1^2$，$m_{22} = \mu_2^2$，$m_{12} = \mu_1\mu_2$，$\sigma_1 = S_{11} - m_{11}$，$\sigma_2 = S_{22} - m_{22}$，$\sigma_{12} = S_{12} - m_{12}$，并令：

$$
A = 2m_{12} + C_1, \quad B = 2\sigma_{12} + C_2, \quad P = m_{11} + m_{22} + C_1, \quad Q = \sigma_1 + \sigma_2 + C_2
$$

$$
M = P \cdot Q, \quad ssim = \frac{A \cdot B}{M}, \quad scale = \frac{2g}{M}
$$

其中 $C_1 = 0.0001$，$C_2 = 0.0009$。中间量 $K_a$、$K_b$、$K_g$、$K_{a2}$ 为：

$$
K_a = scale \cdot \big(\mu_2(B - A) - ssim \cdot \mu_1(Q - P)\big)
$$

$$
K_b = scale \cdot A, \quad K_g = -scale \cdot ssim \cdot P
$$

$$
K_{a2} = scale \cdot \big(\mu_1(B - A) - ssim \cdot \mu_2(Q - P)\big) \quad (\text{仅当 } need\_grad\_img2 = 1)
$$

以 $W \circledast \cdot$ 表示使用一维高斯窗 $W$ 的二维可分离卷积，$\odot$ 表示逐元素相乘，则：

$$
grad\_img1 = W \circledast K_a + img2 \odot (W \circledast K_b) + img1 \odot (W \circledast K_g)
$$

$$
grad\_img2 = W \circledast K_{a2} + img1 \odot (W \circledast K_b) + img2 \odot (W \circledast K_g)
$$

当 `need_grad_img2` 为 0 时，仅计算 `grad_img1`（1-sided 热路径），`grad_img2` 不写出。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnFusedSsimBackwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnFusedSsimBackward"接口执行计算。

```Cpp
aclnnStatus aclnnFusedSsimBackwardGetWorkspaceSize(
    const aclTensor *gradSsimMap,
    const aclTensor *img1,
    const aclTensor *img2,
    const aclTensor *weight,
    const aclTensor *mu1,
    const aclTensor *mu2,
    const aclTensor *S11,
    const aclTensor *S22,
    const aclTensor *S12,
    int64_t          windowSize,
    int64_t          needGradImg2,
    const aclTensor *gradImg1,
    const aclTensor *gradImg2,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFusedSsimBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnFusedSsimBackwardGetWorkspaceSize

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
      <td>gradSsimMap</td>
      <td>输入</td>
      <td>SSIM 损失在前向输出上的梯度。</td>
      <td>不支持空tensor。与 img1/img2 同 shape。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>img1</td>
      <td>输入</td>
      <td>第一张输入图像。</td>
      <td>不支持空tensor。</td>
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
      <td>mu1</td>
      <td>输入</td>
      <td>img1 的窗口滑动均值，由前向保存。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mu2</td>
      <td>输入</td>
      <td>img2 的窗口滑动均值，由前向保存。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>S11</td>
      <td>输入</td>
      <td>img1 的窗口滑动二阶矩，由前向保存。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>S22</td>
      <td>输入</td>
      <td>img2 的窗口滑动二阶矩，由前向保存。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>S12</td>
      <td>输入</td>
      <td>img1 与 img2 的窗口滑动互相关，由前向保存。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
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
      <td>needGradImg2</td>
      <td>输入</td>
      <td>是否计算 img2 的梯度。</td>
      <td>取0或1；为0时仅计算 grad_img1（1-sided 热路径），为1时同时计算 grad_img2。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradImg1</td>
      <td>输出</td>
      <td>img1 的梯度。</td>
      <td>不支持空tensor。与 gradSsimMap 同 shape。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradImg2</td>
      <td>输出</td>
      <td>img2 的梯度。</td>
      <td>不支持空tensor。needGradImg2 为1时有效，否则不写出（保持调用方缓冲区初值）。</td>
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

## aclnnFusedSsimBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedSsimBackwardGetWorkspaceSize获取。</td>
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

- 数据类型：`gradSsimMap`、`img1`、`img2`、`weight`、`mu1`、`mu2`、`S11`、`S22`、`S12` 及输出 `gradImg1`、`gradImg2` 均为 FLOAT；数据格式均为 ND。
- `gradSsimMap`、`img1`、`img2`、`mu1`、`mu2`、`S11`、`S22`、`S12`、`gradImg1`、`gradImg2` 的 shape 必须一致，均为4维 (batch_size, channel_num, image_height, image_width)。
- `weight` 必须为1维且长度固定为16（一维高斯窗零填充到16），有效窗长由 `windowSize` 决定。
- `windowSize` 必须为正奇数且不超过16，典型值为7、11、15。
- `needGradImg2` 取值为0或1：为0时对应 1-sided 路径（仅计算 `grad_img1`），为1时对应 2-sided 路径（同时计算 `grad_img2` 与附加中间量 $K_{a2}$）。
- `mu1`、`mu2`、`S11`、`S22`、`S12` 必须与 `fused_ssim_forward` 前向保存的中间结果一致，且 `weight`、`windowSize` 需与前向调用保持一致。
- 两个输出均为必选输出，第一段接口需全部传入有效tensor。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的图像批次数量。
- channel_num：图像通道数。
- image_height：图像高度。
- image_width：图像宽度。

## 调用示例

见tests/test_fused_ssim_backward.py
