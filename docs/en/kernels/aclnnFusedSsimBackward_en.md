# aclnnFusedSsimBackward

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_backward/op_kernel/arch22/fused_ssim_backward.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_backward/op_kernel/arch35/fused_ssim_backward.cpp)

## Product Support

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    Yes    |
| Atlas A3 training series products/Atlas A3 inference series products |    Yes    |
| Atlas A2 training series products/Atlas A2 inference series products |    Yes    |
| Atlas 200I/500 A2 inference products                  |    No    |
| Atlas inference series products                          |    No    |
| Atlas training series products                          |    No    |

## Function Description

- Interface function: `fused_ssim_backward` computes the backward gradient of the structural similarity (SSIM) loss with respect to the input images. The inputs are the gradient `grad_ssim_map` of the SSIM loss at the forward output, together with the window sliding-sum intermediates (`mu1`, `mu2`, `S11`, `S22`, `S12`) saved by `fused_ssim_forward`. The outputs are the gradient `grad_img1` of `img1` and the (optional) gradient `grad_img2` of `img2`.
- Computation formula:

Let $g = grad\_ssim\_map$, $m_{11} = \mu_1^2$, $m_{22} = \mu_2^2$, $m_{12} = \mu_1\mu_2$, $\sigma_1 = S_{11} - m_{11}$, $\sigma_2 = S_{22} - m_{22}$, $\sigma_{12} = S_{12} - m_{12}$, and define:

$$
A = 2m_{12} + C_1, \quad B = 2\sigma_{12} + C_2, \quad P = m_{11} + m_{22} + C_1, \quad Q = \sigma_1 + \sigma_2 + C_2
$$

$$
M = P \cdot Q, \quad ssim = \frac{A \cdot B}{M}, \quad scale = \frac{2g}{M}
$$

where $C_1 = 0.0001$ and $C_2 = 0.0009$. The intermediate quantities $K_a$, $K_b$, $K_g$, $K_{a2}$ are:

$$
K_a = scale \cdot \big(\mu_2(B - A) - ssim \cdot \mu_1(Q - P)\big)
$$

$$
K_b = scale \cdot A, \quad K_g = -scale \cdot ssim \cdot P
$$

$$
K_{a2} = scale \cdot \big(\mu_1(B - A) - ssim \cdot \mu_2(Q - P)\big) \quad (\text{only when } need\_grad\_img2 = 1)
$$

With $W \circledast \cdot$ denoting the 2D separable convolution using the 1D Gaussian window $W$, and $\odot$ denoting element-wise multiplication:

$$
grad\_img1 = W \circledast K_a + img2 \odot (W \circledast K_b) + img1 \odot (W \circledast K_g)
$$

$$
grad\_img2 = W \circledast K_{a2} + img1 \odot (W \circledast K_b) + img2 \odot (W \circledast K_g)
$$

When `need_grad_img2` is 0, only `grad_img1` is computed (the 1-sided hot path) and `grad_img2` is not written.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnFusedSsimBackwardGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnFusedSsimBackward" interface to execute the computation.

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

- **Parameter Description:**
  
  <table style="width: 100%">
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradSsimMap</td>
      <td>Input</td>
      <td>Gradient of the SSIM loss at the forward output.</td>
      <td>Empty tensors are not supported. Has the same shape as img1/img2.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>img1</td>
      <td>Input</td>
      <td>The first input image.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>img2</td>
      <td>Input</td>
      <td>The second input image.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>1D Gaussian window, zero-padded to a fixed length of 16.</td>
      <td>Empty tensors are not supported. The length must be 16; the effective window length is determined by window_size.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(16,)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>mu1</td>
      <td>Input</td>
      <td>Window sliding mean of img1, saved by the forward pass.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>mu2</td>
      <td>Input</td>
      <td>Window sliding mean of img2, saved by the forward pass.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>S11</td>
      <td>Input</td>
      <td>Window sliding second moment of img1, saved by the forward pass.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>S22</td>
      <td>Input</td>
      <td>Window sliding second moment of img2, saved by the forward pass.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>S12</td>
      <td>Input</td>
      <td>Window sliding cross-correlation of img1 and img2, saved by the forward pass.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>windowSize</td>
      <td>Input</td>
      <td>Effective length of the 1D Gaussian window.</td>
      <td>A positive odd integer in the range (0, 16]; typical values are 7, 11, and 15.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needGradImg2</td>
      <td>Input</td>
      <td>Whether to compute the gradient of img2.</td>
      <td>Either 0 or 1; when 0 only grad_img1 is computed (the 1-sided hot path), when 1 grad_img2 is computed as well.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradImg1</td>
      <td>Output</td>
      <td>Gradient of img1.</td>
      <td>Empty tensors are not supported. Has the same shape as gradSsimMap.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gradImg2</td>
      <td>Output</td>
      <td>Gradient of img2.</td>
      <td>Empty tensors are not supported. Valid when needGradImg2 is 1; otherwise it is not written (keeps the caller's buffer initial value).</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size required to be allocated on the device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **Return Value:**
  
  aclnnStatus: Returns a status code. For details, refer to [aclnn Return Codes](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/aclnn_return_code.md).
  
  The first-stage interface completes input parameter validation and reports errors in the following scenarios:
  
  <table style="width: 100%">
        <thead>
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>Returns 161001 if a required input, output, or required attribute parameter is a null pointer.</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>The data type, data format, or shape of the input/output tensor is not within the supported range, or the parameter value/attribute value is invalid.</td>
            </tr>
        </tbody>
    </table>

## aclnnFusedSsimBackward

- **Parameter Description:**
  
  <table style="width: 100%">
  <thead>
    <tr>
      <th>Parameter Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnFusedSsimBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing the task.</td>
    </tr>
  </tbody>
  </table>
- **Return Value:**
  
  aclnnStatus: Returns a status code. For details, refer to [aclnn Return Codes](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/aclnn_return_code.md).

## Constraints

- Data types: `gradSsimMap`, `img1`, `img2`, `weight`, `mu1`, `mu2`, `S11`, `S22`, `S12`, and the outputs `gradImg1` and `gradImg2` are all FLOAT; the data format is ND for all.
- `gradSsimMap`, `img1`, `img2`, `mu1`, `mu2`, `S11`, `S22`, `S12`, `gradImg1`, and `gradImg2` must have identical shapes, all 4-dimensional (batch_size, channel_num, image_height, image_width).
- `weight` must be 1-dimensional with a fixed length of 16 (a 1D Gaussian window zero-padded to 16); the effective window length is determined by `windowSize`.
- `windowSize` must be a positive odd integer not greater than 16; typical values are 7, 11, and 15.
- `needGradImg2` is either 0 or 1: 0 corresponds to the 1-sided path (only `grad_img1` is computed), and 1 corresponds to the 2-sided path (both `grad_img2` and the additional intermediate quantity $K_{a2}$ are computed).
- `mu1`, `mu2`, `S11`, `S22`, and `S12` must be consistent with the intermediates saved by the `fused_ssim_forward` forward pass, and `weight` and `windowSize` must be consistent with the forward call.
- Both outputs are required outputs; the first-stage interface must be passed valid tensors for both of them.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of image batches processed at once.
- channel_num: Number of image channels.
- image_height: Image height.
- image_width: Image width.

## Calling Example

See tests/test_fused_ssim_backward.py
