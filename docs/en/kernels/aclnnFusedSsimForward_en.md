# aclnnFusedSsimForward

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_forward/op_kernel/arch22/fused_ssim_forward.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fused_ssim_forward/op_kernel/arch35/fused_ssim_forward.cpp)

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

- Interface function: `fused_ssim_forward` computes the forward result of the structural similarity (SSIM). The inputs are two images `img1`, `img2` and a 1D Gaussian window `weight`; the outputs are the SSIM map `ssim_map` and the window sliding-sum intermediates `mu1`, `mu2`, `S11`, `S22`, `S12` required by the backward pass. The intermediates are reused directly during backward to avoid recomputing multiple convolutions.
- Computation formula:

With $W \circledast \cdot$ denoting the 2D separable convolution using the 1D Gaussian window $W$, and $\odot$ denoting element-wise multiplication, the window sliding means and second moments are:

$$
\mu_1 = W \circledast img1, \quad \mu_2 = W \circledast img2
$$

$$
S_{11} = W \circledast (img1 \odot img1), \quad S_{22} = W \circledast (img2 \odot img2), \quad S_{12} = W \circledast (img1 \odot img2)
$$

Let $m_{11} = \mu_1^2$, $m_{22} = \mu_2^2$, $m_{12} = \mu_1\mu_2$, $\sigma_1 = \max(S_{11} - m_{11}, 0)$, $\sigma_2 = \max(S_{22} - m_{22}, 0)$, $\sigma_{12} = S_{12} - m_{12}$, then:

$$
ssim\_map = \frac{(2m_{12} + C_1)(2\sigma_{12} + C_2)}{(m_{11} + m_{22} + C_1)(\sigma_1 + \sigma_2 + C_2)}
$$

where $C_1 = 0.0001$ and $C_2 = 0.0009$.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnFusedSsimForwardGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnFusedSsimForward" interface to execute the computation.

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
      <td>img1</td>
      <td>Input</td>
      <td>The first input image.</td>
      <td>Empty tensors are not supported. Has the same shape as img2.</td>
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
      <td>ssimMap</td>
      <td>Output</td>
      <td>SSIM map.</td>
      <td>Empty tensors are not supported. Has the same shape as img1.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>mu1</td>
      <td>Output</td>
      <td>Window sliding mean of img1.</td>
      <td>Empty tensors are not supported. Reused by the backward pass.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>mu2</td>
      <td>Output</td>
      <td>Window sliding mean of img2.</td>
      <td>Empty tensors are not supported. Reused by the backward pass.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>S11</td>
      <td>Output</td>
      <td>Window sliding second moment of img1.</td>
      <td>Empty tensors are not supported. Reused by the backward pass.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>S22</td>
      <td>Output</td>
      <td>Window sliding second moment of img2.</td>
      <td>Empty tensors are not supported. Reused by the backward pass.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, channel_num, image_height, image_width)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>S12</td>
      <td>Output</td>
      <td>Window sliding cross-correlation of img1 and img2.</td>
      <td>Empty tensors are not supported. Reused by the backward pass.</td>
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

## aclnnFusedSsimForward

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnFusedSsimForwardGetWorkspaceSize.</td>
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

- Data types: `img1`, `img2`, `weight`, and all outputs `ssimMap`, `mu1`, `mu2`, `S11`, `S22`, `S12` are FLOAT; the data format is ND for all.
- `img1`, `img2`, `ssimMap`, `mu1`, `mu2`, `S11`, `S22`, and `S12` must have identical shapes, all 4-dimensional (batch_size, channel_num, image_height, image_width).
- `weight` must be 1-dimensional with a fixed length of 16 (a 1D Gaussian window zero-padded to 16); the effective window length is determined by `windowSize`.
- `windowSize` must be a positive odd integer not greater than 16; typical values are 7, 11, and 15.
- `mu1`, `mu2`, `S11`, `S22`, and `S12` are the intermediates required by the backward pass; they should be used together with `fused_ssim_backward` while keeping `weight` and `windowSize` consistent.
- All outputs are required outputs; the first-stage interface must be passed valid tensors for all of them.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of image batches processed at once.
- channel_num: Number of image channels.
- image_height: Image height.
- image_width: Image width.

## Calling Example

See tests/test_fused_ssim.py
