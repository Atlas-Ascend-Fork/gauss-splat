# aclnnCalcRenderBwdVarClipGsids

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_bwd_var_clip_gsids/op_kernel/calc_render_bwd_var_clip_gsids.cpp)

## Product Support

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    No    |
| Atlas A3 training series products/Atlas A3 inference series products |    Yes    |
| Atlas A2 training series products/Atlas A2 inference series products |    Yes    |
| Atlas 200I/500 A2 inference products                  |    No    |
| Atlas inference series products                          |    No    |
| Atlas training series products                          |    No    |

## Function Description

- Interface function: `calc_render_bwd_var_clip_gsids` calculates the backward propagation of rendering, using a clipping strategy to compute the gradients of Gaussian sphere parameters.
- Computation formula:

Backward propagation calculates the gradients of Gaussian sphere parameters:

$$
v\_Gs = \frac{\partial L}{\partial gs}
$$

The backward gradient computation is based on the forward rendering formula, considering the clipped Gaussian sphere ID sequence.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnCalcRenderBwdVarClipGsids" interface to execute the computation.

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
      <td>vColor</td>
      <td>Input</td>
      <td>Gradient input for color.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vDepth</td>
      <td>Input</td>
      <td>Gradient input for depth.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1, tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>Input</td>
      <td>Cumulative transparency from the forward computation.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>error</td>
      <td>Input</td>
      <td>Error value from the forward computation.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gs</td>
      <td>Input</td>
      <td>Gaussian sphere parameters.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileCoords</td>
      <td>Input</td>
      <td>Tile coordinate information.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 2, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Offset array.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(tileNum*2+vectorCnt)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gsClipIndexGsIds</td>
      <td>Input</td>
      <td>Clipping index and Gaussian sphere ID combination.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(tileNum + totalGauss)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>alphaClipIndex</td>
      <td>Input</td>
      <td>Alpha clipping index.</td>
      <td>Empty tensors are not supported.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(totalGauss, 2)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vGs</td>
      <td>Output</td>
      <td>Gradient output for Gaussian sphere parameters.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be allocated on the device side.</td>
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
                <td>The data type and format of the input/output tensor are not within the supported range.</td>
            </tr>
        </tbody>
    </table>

## aclnnCalcRenderBwdVarClipGsids

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnCalcRenderBwdVarClipGsidsGetWorkspaceSize.</td>
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

- The data types of vColor, vDepth, lastCumsum, error, gs, tileCoords, and vGs are FLOAT.
- The data types of offsets and gsClipIndexGsIds are INT64.
- The data type of alphaClipIndex is UINT8.

**Dimension Variable Description:**

- nPixel: Number of pixels, indicating the number of pixel points for which gradients need to be calculated.
- nGauss: Number of Gaussian sphere parameters, indicating the number of Gaussian sphere parameter groups participating in backward propagation.
- tileNum: Number of tile partitions, indicating the number of tile blocks the image is divided into.
- totalGauss: Total number of Gaussian spheres, indicating the cumulative total number of Gaussian spheres across all tiles.

## Calling Example

See tests/kernel_tests/test_calc_render_bwd_var_clip_gsids.cpp
