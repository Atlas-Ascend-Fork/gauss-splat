# aclnnCalcRenderFwdDoubleClipGsids

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_fwd_double_clip_gsids/op_kernel/arch22/calc_render_fwd_double_clip_gsids.cpp)

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

- Interface function: `calc_render_fwd_double_clip_gsids` calculates the forward rendering process, using a double clipping strategy to process Gaussian sphere IDs for 3D Gaussian Splatting rendering.
- Computation formula:

Forward rendering calculates the output color and depth:

$$
color = \sum_{i} \alpha_i \cdot c_i \cdot T_i
$$

$$
depth = \sum_{i} \alpha_i \cdot d_i \cdot T_i
$$

Where $\alpha_i$ is the opacity, $c_i$ is the color, $d_i$ is the depth, and $T_i = \prod_{j<i}(1-\alpha_j)$ is the cumulative transparency.

The double clipping strategy further optimizes rendering efficiency through gsClipIndex and alphaClipIndex.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnCalcRenderFwdDoubleClipGsidsGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnCalcRenderFwdDoubleClipGsids" interface to execute the computation.

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
      <td>gs</td>
      <td>Input</td>
      <td>Gaussian sphere parameters (including position, color, opacity, and so on).</td>
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
      <td>gsIds</td>
      <td>Input</td>
      <td>Gaussian sphere ID list.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(totalGauss)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>color</td>
      <td>Output</td>
      <td>Rendered output color.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depth</td>
      <td>Output</td>
      <td>Rendered output depth.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1, tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>Output</td>
      <td>Last cumulative transparency value.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>error</td>
      <td>Output</td>
      <td>Rendering error value.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, nPixel)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gsClipIndex</td>
      <td>Output</td>
      <td>Gaussian sphere clipping index.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(tileNum)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>alphaClipIndex</td>
      <td>Output</td>
      <td>Alpha clipping index.</td>
      <td>Empty tensors are not supported.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(totalGauss, 2)</td>
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

## aclnnCalcRenderFwdDoubleClipGsids

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnCalcRenderFwdDoubleClipGsidsGetWorkspaceSize.</td>
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

- The data types of gs and tileCoords are FLOAT.
- The data types of offsets, gsIds, and gsClipIndex are INT64.
- The data types of color, depth, lastCumsum, and error are FLOAT.
- The data type of alphaClipIndex is UINT8.

**Dimension Variable Description:**

- nGauss: Number of Gaussian sphere parameters, indicating the number of Gaussian sphere parameter groups participating in rendering.
- tileNum: Number of tile partitions, indicating the number of tile blocks the image is divided into.
- nPixel: Number of pixels, indicating the number of pixel points to be rendered within each tile.
- totalGauss: Total number of Gaussian spheres, indicating the cumulative total number of Gaussian spheres across all tiles.

## Calling Example

See tests/kernel_tests/test_calc_render_fwd_double_clip_gsids.cpp
