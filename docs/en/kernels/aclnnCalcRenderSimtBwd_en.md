# aclnnCalcRenderSimtBwd

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_simt_bwd/op_kernel/arch35/calc_render_simt_bwd_apt.cpp)

## Product Support

| Product                                                     | Supported |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    Yes    |
| Atlas A3 training series products/Atlas A3 inference series products |    No    |
| Atlas A2 training series products/Atlas A2 inference series products |    No    |
| Atlas 200I/500 A2 inference products                  |    No    |
| Atlas inference series products                          |    No    |
| Atlas training series products                          |    No    |

## Function Description

- Interface function: `calc_render_simt_bwd` calculates the backward propagation of SIMT rendering. It computes the gradients of Gaussian sphere parameters using the intermediate results saved by the forward pass (`lastCumsum`, `error`, `lastIds`), for 3D Gaussian Splatting training. This operator is used in pairs with `calc_render_simt_fwd`; the backward traverses the Gaussian sphere list in reverse order per pixel (the traversal upper bound is determined by the forward `lastIds`) and accumulates the gradients.
- Computation formula:

Backward propagation calculates the gradients of Gaussian sphere parameters:

$$
v\_Gs = \frac{\partial L}{\partial gs}
$$

Traverse the Gaussian sphere list in reverse order per pixel. Let the pixel coordinate be $(p_x, p_y)$ and the current Gaussian sphere parameters be $(\mu_x, \mu_y, c_0, c_1, c_2, o, c_r, c_g, c_b, d)$. Define:

$$
dx = \mu_x - p_x, \quad dy = \mu_y - p_y
$$

$$
w = \exp\left(-\frac{1}{2}\left(c_0 dx^2 + 2 c_1 dx\,dy + c_2 dy^2\right)\right), \quad \alpha = \min(0.999,\ o \cdot w)
$$

Let $T_{i-1}$ be the accumulated transmittance before the current Gaussian sphere (saved in the log domain as `lastCumsum` in the forward pass, and rolled back Gaussian by Gaussian during the reverse traversal), and let $C_r, C_g, C_b, C_d = \sum_{j>i} \alpha_j T_{j-1} c_{j,\cdot}$ be the color/depth accumulations of the Gaussian spheres located behind the current one that have already been processed during the reverse traversal (consistent with the forward accumulation $color = \sum_i \alpha_i T_{i-1} c_i$). The aggregated color/depth backward contribution is:

$$
g_\alpha = \left(T_{i-1} c_r - \frac{C_r}{1-\alpha}\right)v_r + \left(T_{i-1} c_g - \frac{C_g}{1-\alpha}\right)v_g + \left(T_{i-1} c_b - \frac{C_b}{1-\alpha}\right)v_b + \left(T_{i-1} c_d - \frac{C_d}{1-\alpha}\right)v_d
$$

where $v_r, v_g, v_b, v_d$ are the color/depth gradient inputs of the corresponding pixel. The gradients for the opacity and the position/conic parameters are then obtained as:

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

When $o \cdot w$ exceeds the upper bound 0.999 ($\alpha$ is clipped), the opacity, position, and conic gradients are treated as invalid (0), while the color/depth gradients are still accumulated as $\alpha T_{i-1} v$; when $\alpha$ falls below the validity threshold $\frac{1}{255}$, all gradients of that Gaussian sphere are treated as invalid (0). The results of all pixels are atomically accumulated by Gaussian sphere ID to obtain the final $v\_Gs$. The output $v\_Gs$ is cooperatively zeroed inside the operator, so the caller does not need to pre-zero it.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnCalcRenderSimtBwdGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnCalcRenderSimtBwd" interface to execute the computation.

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
      <td>Empty tensors are not supported. Shares the pixel grid (tileNum, 1024) with tileCoords.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vDepth</td>
      <td>Input</td>
      <td>Gradient input for depth; corresponds to the forward `depth` output with one extra channel dimension.</td>
      <td>Empty tensors are not supported. Shares the pixel grid (tileNum, 1024) with tileCoords.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1, tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>Input</td>
      <td>Per-pixel accumulated transmittance (log domain) saved by the forward computation.</td>
      <td>Empty tensors are not supported. Shares the pixel grid (tileNum, 1024) with tileCoords.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>error</td>
      <td>Input</td>
      <td>Kahan compensation error term saved by the forward computation (always 0 in the forward output).</td>
      <td>Empty tensors are not supported. Shares the pixel grid (tileNum, 1024) with tileCoords.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gs</td>
      <td>Input</td>
      <td>Gaussian sphere parameters; each row contains 10 fields (position, conic, opacity, color, depth).</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileCoords</td>
      <td>Input</td>
      <td>Per-tile pixel coordinate information.</td>
      <td>Empty tensors are not supported. The second dimension must be 2 and the third dimension (nPixel) is fixed at 1024.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 2, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Inter-core schedule offset table in a three-segment layout: [cumulative vector-core schedule counts | tile schedule sequence | cumulative Gaussian offsets].</td>
      <td>Empty tensors are not supported. Must be 1-dimensional.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(vectorNum + 2*tileNum)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gsIds</td>
      <td>Input</td>
      <td>Depth-sorted list of global Gaussian sphere IDs.</td>
      <td>Empty tensors are not supported. Must be 1-dimensional.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(totalGauss)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>lastIds</td>
      <td>Input</td>
      <td>Per-pixel last contributing Gaussian sphere index saved by the forward computation.</td>
      <td>Empty tensors are not supported. Shares the pixel grid (tileNum, 1024) with tileCoords.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vGs</td>
      <td>Output</td>
      <td>Gradient output for Gaussian sphere parameters.</td>
      <td>Empty tensors are not supported. Shape must be the same as gs.</td>
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
                <td>The data type and format of the input/output tensor are not within the supported range, or the input/output shape does not satisfy the constraints.</td>
            </tr>
        </tbody>
    </table>

## aclnnCalcRenderSimtBwd

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnCalcRenderSimtBwdGetWorkspaceSize.</td>
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
- The data types of offsets and gsIds are INT64.
- The data type of lastIds is INT32.
- gs and vGs must have the same shape, both (nGauss, 10).
- tileCoords must be 3-dimensional with the second dimension equal to 2, and the third dimension (nPixel) is fixed at 1024.
- offsets and gsIds must be 1-dimensional; for non-empty tiles, the length of offsets (vectorNum + 2*tileNum) must be greater than 2*tileNum.
- vColor, vDepth, lastCumsum, error, and lastIds must share the same pixel grid (tileNum, 1024) with tileCoords; otherwise the first-stage interface returns a shape_mismatch error.
- This operator must be used together with calc_render_simt_fwd: `gs`, `tileCoords`, `offsets`, and `gsIds` are the same as the forward inputs; `lastCumsum`, `error`, and `lastIds` must be the corresponding forward outputs; and `vDepth` is obtained by adding a channel dimension to the forward `depth`.
- The output vGs is cooperatively zeroed inside the operator before accumulation, so the caller does not need to pre-zero it.
- For empty tensors (tileNum == 0 or nGauss == 0), the output vGs is all zeros.

**Dimension Variable Description:**

- nPixel: Number of pixels per tile, fixed at 1024 (32×32 tile).
- tileNum: Number of tile partitions, indicating the number of tile blocks the image is divided into.
- nGauss: Number of Gaussian sphere parameters, indicating the number of Gaussian sphere parameter groups participating in backward propagation.
- totalGauss: Length of gsIds, indicating the cumulative total number of Gaussian sphere IDs across all tiles.
- vectorNum: Number of vector cores (blockDim), queried dynamically from the platform.

## Calling Example

See tests/test_calc_render_bwd.py (invoked automatically through the backward pass of `gauss_splat.calc_render`; for the low-level interface see gauss_splat/csrc/CalcRenderSimt.cpp).
