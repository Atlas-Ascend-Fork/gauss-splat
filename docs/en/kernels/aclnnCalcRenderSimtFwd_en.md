# aclnnCalcRenderSimtFwd

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/calc_render_simt_fwd/op_kernel/arch35/calc_render_simt_fwd_apt.cpp)

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

- Interface function: `calc_render_simt_fwd` calculates the forward pass of SIMT rendering. For each pixel it performs alpha blending accumulation over Gaussian spheres in depth-sorted order, outputs the color and depth, and saves the intermediate results (`lastCumsum`, `error`, `lastIds`) required by the backward pass. This operator is used in pairs with `calc_render_simt_bwd`.
- Computation formula:

Traverse the Gaussian spheres in sorted order per pixel. Let the pixel coordinate be $(p_x, p_y)$ and the current Gaussian sphere parameters be $(\mu_x, \mu_y, c_0, c_1, c_2, o, c_r, c_g, c_b, d)$. Define the displacement (the quadratic form is invariant when $dx, dy$ both change sign, so the sign convention matches the backward pass):

$$
dx = \mu_x - p_x, \quad dy = \mu_y - p_y
$$

$$
w = \exp\left(-\frac{1}{2}\left(c_0 dx^2 + 2 c_1 dx\,dy + c_2 dy^2\right)\right)
$$

$$
\alpha = \min(o \cdot w,\ 0.999)
$$

When $\alpha \ge \frac{1}{255}$, let $T_{i-1}$ be the transmittance before accumulation and compute $T_i = T_{i-1}(1-\alpha)$:

- If $T_i > 10^{-4}$, add the color/depth contribution of this Gaussian sphere to the accumulation:

$$
color = \sum_{i} \alpha_i T_{i-1} c_i, \quad depth = \sum_{i} \alpha_i T_{i-1} d_i
$$

- Otherwise the pixel reaches the per-pixel done condition; record `lastIds` and terminate the traversal (this Gaussian sphere and all following ones are no longer accumulated).

In both cases the log-domain accumulated transmittance (`lastCumsum`) is updated with Kahan compensation:

$$
s \mathrel{+}= \ln(1-\alpha), \quad T_{i-1} = e^{s}
$$

The forward pass also outputs:

- `lastCumsum`: the accumulated transmittance in the log domain $s = \sum \ln(1-\alpha)$;
- `error`: the Kahan compensation error term (mathematically always 0);
- `lastIds`: the position, in the sorted list, of the last contributing Gaussian sphere for each pixel (0 for pixels with no contribution).

These intermediate outputs, together with the inputs `gs`, `tileCoords`, `offsets`, and `gsIds`, serve as the inputs of `calc_render_simt_bwd`; note that `depth` has shape (tileNum, 1024), and must be extended to (1, tileNum, 1024) as `vDepth` for the backward pass.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnCalcRenderSimtFwdGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnCalcRenderSimtFwd" interface to execute the computation.

```Cpp
aclnnStatus aclnnCalcRenderSimtFwdGetWorkspaceSize(
    const aclTensor *gs,
    const aclTensor *tileCoords,
    const aclTensor *offsets,
    const aclTensor *gsIds,
    const aclTensor *color,
    const aclTensor *depth,
    const aclTensor *lastCumsum,
    const aclTensor *error,
    const aclTensor *lastIds,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCalcRenderSimtFwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnCalcRenderSimtFwdGetWorkspaceSize

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
      <td>Gaussian sphere parameters; each row contains 10 fields (position, conic, opacity, color, depth).</td>
      <td>Empty tensors are not supported. The second dimension must be 10.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(nGauss, 10)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileCoords</td>
      <td>Input</td>
      <td>Per-tile pixel coordinate information.</td>
      <td>Empty tensors are not supported. The second dimension must be 2 and the third dimension is fixed at 1024.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 2, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Input</td>
      <td>Inter-core schedule offset table in a three-segment layout: [cumulative vector-core schedule counts | tile schedule sequence | cumulative Gaussian offsets].</td>
      <td>Empty tensors are not supported. Must be 1-dimensional with length equal to the number of vector cores + 2*tileNum.</td>
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
      <td>color</td>
      <td>Output</td>
      <td>Rendered output color.</td>
      <td>Empty tensors are not supported. Shape matches the tileCoords pixel grid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(3, tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depth</td>
      <td>Output</td>
      <td>Rendered output depth.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>lastCumsum</td>
      <td>Output</td>
      <td>Accumulated transmittance in the log domain.</td>
      <td>Empty tensors are not supported. Used as a backward pass input.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>error</td>
      <td>Output</td>
      <td>Kahan compensation error term (mathematically always 0).</td>
      <td>Empty tensors are not supported. Used as a backward pass input.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>lastIds</td>
      <td>Output</td>
      <td>Position, in the sorted list, of the last contributing Gaussian sphere for each pixel.</td>
      <td>Empty tensors are not supported. Used as a backward pass input.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(tileNum, 1024)</td>
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

## aclnnCalcRenderSimtFwd

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnCalcRenderSimtFwdGetWorkspaceSize.</td>
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

- The data types of gs, tileCoords, color, depth, lastCumsum, and error are FLOAT.
- The data types of offsets and gsIds are INT64.
- The data type of lastIds is INT32.
- The second dimension of gs is fixed at 10; the second dimension of tileCoords must be 2, and the third dimension (nPixel) is fixed at 1024.
- offsets must be 1-dimensional with length equal to the number of vector cores (blockDim) + 2*tileNum; otherwise the first-stage interface reports an error.
- color must be (3, tileNum, 1024); depth, lastCumsum, error, and lastIds must be (tileNum, 1024), and must be pre-allocated by the caller.
- For empty tensors (tileNum == 0), the operator returns immediately without writing the outputs.
- This operator must be used together with calc_render_simt_bwd; the `lastCumsum`, `error`, and `lastIds` outputs need to be passed to the backward operator.

**Dimension Variable Description:**

- nPixel: Number of pixels per tile, fixed at 1024 (32×32 tile).
- tileNum: Number of tile partitions, indicating the number of tile blocks the image is divided into.
- nGauss: Number of Gaussian sphere parameters, indicating the number of Gaussian sphere parameter groups participating in rendering.
- totalGauss: Length of gsIds, indicating the cumulative total number of Gaussian sphere IDs across all tiles.
- vectorNum: Number of vector cores (blockDim), queried dynamically from the platform.

## Calling Example

See tests/kernel_tests/test_calc_render_simt_fwd.cpp (direct aclnn invocation in C++); for the end-to-end path see tests/test_calc_render_fwd.py (invoked through `gauss_splat.calc_render`; for the low-level interface see gauss_splat/csrc/CalcRenderSimt.cpp).
