# aclnnGaussianFilter

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_filter/op_kernel/arch22/gaussian_filter.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_filter/op_kernel/arch35/gaussian_filter.cpp)

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

- Interface function: `gaussian_filter` filters Gaussian spheres for each (batch, camera) view based on view frustum parameters, culls Gaussian spheres that are not within the visible range, and compacts the filtering result along the gaussian dimension (compact). It also outputs filter flags (a bit-packed bitmap and a float mask), the number of valid Gaussian spheres per view, and a mapping from compacted positions back to original dense indices, and supports depth correction with `vi` and `sigma`.
- Computation formula:

For the $i$-th Gaussian sphere under each (batch, camera) view, let:

$$
d_i = depths[b,c,i], \quad m_i = means2d[b,c,:,i], \quad r_i = radius[b,c,:,i]
$$

The filter flag is:

$$
filter[b,c,i] = \begin{cases}
1 & \text{if } i < gaussian\_num \ \wedge\ det[b,c,i] > 0 \ \wedge\ near\_plane < d_i < far\_plane \\
  & \wedge\ vi[b,c,i] > 0 \\
  & \wedge\ (m_{x,i} + r_{x,i} > 0) \wedge (m_{x,i} - r_{x,i} < width) \\
  & \wedge\ (m_{y,i} + r_{y,i} > 0) \wedge (m_{y,i} - r_{y,i} < height) \\
0 & \text{otherwise}
\end{cases}
$$

The `vi > 0` condition takes effect on the A5 (Ascend 950) path.

Number of valid Gaussian spheres per view:

$$
cnt[b,c] = \sum_{i} filter[b,c,i]
$$

Compact layout and index mapping ($\{\, i \mid filter[b,c,i]=1 \,\}$ ordered by original index):

$$
culling[b,c,:,0:cnt] = args[b,c,:,i] \big|_{filter[b,c,i]=1}, \quad denseIndices[b,c,k] = \text{original index of the } k\text{-th valid Gaussian sphere}
$$

Here `args` denotes each parameter (`means`, `colors`, `quats`, `scales`, `means2d`, `depths`, `radius`, `covars2d`, `conics`, `opacities`, `vi`, `compensations`), written into the corresponding `*Culling` output; `filter` is a bit-packed UINT8 bitmap, and `filterFloat` is the corresponding 0/1 float mask. When `supportFeatures` bit0 (depth correction) is enabled, `viCulling = vi × max(1 − depths / sigma, 0)`; otherwise `viCulling = vi`.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnGaussianFilterGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnGaussianFilter" interface to execute the computation.

```Cpp
aclnnStatus aclnnGaussianFilterGetWorkspaceSize(
    const aclTensor *means,
    const aclTensor *colors,
    const aclTensor *det,
    const aclTensor *opacities,
    const aclTensor *means2d,
    const aclTensor *depths,
    const aclTensor *radius,
    const aclTensor *conics,
    const aclTensor *covars2d,
    const aclTensor *compensations,
    const aclTensor *quats,
    const aclTensor *scales,
    const aclTensor *vi,
    const aclTensor *sigma,
    int64_t          width,
    int64_t          height,
    float            nearPlane,
    float            farPlane,
    int64_t          supportFeatures,
    const aclTensor *meansCulling,
    const aclTensor *colorsCulling,
    const aclTensor *means2dCulling,
    const aclTensor *depthsCulling,
    const aclTensor *radiusCulling,
    const aclTensor *covars2dCulling,
    const aclTensor *conicsCulling,
    const aclTensor *opacitiesCulling,
    const aclTensor *quatsCulling,
    const aclTensor *scalesCulling,
    const aclTensor *viCulling,
    const aclTensor *compensationsCulling,
    const aclTensor *filter,
    const aclTensor *cnt,
    const aclTensor *filterFloat,
    const aclTensor *denseIndices,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGaussianFilter(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnGaussianFilterGetWorkspaceSize

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
      <td>means</td>
      <td>Input</td>
      <td>3D Gaussian sphere positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>colors</td>
      <td>Input</td>
      <td>Gaussian sphere colors.</td>
      <td>Empty tensors are not supported. The channel count color_dims is usually 3 and can be 7 when extended features are enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, color_dims, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>det</td>
      <td>Input</td>
      <td>Covariance determinant values.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>opacities</td>
      <td>Input</td>
      <td>Gaussian sphere opacities.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>means2d</td>
      <td>Input</td>
      <td>Projected 2D positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>Input</td>
      <td>Gaussian sphere depths.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>Input</td>
      <td>Gaussian sphere screen radii.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>Input</td>
      <td>Conic parameters.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars2d</td>
      <td>Input</td>
      <td>2D covariance matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>Input</td>
      <td>Opacity compensation values.</td>
      <td>Optional input, can pass nullptr. When not empty, opacitiesCulling is weighted by opacity × compensation, and compensationsCulling is output.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>quats</td>
      <td>Input</td>
      <td>Quaternions.</td>
      <td>Optional input, can pass nullptr. When not empty, participates in filtering and is output as quatsCulling.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>Scale factors.</td>
      <td>Optional input, can pass nullptr. When not empty, participates in filtering and is output as scalesCulling.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vi</td>
      <td>Input</td>
      <td>Per-view depth correction weights (per-Gaussian, per-view weights in LCWSR).</td>
      <td>Optional input, can pass nullptr. When not empty, the filter requires vi &gt; 0, and viCulling is output.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>Input</td>
      <td>Depth correction scale parameter (scalar).</td>
      <td>Optional input, can pass nullptr. Used when depth correction (supportFeatures bit0) is enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1,)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>width</td>
      <td>Input</td>
      <td>Image width.</td>
      <td>Positive integer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>height</td>
      <td>Input</td>
      <td>Image height.</td>
      <td>Positive integer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nearPlane</td>
      <td>Input</td>
      <td>Near plane distance.</td>
      <td>Floating-point number, typically 0.01.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>farPlane</td>
      <td>Input</td>
      <td>Far plane distance.</td>
      <td>Floating-point number, typically 100.0.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>supportFeatures</td>
      <td>Input</td>
      <td>Feature switch bitmask.</td>
      <td>Defaults to 0. bit0: enable depth correction (viCulling is weighted by vi×(1−depths/sigma)); other bits are reserved.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meansCulling</td>
      <td>Output</td>
      <td>Filtered 3D positions (compacted along the gaussian dimension).</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>colorsCulling</td>
      <td>Output</td>
      <td>Filtered colors.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, color_dims, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>means2dCulling</td>
      <td>Output</td>
      <td>Filtered 2D positions.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depthsCulling</td>
      <td>Output</td>
      <td>Filtered depths.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>radiusCulling</td>
      <td>Output</td>
      <td>Filtered radii.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars2dCulling</td>
      <td>Output</td>
      <td>Filtered 2D covariances.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>conicsCulling</td>
      <td>Output</td>
      <td>Filtered conic parameters.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>opacitiesCulling</td>
      <td>Output</td>
      <td>Filtered opacities.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>quatsCulling</td>
      <td>Output</td>
      <td>Filtered quaternions.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>scalesCulling</td>
      <td>Output</td>
      <td>Filtered scale factors.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>viCulling</td>
      <td>Output</td>
      <td>Filtered depth correction weights.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid. When depth correction is enabled, weighted by vi×(1−depths/sigma).</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>compensationsCulling</td>
      <td>Output</td>
      <td>Filtered opacity compensation values.</td>
      <td>Empty tensors are not supported. Only the first cnt positions are valid.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>Output</td>
      <td>Filter flags (bit-packed UINT8 bitmap).</td>
      <td>Empty tensors are not supported. Bit i being 1 indicates the i-th Gaussian sphere passes the filter.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ceil(gaussian_num/8))</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>Output</td>
      <td>Number of filtered Gaussian spheres.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>filterFloat</td>
      <td>Output</td>
      <td>Float mask (0/1) of the filter flags.</td>
      <td>Empty tensors are not supported. The i-th element being 1 indicates the i-th Gaussian sphere passes the filter.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>denseIndices</td>
      <td>Output</td>
      <td>Mapping from compacted positions to original dense Gaussian sphere indices.</td>
      <td>Empty tensors are not supported. The k-th element is the original index of the k-th valid Gaussian sphere in the compacted layout.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
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
                <td>The data type, format, or shape of the input/output tensor is not within the supported range, or the parameter/attribute values are invalid.</td>
            </tr>
        </tbody>
    </table>

## aclnnGaussianFilter

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnGaussianFilterGetWorkspaceSize.</td>
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

- Data types: `means`, `colors`, `det`, `opacities`, `means2d`, `depths`, `radius`, `conics`, `covars2d`, `compensations`, `quats`, `scales`, `vi`, `sigma`, all `*Culling` outputs, and `filterFloat` are FLOAT; `filter` is UINT8; `cnt` is FLOAT; `denseIndices` is INT32. All data formats are ND.
- `compensations`, `quats`, `scales`, `vi`, and `sigma` are optional inputs and can pass nullptr.
- `width` and `height` must be positive integers; `nearPlane` and `farPlane` are floating-point numbers; `supportFeatures` is a bitmask (defaults to 0).
- `supportFeatures` bit definitions:
  - bit0 (1): enable depth correction, `viCulling = vi × max(1 − depths / sigma, 0)`; when disabled, `viCulling = vi`.
  - Other bits are reserved.
- The filtering result is a compact layout along the gaussian dimension: the last dimension of each `*Culling` output and `denseIndices` is still `gaussian_num`, but only the first `cnt` positions are valid for each (batch, camera) view.
- The actually written (valid) outputs differ per product path:
  - Atlas A2/A3: outputs `meansCulling`, `colorsCulling`, `means2dCulling`, `depthsCulling`, `radiusCulling`, `covars2dCulling`, `conicsCulling`, `opacitiesCulling`, `filter`, and `cnt`; `quatsCulling`, `scalesCulling`, `viCulling`, `compensationsCulling`, `filterFloat`, and `denseIndices` are reserved outputs.
  - A5 (Ascend 950): outputs all 16 outputs; `filter` is a bit-packed bitmap and `filterFloat` is the corresponding float mask.
- All outputs are required outputs, and all of them must be passed as valid tensors to the first-stage interface.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.
- color_dims: Number of color channels, usually 3, and 7 when extended features are enabled.

## Calling Example

See tests/kernel_tests/test_gaussian_filter.cpp
