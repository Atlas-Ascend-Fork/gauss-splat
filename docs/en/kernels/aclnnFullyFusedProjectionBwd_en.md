# aclnnFullyFusedProjectionBwd

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fully_fused_projection_bwd/op_kernel/arch22/fully_fused_projection_bwd.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fully_fused_projection_bwd/op_kernel/arch35/fully_fused_projection_bwd.cpp)

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

- Interface function: `fully_fused_projection_bwd` calculates the backward propagation of fully fused projection, used to compute gradients of parameters during training. The inputs are the gradients of the outputs produced by the forward projection and culling, and the outputs are the gradients of the Gaussian sphere position, quaternion, scale, color, and opacity, as well as the gradients of the camera pose (view matrix, intrinsics) and depth-correction parameters (`vi`, `sigma`).
- Computation formula:

Gradients of position, quaternion, and scale:

$$
v\_pW = \frac{\partial L}{\partial means}, \quad v\_quats = \frac{\partial L}{\partial quats}
$$

$$
v\_scales = \frac{\partial L}{\partial scales}
$$

Gradients of color and opacity (the opacity gradient is weighted by `compensations` and accumulated over the camera dimension):

$$
v\_colors = \frac{\partial L}{\partial colors}, \quad v\_opacities = \frac{\partial L}{\partial opacities}
$$

Gradients of the camera pose and depth-correction parameters:

$$
v\_R = \frac{\partial L}{\partial R}, \quad v\_viewmats = \frac{\partial L}{\partial viewmat}, \quad v\_Ks = \frac{\partial L}{\partial K}
$$

$$
dL\_dvi = \frac{\partial L}{\partial vi}, \quad dL\_dsigma = \frac{\partial L}{\partial \sigma}
$$

where $L$ is the loss, $R$ is the rotation part of the view transformation matrix, $\sigma$ is the depth-correction scale (`sigma`), and $vi$ is the per-view depth-correction weight.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnFullyFusedProjectionBwdGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnFullyFusedProjectionBwd" interface to execute the computation.

```Cpp
aclnnStatus aclnnFullyFusedProjectionBwdGetWorkspaceSize(
    const aclTensor *means,
    const aclTensor *quats,
    const aclTensor *scales,
    const aclTensor *conics,
    const aclTensor *viewmats,
    const aclTensor *Ks,
    const aclTensor *vMeans2d,
    const aclTensor *vDepths,
    const aclTensor *vConics,
    const aclTensor *vColorsCulling,
    const aclTensor *vOpacitiesCulling,
    const aclTensor *filter,
    const aclTensor *compensations,
    const aclTensor *vViCulling,
    const aclTensor *opacities,
    const aclTensor *vi,
    const aclTensor *sigma,
    const aclTensor *depths,
    const aclTensor *denseIndices,
    const aclTensor *cnt,
    int64_t          width,
    int64_t          height,
    int64_t          maxCompactCount,
    int64_t          supportFeatures,
    const aclTensor *vPW,
    const aclTensor *vQuats,
    const aclTensor *vScales,
    const aclTensor *vR,
    const aclTensor *vColors,
    const aclTensor *vOpacities,
    const aclTensor *vViewmats,
    const aclTensor *vKs,
    const aclTensor *dLdvi,
    const aclTensor *dLdsigma,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFullyFusedProjectionBwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnFullyFusedProjectionBwdGetWorkspaceSize

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
      <td>Empty tensors are not supported. When culling is enabled (supportFeatures bit2), it is the culled compact layout with the (batch, camera) dimensions merged.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num) or (batch_size*camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>quats</td>
      <td>Input</td>
      <td>Quaternions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num) or (batch_size*camera_num, 4, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>Scale factors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num) or (batch_size*camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>Input</td>
      <td>Conic parameters (inverse of the 2D covariance matrix).</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num) or (batch_size*camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>viewmats</td>
      <td>Input</td>
      <td>View transformation matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4) or (batch_size*camera_num, 16)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>Ks</td>
      <td>Input</td>
      <td>Camera intrinsic matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3) or (batch_size*camera_num, 9)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vMeans2d</td>
      <td>Input</td>
      <td>Gradient of the filtered 2D positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num) or (batch_size*camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vDepths</td>
      <td>Input</td>
      <td>Gradient of the filtered depths.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vConics</td>
      <td>Input</td>
      <td>Gradient of the filtered conic parameters.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num) or (batch_size*camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vColorsCulling</td>
      <td>Input</td>
      <td>Gradient of the filtered colors.</td>
      <td>Empty tensors are not supported. The channel count is 6 when alpha max and gradient absolute value output is enabled (supportFeatures bit3), otherwise 3.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, color_dims, gaussian_num) or (batch_size*camera_num, color_dims, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vOpacitiesCulling</td>
      <td>Input</td>
      <td>Gradient of the filtered opacities.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>Input</td>
      <td>Filter flags.</td>
      <td>Empty tensors are not supported. On Atlas A2/A3 it is a bit-packed UINT8 bitmap; on A5 (Ascend 950) it is a FLOAT mask.</td>
      <td>UINT8 (A2/A3) / FLOAT (A5)</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ceil(gaussian_num/8)) (A2/A3) or (batch_size, camera_num, gaussian_num) (A5)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>Input</td>
      <td>Opacity compensation values.</td>
      <td>Optional input, can pass nullptr; when not null it is used to weight the opacity gradient.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vViCulling</td>
      <td>Input</td>
      <td>Gradient of the filtered depth-correction weight `vi`.</td>
      <td>Optional input, can pass nullptr; used when depth correction is enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>opacities</td>
      <td>Input</td>
      <td>Filtered raw opacities.</td>
      <td>Optional input, can pass nullptr.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vi</td>
      <td>Input</td>
      <td>Filtered depth-correction weight (the per-view weight of each Gaussian sphere in LCWSR).</td>
      <td>Optional input, can pass nullptr; used when depth correction is enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>Input</td>
      <td>Depth-correction scale parameter (scalar).</td>
      <td>Optional input, can pass nullptr; used when depth correction is enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1,)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>Input</td>
      <td>Filtered Gaussian sphere depths.</td>
      <td>Optional input, can pass nullptr; used when depth correction is enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) or (batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>denseIndices</td>
      <td>Input</td>
      <td>Mapping indices from the culled compact positions to the original dense Gaussian sphere indices.</td>
      <td>Optional input, can pass nullptr; used when the culled compact layout is enabled (supportFeatures bit2).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size*camera_num, max_compact_count)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>Input</td>
      <td>Number of valid Gaussian spheres after culling for each (batch, camera) view.</td>
      <td>Optional input, can pass nullptr; used when the culled compact layout is enabled.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size*camera_num,)</td>
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
      <td>maxCompactCount</td>
      <td>Input</td>
      <td>Maximum number of Gaussian spheres after culling over all views, i.e. the number of columns of the compact buffer.</td>
      <td>Positive integer, default is 1; when the culled compact layout is enabled, it must match the maximum value of cnt.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>supportFeatures</td>
      <td>Input</td>
      <td>Feature switch bitmask.</td>
      <td>Default is 0; bit0: enable depth correction; bit1: cbs backend; bit2: projection backward is culled (inputs use the compact layout with the batch/camera dimensions merged into batch_size*camera_num); bit3: additionally output alpha max and gradient absolute value (color_dims changes from 3 to 6).</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vPW</td>
      <td>Output</td>
      <td>Gradient of the Gaussian sphere positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 3) (A2/A3) or (batch_size, 3, gaussian_num) (A5)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vQuats</td>
      <td>Output</td>
      <td>Gradient of quaternions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 4) (A2/A3) or (batch_size, 4, gaussian_num) (A5)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vScales</td>
      <td>Output</td>
      <td>Gradient of scale factors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 3) (A2/A3) or (batch_size, 3, gaussian_num) (A5)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vR</td>
      <td>Output</td>
      <td>Gradient of rotation matrices.</td>
      <td>Empty tensors are not supported. (Reserved output on the A5 path.)</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vColors</td>
      <td>Output</td>
      <td>Gradient of colors. When bit3 is enabled, alpha max and gradient absolute value are additionally output.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, color_dims, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vOpacities</td>
      <td>Output</td>
      <td>Gradient of opacities (after accumulation over the camera dimension).</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vViewmats</td>
      <td>Output</td>
      <td>Gradient of the view transformation matrices (for pose optimization).</td>
      <td>Empty tensors are not supported. (Reserved output.)</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vKs</td>
      <td>Output</td>
      <td>Gradient of the camera intrinsic matrices.</td>
      <td>Empty tensors are not supported. (Reserved output.)</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>dLdvi</td>
      <td>Output</td>
      <td>Gradient of the depth-correction weight `vi`.</td>
      <td>Empty tensors are not supported; valid when depth correction is enabled.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size*camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>dLdsigma</td>
      <td>Output</td>
      <td>Gradient of the depth-correction scale `sigma` (scalar).</td>
      <td>Empty tensors are not supported; valid when depth correction is enabled, only the 0th element is meaningful.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1,)</td>
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
                <td>The data type, data format, or shape of the input/output tensor is not within the supported range, or the parameter value/attribute value is invalid.</td>
            </tr>
        </tbody>
    </table>

## aclnnFullyFusedProjectionBwd

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnFullyFusedProjectionBwdGetWorkspaceSize.</td>
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

- Data types: `means`, `quats`, `scales`, `conics`, `viewmats`, `Ks`, `vMeans2d`, `vDepths`, `vConics`, `vColorsCulling`, `vOpacitiesCulling`, `compensations`, `vViCulling`, `opacities`, `vi`, `sigma`, `depths`, and all outputs are FLOAT; `denseIndices` and `cnt` are INT32; `filter` is UINT8 (bitmap) on Atlas A2/A3 and FLOAT on A5 (Ascend 950). The data format is ND for all.
- `compensations`, `vViCulling`, `opacities`, `vi`, `sigma`, `depths`, `denseIndices`, and `cnt` are optional inputs and can pass nullptr.
- `width` and `height` must be positive integers; `maxCompactCount` is a positive integer (default 1); `supportFeatures` is a bitmask (default 0).
- `supportFeatures` bit definitions:
  - bit0 (1): enable depth correction, which uses `vi`, `sigma`, `depths`, `vViCulling` and outputs `dLdvi`, `dLdsigma`.
  - bit1 (2): cbs backend.
  - bit2 (4): projection backward is culled; the inputs (`means`, `quats`, `scales`, `conics`, `viewmats`, `Ks`, and the gradients) use the culled compact layout with the (batch, camera) dimensions merged into `batch_size*camera_num`, and `denseIndices` and `cnt` must be provided.
  - bit3 (8): additionally output alpha max and gradient absolute value; the channel count (color_dims) of `vColorsCulling` and `vColors` changes from 3 to 6.
- When bit2 is set, `denseIndices` and `cnt` are used to map the compact positions back to the original dense Gaussian sphere indices; the value range of each `cnt` element is `[0, gaussian_num]`, and `maxCompactCount` must be no less than the maximum value of `cnt`.
- The outputs actually written (valid) differ by product path:
  - Atlas A2/A3: outputs `vPW`, `vQuats`, `vScales`, `vR`, `vColors`, `vOpacities`; `vViewmats`, `vKs`, `dLdvi`, `dLdsigma` are reserved outputs (not written).
  - A5 (Ascend 950): outputs `vPW`, `vQuats`, `vScales`, `vColors`, `vOpacities`, `dLdvi`, `dLdsigma`; `vR`, `vViewmats`, `vKs` are reserved outputs (not written).
- All outputs are required outputs; the first-stage interface must be passed valid tensors for all of them.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.
- color_dims: Number of color channels, default is 3, and becomes 6 when `supportFeatures` bit3 is enabled.

## Calling Example

See tests/kernel_tests/test_fully_fused_projection_bwd.cpp
