# aclnnFullyFusedProjectionBwd

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fully_fused_projection_bwd/op_kernel/fully_fused_projection_bwd.cpp)

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

- Interface function: `fully_fused_projection_bwd` calculates the backward propagation of fully fused projection, used to compute gradients of parameters during training.
- Computation formula:

Backward propagation calculates gradients of position, quaternion, scale, color, opacity, and other parameters:

$$
v\_means = \frac{\partial L}{\partial means}, \quad v\_quats = \frac{\partial L}{\partial quats}
$$

$$
v\_scales = \frac{\partial L}{\partial scales}, \quad v\_colors = \frac{\partial L}{\partial colors}
$$

$$
v\_opacities = \frac{\partial L}{\partial opacities}
$$

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
    int64_t          width,
    int64_t          height,
    const aclTensor *vPW,
    const aclTensor *vQuats,
    const aclTensor *vScales,
    const aclTensor *vR,
    const aclTensor *vColors,
    const aclTensor *vOpacities,
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
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>quats</td>
      <td>Input</td>
      <td>Quaternions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>Scale factors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
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
      <td>viewmats</td>
      <td>Input</td>
      <td>View transformation matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>Ks</td>
      <td>Input</td>
      <td>Camera intrinsic matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vMeans2d</td>
      <td>Input</td>
      <td>Gradient of 2D positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vDepths</td>
      <td>Input</td>
      <td>Gradient of depths.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vConics</td>
      <td>Input</td>
      <td>Gradient of conic parameters.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vColorsCulling</td>
      <td>Input</td>
      <td>Gradient of filtered colors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vOpacitiesCulling</td>
      <td>Input</td>
      <td>Gradient of filtered opacities.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>Input</td>
      <td>Filter flags.</td>
      <td>Empty tensors are not supported.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ceil(gaussian_num/8))</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>Input</td>
      <td>Compensation values (optional).</td>
      <td>Optional input, can pass nullptr.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
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
      <td>vPW</td>
      <td>Output</td>
      <td>Gradient of projection matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vQuats</td>
      <td>Output</td>
      <td>Gradient of quaternions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 4)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vScales</td>
      <td>Output</td>
      <td>Gradient of scale factors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vR</td>
      <td>Output</td>
      <td>Gradient of rotation matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vColors</td>
      <td>Output</td>
      <td>Gradient of colors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vOpacities</td>
      <td>Output</td>
      <td>Gradient of opacities.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
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
                <td>The data type and format of the input/output tensor are not within the supported range, or the parameter values are invalid.</td>
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

- The data types of most input and output tensors are FLOAT.
- The data type of filter is UINT8.
- compensations is an optional input and can pass nullptr.
- width and height must be positive integers.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.

## Calling Example

See tests/kernel_tests/test_fully_fused_projection_bwd.cpp
