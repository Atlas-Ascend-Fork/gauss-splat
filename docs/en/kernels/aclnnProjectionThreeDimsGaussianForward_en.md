# aclnnProjectionThreeDimsGaussianForward

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/projection_three_dims_gaussian_forward/op_kernel/projection_three_dims_gaussian_forward.cpp)

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

- Interface function: `projection_three_dims_gaussian_forward` projects 3D Gaussian spheres onto the 2D image plane, computing the projected position, depth, conic parameters, and so on.
- Computation formula:

Projects 3D Gaussian spheres onto the 2D plane through the camera matrix:

$$
means2d = K \cdot (viewmat \cdot means)
$$

Projection of the covariance matrix:

$$
covars2d = J \cdot viewmat \cdot covars \cdot viewmat^T \cdot J^T
$$

Where $J$ is the Jacobian matrix, $K$ is the camera intrinsic matrix, and $viewmat$ is the view transformation matrix.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnProjectionThreeDimsGaussianForward" interface to execute the computation.

```Cpp
aclnnStatus aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize(
    const aclTensor *means,
    const aclTensor *covars,
    const aclTensor *viewmats,
    const aclTensor *ks,
    int64_t          imageWidth,
    int64_t          imageHeight,
    float            eps2d,
    bool             calcCompensations,
    int64_t          cameraModel,
    const aclTensor *means2d,
    const aclTensor *depths,
    const aclTensor *conics,
    const aclTensor *compensations,
    const aclTensor *det,
    const aclTensor *radius,
    const aclTensor *covars2d,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnProjectionThreeDimsGaussianForward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize

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
      <td>3D Gaussian sphere position coordinates.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars</td>
      <td>Input</td>
      <td>3D Gaussian sphere covariance matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>viewmats</td>
      <td>Input</td>
      <td>View transformation matrices (camera extrinsics).</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>ks</td>
      <td>Input</td>
      <td>Camera intrinsic matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>imageWidth</td>
      <td>Input</td>
      <td>Image width.</td>
      <td>Positive integer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>imageHeight</td>
      <td>Input</td>
      <td>Image height.</td>
      <td>Positive integer.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps2d</td>
      <td>Input</td>
      <td>Epsilon value for 2D covariance to prevent numerical instability.</td>
      <td>Recommended value is 0.3.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>calcCompensations</td>
      <td>Input</td>
      <td>Whether to calculate compensation values.</td>
      <td>true or false.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cameraModel</td>
      <td>Input</td>
      <td>Camera model type.</td>
      <td>0 indicates pinhole camera model.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>means2d</td>
      <td>Output</td>
      <td>Projected 2D position coordinates.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>Output</td>
      <td>Projected depth values.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>Output</td>
      <td>Conic parameters for rendering computation.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>Output</td>
      <td>Compensation value output.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>det</td>
      <td>Output</td>
      <td>Covariance determinant values.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>Output</td>
      <td>Gaussian sphere radii on the screen.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars2d</td>
      <td>Output</td>
      <td>Projected 2D covariance matrices.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
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

## aclnnProjectionThreeDimsGaussianForward

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize.</td>
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

## Calling Example

See tests/kernel_tests/test_projection_three_dims_gaussian_forward.cpp
