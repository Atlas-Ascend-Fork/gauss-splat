# aclnnGaussianFilter

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_filter/op_kernel/gaussian_filter.cpp)

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

- Interface function: `gaussian_filter` filters Gaussian spheres based on view frustum parameters, culling Gaussian spheres that are not within the visible range.
- Computation formula:

Filters Gaussian spheres based on near_plane, far_plane, image boundaries, and other parameters:

$$
filter[i] = \begin{cases} 1 & \text{if } depth_i \in [near\_plane, far\_plane] \text{ and } means2d_i \in image \\ 0 & \text{otherwise} \end{cases}
$$

The filtered output maintains the original data order but retains only visible Gaussian spheres.

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
    int64_t          width,
    int64_t          height,
    float            nearPlane,
    float            farPlane,
    const aclTensor *meansCulling,
    const aclTensor *colorsCulling,
    const aclTensor *means2dCulling,
    const aclTensor *depthsCulling,
    const aclTensor *radiusCulling,
    const aclTensor *covars2dCulling,
    const aclTensor *conicsCulling,
    const aclTensor *opacitiesCulling,
    const aclTensor *filter,
    const aclTensor *cnt,
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
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
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
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
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
      <td>Compensation values (optional).</td>
      <td>Optional input, can pass nullptr.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
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
      <td>meansCulling</td>
      <td>Output</td>
      <td>Filtered 3D positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>colorsCulling</td>
      <td>Output</td>
      <td>Filtered colors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>means2dCulling</td>
      <td>Output</td>
      <td>Filtered 2D positions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depthsCulling</td>
      <td>Output</td>
      <td>Filtered depths.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>radiusCulling</td>
      <td>Output</td>
      <td>Filtered radii.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars2dCulling</td>
      <td>Output</td>
      <td>Filtered 2D covariances.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>conicsCulling</td>
      <td>Output</td>
      <td>Filtered conic parameters.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>opacitiesCulling</td>
      <td>Output</td>
      <td>Filtered opacities.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>Output</td>
      <td>Filter flags.</td>
      <td>Empty tensors are not supported.</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>Output</td>
      <td>Number of filtered Gaussian spheres.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num)</td>
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

- The data types of most input and output tensors are FLOAT.
- The data type of filter is UINT8, and the data type of cnt is INT32.
- compensations is an optional input and can pass nullptr.
- width and height must be positive integers. nearPlane and farPlane are floating-point numbers.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.

## Calling Example

See tests/kernel_tests/test_gaussian_filter.cpp
