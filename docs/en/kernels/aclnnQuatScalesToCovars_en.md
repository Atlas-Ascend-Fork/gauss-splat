# aclnnQuatScalesToCovars

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/quat_scales_to_covars/op_kernel/arch22/quat_scales_to_covars.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/quat_scales_to_covars/op_kernel/arch35/quat_scales_to_covars.cpp)

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

- Interface function: `quat_scales_to_covars` converts quaternions and scale factors into 3D covariance matrices for Gaussian sphere representation in 3D Gaussian Splatting.
- Computation formula:

Covariance matrix computation formula:

$$
\Sigma = R \cdot S \cdot S^T \cdot R^T
$$

Where $R$ is the rotation matrix converted from the quaternion $q$, and $S$ is the scale matrix:

$$
R = \text{QuaternionToRotationMatrix}(q)
$$

$$
S = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & s_z \end{bmatrix}
$$

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnQuatScalesToCovarsGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnQuatScalesToCovars" interface to execute the computation.

```Cpp
aclnnStatus aclnnQuatScalesToCovarsGetWorkspaceSize(
    const aclTensor *quat,
    const aclTensor *scales,
    const aclTensor *covars,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuatScalesToCovars(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnQuatScalesToCovarsGetWorkspaceSize

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
      <td>quat</td>
      <td>Input</td>
      <td>Quaternion, representing the rotation of the Gaussian sphere.</td>
      <td>Empty tensors are not supported. Components are ordered as $(w, x, y, z)$ (real part first); the kernel normalizes the quaternion internally, so the caller does not need to pre-normalize it.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>Input</td>
      <td>Scale factors, representing the scale of the Gaussian sphere along each axis.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars</td>
      <td>Output</td>
      <td>3D covariance matrix output.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, 3, gaussian_num)</td>
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
                <td>The data types and formats of quat, scales, and covars are not within the supported range.</td>
            </tr>
        </tbody>
    </table>

## aclnnQuatScalesToCovars

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnQuatScalesToCovarsGetWorkspaceSize.</td>
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

- Data types: `quat`, `scales`, and `covars` are FLOAT; the data format is ND for all.
- Empty tensors are not supported for `quat`, `scales`, or `covars`; non-contiguous tensors are supported.
- The batch_size and gaussian_num of `quat` and `scales` must match; the shape of the output `covars` is inferred from `quat` and is (batch_size, 3, 3, gaussian_num).
- The component order of `quat` is $(w, x, y, z)$ (real part first); the kernel normalizes the quaternion internally, so the caller does not need to pre-normalize it.
- Both Atlas A2/A3 and A5 (Ascend 950) support this operator; `gaussian_num` is split across AIV cores after being aligned to 8 elements.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at a time.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.

## Calling Example

See tests/kernel_tests/test_quat_scales_to_covars.cpp
