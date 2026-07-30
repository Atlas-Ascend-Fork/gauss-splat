# aclnnSphericalHarmonicsBwd

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_bwd/op_kernel/spherical_harmonics_bwd.cpp)

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

- Interface function: `spherical_harmonics_bwd` calculates the backward propagation of spherical harmonics, used for gradient computation during 3D Gaussian Splatting training.
- Computation formula:

Backward propagation calculates gradients of directions and coefficients:

$$
v\_dirs = \frac{\partial L}{\partial dirs}, \quad v\_coeffs = \frac{\partial L}{\partial coeffs}
$$

The backward gradient computation is based on the forward formula $output = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_{lm}(dirs)$.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnSphericalHarmonicsBwdGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnSphericalHarmonicsBwd" interface to execute the computation.

```Cpp
aclnnStatus aclnnSphericalHarmonicsBwdGetWorkspaceSize(
    const aclTensor *dirs,
    const aclTensor *coeffs,
    const aclTensor *vColors,
    int64_t          degree,
    const aclTensor *vDirs,
    const aclTensor *vCoeffs,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSphericalHarmonicsBwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnSphericalHarmonicsBwdGetWorkspaceSize

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
      <td>dirs</td>
      <td>Input</td>
      <td>Direction vectors, representing ray directions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>coeffs</td>
      <td>Input</td>
      <td>Spherical harmonics coefficients, used for color computation.</td>
      <td>Empty tensors are not supported. K must equal (degree+1)*(degree+1).</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, K, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vColors</td>
      <td>Input</td>
      <td>Gradient input for colors.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>degree</td>
      <td>Input</td>
      <td>Spherical harmonics degree used.</td>
      <td>Supports 0 to 3.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vDirs</td>
      <td>Output</td>
      <td>Gradient output for directions.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>vCoeffs</td>
      <td>Output</td>
      <td>Gradient output for coefficients.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, K, 3, gaussian_num)</td>
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
                <td>The data types and formats of dirs, coeffs, vColors, vDirs, and vCoeffs are not within the supported range, or degree is not within the range of 0 to 3.</td>
            </tr>
        </tbody>
    </table>

## aclnnSphericalHarmonicsBwd

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnSphericalHarmonicsBwdGetWorkspaceSize.</td>
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

See tests/kernel_tests/test_spherical_harmonics_bwd.cpp
