# aclnnSphericalHarmonicsForward

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_forward/op_kernel/arch22/spherical_harmonics_forward.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_forward/op_kernel/arch35/spherical_harmonics_forward.cpp)

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

- Interface function: `spherical_harmonics_forward` calculates the forward propagation of spherical harmonics, used for color rendering in 3D Gaussian Splatting.
- Computation formula:

Spherical harmonics maps direction vectors to color space, with the following formula:

$$
output = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_{lm}(dirs)
$$

Where $Y_{lm}$ is the spherical harmonics basis function of degree $l$ and order $m$, $c_{lm}$ is the spherical harmonics coefficient, and $dirs$ is the direction vector.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnSphericalHarmonicsForwardGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnSphericalHarmonicsForward" interface to execute the computation.

```Cpp
aclnnStatus aclnnSphericalHarmonicsForwardGetWorkspaceSize(
    const aclTensor *dirs,
    const aclTensor *coeffs,
    int64_t          degreesToUse,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSphericalHarmonicsForward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnSphericalHarmonicsForwardGetWorkspaceSize

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
      <td>Empty tensors are not supported. The number of coefficients K is related to degrees_to_use, and the channel count C supports 1 to 4, usually 3.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, K, C, gaussian_num), where K=(degrees_to_use+1)*(degrees_to_use+1)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>degreesToUse</td>
      <td>Input</td>
      <td>Spherical harmonics degree used.</td>
      <td>Supports 0 to 4. Using 2, 3, or 4 is recommended.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Computed color output.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, C, gaussian_num)</td>
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
                <td>The data types and formats of dirs, coeffs, and output are not within the supported range, or degrees_to_use is not within the range of 0 to 4, or the channel dimension C of coeffs is not within the range of 1 to 4.</td>
            </tr>
        </tbody>
    </table>

## aclnnSphericalHarmonicsForward

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnSphericalHarmonicsForwardGetWorkspaceSize.</td>
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

- Data types: `dirs`, `coeffs`, and `output` are all FLOAT; data formats are all ND.
- `dirs`, `coeffs`, and `output` do not support empty tensors, and support non-contiguous tensors.
- `degrees_to_use` supports 0 to 4; the second dimension `K` of `coeffs` must equal `(degrees_to_use+1)*(degrees_to_use+1)`, and the channel dimension `C` supports 1 to 4, usually 3.
- The batchSize and gaussian_num of `dirs` and `coeffs` must be consistent; the shape of `output` is `(batchSize, C, gaussian_num)`.
- Direction vectors are normalized inside the operator; the caller does not need to normalize them beforehand.
- gaussian_num is split across AIV cores for parallel computation. A larger `degrees_to_use` means a larger K, and the number of gaussians a single core can process decreases accordingly.
- Both Atlas A2/A3 and A5 (Ascend 950) support this operator.

**Dimension Variable Description:**

- batchSize: Batch size, the number of data batches processed at a time.
- gaussian_num: Number of gaussians, the number of 3D gaussians in each batch.
- K: Number of spherical harmonics coefficients, equal to $(degrees\_to\_use+1)^2$, ranging from 1 (degree=0) to 25 (degree=4).
- C: Number of color channels, supports 1 to 4, usually 3.

## Calling Example

See tests/kernel_tests/test_spherical_harmonics_forward.cpp
