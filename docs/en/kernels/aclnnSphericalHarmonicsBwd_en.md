# aclnnSphericalHarmonicsBwd

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_bwd/op_kernel/arch22/spherical_harmonics_bwd.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_bwd/op_kernel/arch35/spherical_harmonics_bwd.cpp)

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

- Interface function: `spherical_harmonics_bwd` calculates the backward propagation of the forward operator `spherical_harmonics_forward`, used for gradient computation during 3D Gaussian Splatting training. Given the forward inputs `dirs`, `coeffs` and the color gradient `vColors`, it outputs the direction gradient `vDirs` and the coefficient gradient `vCoeffs`.
- Computation formula:

Let the forward degree be $L$ (`degree`), the number of coefficients $K=(L+1)^2$, the number of channels be 3, and the unit direction be $\hat{d} = d / \lVert d \rVert$. The forward computation is:

$$
output_{k,c} = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_{lm}(\hat{d})
$$

where $c_{lm}$ are the spherical harmonics coefficients (`coeffs`) and $Y_{lm}$ are the spherical harmonics basis functions. Let $v_{c} = \dfrac{\partial L}{\partial output_{k,c}}$ be the color gradient (`vColors`); the coefficient gradient is:

$$
v\_coeffs_{k,c} = Y_{k}(\hat{d}) \cdot v_{c}
$$

The direction gradient first accumulates the contributions of each degree's basis derivative:

$$
\tilde{g} = \sum_{k,c} \text{coeffs}_{k,c} \cdot v_{c} \cdot \nabla Y_{k}(\hat{d})
$$

and is then projected back to the original direction vector through the normalization Jacobian:

$$
v\_dirs = \frac{\tilde{g} - \hat{d}\,\left(\hat{d} \cdot \tilde{g}\right)}{\lVert d \rVert}
$$

The direction normalization is performed inside the operator; when $\lVert d \rVert$ is too small it is clamped to $10^{-12}$ to avoid division by zero. When `degree` is 0, $Y_0$ is a constant, so $v\_dirs$ is all zeros and $v\_coeffs = 0.28209479177387814 \cdot vColors$.

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
      <td>Supports 0 to 4.</td>
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
                <td>The data types and formats of dirs, coeffs, vColors, vDirs, and vCoeffs are not within the supported range, degree is not within the range of 0 to 4, or K of coeffs does not equal (degree+1)*(degree+1).</td>
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

## Constraints

- Data type: `dirs`, `coeffs`, `vColors`, `vDirs`, and `vCoeffs` are all FLOAT; data format is ND for all.
- `dirs`, `coeffs`, `vColors`, `vDirs`, and `vCoeffs` do not support empty tensors and support non-contiguous tensors.
- `degree` supports 0 to 4; the second dimension `K` of `coeffs` must equal `(degree+1)*(degree+1)`, and the channel dimension is fixed to 3.
- The batchSize and gaussian_num of `dirs`, `coeffs`, and `vColors` must be consistent; the shape of `vDirs` matches `dirs`, and the shape of `vCoeffs` matches `coeffs`.
- Direction vectors are normalized inside the operator, so callers do not need to pre-normalize; when the direction norm is too small it is clamped to $10^{-12}$.
- gaussian_num is split across AIV cores for parallel computation; a larger `degree` gives a larger K, reducing the number of gaussians processed per core.
- This operator is supported on both Atlas A2/A3 and A5 (Ascend 950).

**Dimension Variable Description:**

- batchSize: Batch size, the number of data batches processed at a time.
- gaussian_num: Number of gaussians, the number of 3D gaussians in each batch.
- K: Number of spherical harmonics coefficients, equal to $(degree+1)^2$, ranging from 1 (degree=0) to 25 (degree=4).
- Channel count: Number of color channels, fixed to 3.

## Calling Example

See tests/kernel_tests/test_spherical_harmonics_bwd.cpp
