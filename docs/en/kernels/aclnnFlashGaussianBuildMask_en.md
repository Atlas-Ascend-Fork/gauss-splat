# aclnnFlashGaussianBuildMask

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/flash_gaussian_build_mask/op_kernel/flash_gaussian_build_mask.cpp)

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

- Interface function: `flash_gaussian_build_mask` builds the mask information required for Flash rendering to optimize rendering efficiency.
- Computation formula:

Builds tile masks based on Gaussian sphere position, opacity, and other parameters:

$$
tile\_sum = \sum_{gaussians\_in\_tile} valid\_mask
$$

$$
tile\_depths = \sum_{gaussians\_in\_tile} depth \cdot valid\_mask
$$

The mask construction process optimizes the parallel computing efficiency of subsequent rendering.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnFlashGaussianBuildMaskGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnFlashGaussianBuildMask" interface to execute the computation.

```Cpp
aclnnStatus aclnnFlashGaussianBuildMaskGetWorkspaceSize(
    const aclTensor *means2d,
    const aclTensor *opacity,
    const aclTensor *conics,
    const aclTensor *covars2d,
    const aclTensor *depths,
    const aclTensor *cnt,
    const aclTensor *tileGrid,
    const aclTensor *gatherMask,
    float            imageWidth,
    float            imageHeight,
    int64_t          tileSize,
    const aclTensor *tileSum,
    const aclTensor *tileDepths,
    const aclTensor *gaussIndex,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFlashGaussianBuildMask(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnFlashGaussianBuildMaskGetWorkspaceSize

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
      <td>means2d</td>
      <td>Input</td>
      <td>Projected 2D position coordinates.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>opacity</td>
      <td>Input</td>
      <td>Gaussian sphere opacity.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
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
      <td>2D covariance matrix.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>Input</td>
      <td>Gaussian sphere depth.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>Input</td>
      <td>Gaussian sphere count per tile.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileGrid</td>
      <td>Input</td>
      <td>Tile grid information.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(numTile, 2)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gatherMask</td>
      <td>Input</td>
      <td>Gather mask information.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(48, 20, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>imageWidth</td>
      <td>Input</td>
      <td>Image width.</td>
      <td>Floating-point number.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>imageHeight</td>
      <td>Input</td>
      <td>Image height.</td>
      <td>Floating-point number.</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSize</td>
      <td>Input</td>
      <td>Tile size.</td>
      <td>Positive integer, typically 16.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSum</td>
      <td>Output</td>
      <td>Total Gaussian sphere count per tile.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, 1)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileDepths</td>
      <td>Output</td>
      <td>Depth information per tile.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gaussIndex</td>
      <td>Output</td>
      <td>Gaussian sphere index information.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, gaussian_num)</td>
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

## aclnnFlashGaussianBuildMask

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnFlashGaussianBuildMaskGetWorkspaceSize.</td>
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

- The data types of means2d, opacity, conics, covars2d, depths, tileGrid, gatherMask, tileDepths, and gaussIndex are FLOAT.
- The data types of cnt and tileSum are INT32.
- tileSize is typically 16, used for tile partition computation.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.
- numTile: Number of tiles, indicating the total number of tile blocks the image is divided into.

## Calling Example

See tests/kernel_tests/test_flash_gaussian_build_mask.cpp
