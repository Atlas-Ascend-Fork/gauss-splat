# aclnnFlashGaussianBuildMask

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/flash_gaussian_build_mask/op_kernel/arch22/flash_gaussian_build_mask.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/flash_gaussian_build_mask/op_kernel/arch35/flash_gaussian_build_mask.cpp)

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

- Interface function: `flash_gaussian_build_mask` performs three-level filtering and ordered compaction on the Gaussian spheres under each (batch, camera) view by tile assignment, building the tile masks required by Flash rendering (the count, index list, and depth list of valid Gaussian spheres in each tile), to optimize the parallel computing efficiency of subsequent rendering.
- Computation formula:

For each (batch, camera, tile):

$$
tile\_sum[b,c,t] = \sum_{g} valid\_mask[b,c,t,g]
$$

$$
gauss\_index[b,c,t,0:tile\_sum] = \{\, g \mid valid\_mask[b,c,t,g] = 1 \,\}
$$

$$
tile\_depths[b,c,t,0:tile\_sum] = depths[b,c,g] \big|_{g \in gauss\_index[b,c,t]}
$$

where $valid\_mask$ is determined by three-level filtering: first pre-filter by the tile's Y-axis range, then coarse-filter by the bounding box, and finally fine-filter by center/tile intersection and ellipse intersection.

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
    double           imageWidth,
    double           imageHeight,
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
      <td>Optional input; empty tensors are supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>opacity</td>
      <td>Input</td>
      <td>Gaussian sphere opacity.</td>
      <td>Optional input; empty tensors are supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>Input</td>
      <td>Conic parameters.</td>
      <td>Optional input; empty tensors are supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>covars2d</td>
      <td>Input</td>
      <td>2D covariance matrix.</td>
      <td>Optional input; empty tensors are supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>Input</td>
      <td>Gaussian sphere depth.</td>
      <td>Optional input; empty tensors are supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>Input</td>
      <td>Valid Gaussian sphere count in each view.</td>
      <td>Optional input; empty tensors are supported; each element must satisfy 0 ≤ cnt ≤ gaussian_num.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num) or (batch_size, camera_num, 1)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileGrid</td>
      <td>Input</td>
      <td>Tile grid information.</td>
      <td>Optional input; empty tensors are supported; numTile = ceil(image_height / tile_size) * ceil(image_width / tile_size).</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(numTile, 2)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gatherMask</td>
      <td>Input</td>
      <td>Inter-core/inter-stage Gather staging buffer; not involved in output computation.</td>
      <td>Optional input; empty tensors are supported; block_num is the platform-adapted workspace row count (typically 48 for the A2 series and 56/64 for the A5 series), with one row per core.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(block_num, 20, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>imageWidth</td>
      <td>Input</td>
      <td>Image width.</td>
      <td>Floating-point number, range [1.0, 4294967295.0].</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>imageHeight</td>
      <td>Input</td>
      <td>Image height.</td>
      <td>Floating-point number, range [1.0, 4294967295.0].</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSize</td>
      <td>Input</td>
      <td>Tile size.</td>
      <td>Positive integer, tile partition granularity; must satisfy numTile = ceil(image_height / tile_size) * ceil(image_width / tile_size) with tileGrid.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSum</td>
      <td>Output</td>
      <td>Total number of valid Gaussian spheres in each tile.</td>
      <td>Required output; empty tensors are supported (empty when any of batch_size/camera_num/gaussian_num is 0).</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, 1)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>tileDepths</td>
      <td>Output</td>
      <td>Ordered list of Gaussian sphere depths in each tile.</td>
      <td>Required output; empty tensors are supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gaussIndex</td>
      <td>Output</td>
      <td>Ordered list of Gaussian sphere indices in each tile.</td>
      <td>Required output; empty tensors are supported.</td>
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
                <td>The data type, format, or shape of the input/output tensor is not within the supported range, or the parameter/attribute values are invalid (for example, a cnt element is outside [0, gaussian_num]).</td>
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

- Data types: means2d, opacity, conics, covars2d, depths, tileGrid, gatherMask, tileDepths, and gaussIndex are FLOAT; cnt and tileSum are INT32. All are in ND format.
- All 8 inputs are optional and support empty tensors: when any of batch_size, camera_num, or gaussian_num is 0, the empty branch is taken, only tileSum is zero-filled, and tileDepths/gaussIndex are 0-element empty lists with no write-back.
- cnt supports rank-2 (batch_size, camera_num) or rank-3 (batch_size, camera_num, 1); each element must satisfy 0 ≤ cnt[b,c] ≤ gaussian_num, otherwise the first-stage interface returns 161002.
- The channel dimensions of means2d, opacity, conics, covars2d, and depths are 2, 1, 3, 3, and 1 respectively, and their batch_size, camera_num, and gaussian_num must be consistent with cnt and gatherMask.
- tileGrid has shape (numTile, 2), where numTile must satisfy numTile = ceil(image_height / tile_size) * ceil(image_width / tile_size), in the range [1, 65536].
- gatherMask has shape (block_num, 20, gaussian_num), where block_num is the platform-adapted workspace row count on the caller side (typically 48 for the A2 series and 56/64 for the A5 series).
- Attribute constraints: image_width and image_height are in the range [1.0, 4294967295.0]; tile_size is a positive integer.
- gaussian_num is in the range [0, 2^24 - 1].
- tileSize is typically 16, used for tile partition computation.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- gaussian_num: Number of Gaussian spheres, indicating the number of 3D Gaussian spheres in each batch.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.
- numTile: Number of tiles, equal to ceil(image_height / tile_size) * ceil(image_width / tile_size).
- block_num: First dimension (row count) of gatherMask (the gather workspace row count), platform-adapted.

## Calling Example

See tests/kernel_tests/test_flash_gaussian_build_mask.cpp
