# aclnnGaussianSort

[View Source Code](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_sort/op_kernel/gaussian_sort.cpp)

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

- Interface function: `gaussian_sort` sorts Gaussian spheres by depth, sorting Gaussian sphere IDs based on tile partitions and depth values, used for alpha blending computation during rendering.
- Computation formula:

Sorts Gaussian spheres based on depth values, outputting the sorted Gaussian sphere ID sequence:

$$
sorted\_gs\_ids = \text{SortByDepth}(gs\_ids, depths)
$$

The sorting order is from near to far (depth values from small to large) to ensure the correct alpha blending order.

## Function Prototype

Each operator is divided into a two-stage interface. You must first call the "aclnnGaussianSortGetWorkspaceSize" interface to obtain the required workspace size and the executor that contains the operator computation process, and then call the "aclnnGaussianSort" interface to execute the computation.

```Cpp
aclnnStatus aclnnGaussianSortGetWorkspaceSize(
    const aclTensor *lbSched,
    const aclTensor *gaussianCnt,
    const aclTensor *gsIds,
    const aclTensor *depths,
    const aclTensor *sortedOffset,
    int64_t          maxTileGauss,
    const aclTensor *sortedGsIds,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGaussianSort(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnGaussianSortGetWorkspaceSize

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
      <td>lbSched</td>
      <td>Input</td>
      <td>Load balancing scheduling information.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size, camera_num, schedule_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gaussianCnt</td>
      <td>Input</td>
      <td>Gaussian sphere count per tile.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num, 1)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gsIds</td>
      <td>Input</td>
      <td>Gaussian sphere ID list.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>Input</td>
      <td>Gaussian sphere depth values.</td>
      <td>Empty tensors are not supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num, gaussian_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>sortedOffset</td>
      <td>Input</td>
      <td>Sorted offsets.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size * camera_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>maxTileGauss</td>
      <td>Input</td>
      <td>Maximum number of Gaussian spheres per tile.</td>
      <td>Positive integer, used to determine the workspace size.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sortedGsIds</td>
      <td>Output</td>
      <td>Sorted Gaussian sphere ID output.</td>
      <td>Empty tensors are not supported.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(totalGauss)</td>
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
                <td>The data type and format of the input/output tensor are not within the supported range, or maxTileGauss is invalid.</td>
            </tr>
        </tbody>
    </table>

## aclnnGaussianSort

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
      <td>Workspace size allocated on the device side, obtained by the first-stage interface aclnnGaussianSortGetWorkspaceSize.</td>
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

- The data types of lbSched and sortedOffset are INT64.
- The data types of gaussianCnt and sortedGsIds are INT32.
- The data types of gsIds and depths are FLOAT.
- maxTileGauss determines the workspace size and needs to be set reasonably based on the actual number of Gaussian spheres.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.
- tile_num: Number of tile partitions, indicating the number of tile blocks the image is divided into.
- gaussian_num: Number of Gaussian spheres, indicating the number of Gaussian spheres in each tile.
- schedule_num: Number of schedules, indicating the number of load balancing schedules.

## Calling Example

See tests/kernel_tests/test_gaussian_sort.cpp
