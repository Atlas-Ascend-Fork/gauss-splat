# aclnnGaussianSort

[View Source Code (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_sort/op_kernel/arch22/gaussian_sort.cpp) | [View Source Code (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_sort/op_kernel/arch35/gaussian_sort.cpp)

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

- Interface function: `gaussian_sort` uses the load-balancing scheduling information (lb_sched) to assign the tiles to be sorted for each (batch, camera) view, sorts the first gaussianCnt Gaussian spheres in each tile in ascending depth order, and finally concatenates the sorting results of all valid tiles in tile-index order to output the sorted Gaussian sphere ID sequence, used for alpha blending computation during rendering.
- Computation formula:

For each (batch, camera, tile), let the depth value be $d_i = depths[b,c,t,i]$ and the Gaussian sphere ID be $g_i = gsIds[b,c,t,i]$, where the valid elements participating in the sort are the first $n = gaussianCnt[b,c,t]$ elements:

$$
sorted\_gs\_ids = \text{SortByDepth}(gsIds, depths)
$$

That is, the Gaussian sphere IDs are sorted in ascending order of $d_i$ (from near to far) to ensure the correct alpha blending order. The sort is performed independently within each tile; the sorting results of all valid tiles are written compactly in ascending tile-index order within each (batch, camera) view, and different views are arranged in batch and camera order.

The sort is preferentially completed in a single pass in UB; when the number of Gaussian spheres in a single tile exceeds the UB single-sort capacity, the operator automatically switches to a workspace-based segmented merge sort. The A5 (Ascend 950) path only compares adjacent subsequences during merging, which is a stable sort: Gaussian spheres with equal depth values keep their original relative order.

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
      <td>Load balancing scheduling information. Within each (batch, camera) view, it stores num_bins core offsets, tile_num scheduled tile indices, and tile_num tile offsets in order.</td>
      <td>Empty tensors are not supported. The third dimension schedule_num = num_bins + 2 * tile_num.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size, camera_num, schedule_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gaussianCnt</td>
      <td>Input</td>
      <td>Valid Gaussian sphere count per tile; only the first gaussianCnt Gaussian spheres participate in the sort.</td>
      <td>Empty tensors are not supported. Element values must satisfy 0 ≤ gaussianCnt ≤ gaussian_num.</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num) or (batch_size, camera_num, tile_num, 1)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>gsIds</td>
      <td>Input</td>
      <td>Gaussian sphere ID list.</td>
      <td>Empty tensors are not supported. Gaussian sphere IDs are stored as FLOAT and truncated to INT32 inside the kernel; values should be integer IDs exactly representable in FLOAT.</td>
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
      <td>Exclusive prefix sum of the valid Gaussian sphere counts of all (batch, camera) views, used to determine the write offset of each view in the output.</td>
      <td>Empty tensors are not supported. The k-th element is the sum of the Gaussian sphere counts of the first k views; the total output length is the last element of sortedOffset.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size * camera_num)</td>
      <td>Yes</td>
    </tr>
    <tr>
      <td>maxTileGauss</td>
      <td>Input</td>
      <td>Maximum number of Gaussian spheres per tile.</td>
      <td>Positive integer; must be no less than the maximum Gaussian sphere count of a single tile (max(gaussianCnt)). It determines the workspace size.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sortedGsIds</td>
      <td>Output</td>
      <td>Sorted Gaussian sphere ID output; the valid Gaussian sphere IDs of each (batch, camera) view are arranged compactly in ascending tile-index order.</td>
      <td>Empty tensors are not supported. The shape is (totalGauss), where totalGauss = the sum of the valid Gaussian sphere counts of all (batch, camera) views.</td>
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
                <td>The data type, format, or shape of the input/output tensor is not within the supported range, or maxTileGauss is invalid.</td>
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

- Data types: lbSched and sortedOffset are INT64; gaussianCnt and sortedGsIds are INT32; gsIds and depths are FLOAT. The data format is ND for all.
- gsIds stores Gaussian sphere IDs as FLOAT and is truncated to INT32 inside the kernel; values should be integer IDs exactly representable in FLOAT (usually non-negative integers).
- Only the first gaussianCnt Gaussian spheres of each tile participate in the sort; the remaining elements are ignored. Tiles with gaussianCnt equal to 0 are not sorted and are not written.
- Sorting semantics: sorted in ascending depth order (from near to far), so that the correct alpha blending order is guaranteed.
- A5 (Ascend 950): constrained by the Sort instruction repeat limit, a single UB sort supports at most 255 × 32 = 8160 Gaussian spheres (the actual capacity is also limited by the UB size); when the number of Gaussian spheres in a single tile exceeds this capacity, a workspace-based segmented merge sort is used. On this path the merge only compares adjacent subsequences, which is a stable sort, so Gaussian spheres with equal depth values keep their original relative order.
- maxTileGauss must be no less than the maximum Gaussian sphere count of a single tile (max(gaussianCnt)), otherwise the workspace is insufficient; the workspace is allocated as 32-aligned maxTileGauss × sizeof(FLOAT) × 2 × blockDim.
- blockDim = num_bins, derived from the lbSched length; 1 <= num_bins <= the number of AIV cores.
- Within each (batch, camera) view of lbSched, the first num_bins elements are core offsets, the following tile_num elements are scheduled tile indices, and the last tile_num elements are tile offsets.
- Empty tensors are not supported for any input or output.

**Dimension Variable Description:**

- batch_size: Batch size, indicating the number of data batches processed at once.
- camera_num: Number of cameras, indicating the number of cameras participating in the computation.
- tile_num: Number of tile partitions, indicating the number of tile blocks the image is divided into.
- gaussian_num: Number of Gaussian spheres, indicating the number of Gaussian spheres in each tile.
- schedule_num: Length of the scheduling information, schedule_num = num_bins + 2 * tile_num.
- num_bins: Number of load-balancing cores, equal to schedule_num - 2 * tile_num. When followed by CalcRenderSimtFwd, use the full AIV core count even if tile_num is smaller.
- totalGauss: The sum of the valid Gaussian sphere counts of all (batch, camera) views, equal to the last element of sorted_offset.

## Calling Example

See tests/kernel_tests/test_gaussian_sort.cpp
