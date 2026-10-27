# aclnnGaussianSort

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_sort/op_kernel/arch22/gaussian_sort.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_sort/op_kernel/arch35/gaussian_sort.cpp)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
| Atlas 200I/500 A2 推理产品                  |    ×    |
| Atlas 推理系列产品                          |    ×    |
| Atlas 训练系列产品                          |    ×    |

## 功能说明

- 接口功能：`gaussian_sort` 依据负载均衡调度信息（lb_sched）为每个 (batch, camera) 视图分配待排序的 tile，并对各 tile 内前 gaussianCnt 个高斯球按深度值升序排序，最终将所有有效 tile 的排序结果按 tile 索引紧凑拼接，输出排序后的高斯球ID序列，用于渲染时的 alpha blending 计算。
- 计算公式：

对每个 (batch, camera, tile)，记深度值 $d_i = depths[b,c,t,i]$、高斯球ID $g_i = gsIds[b,c,t,i]$，参与排序的有效元素为前 $n = gaussianCnt[b,c,t]$ 个：

$$
sorted\_gs\_ids = \text{SortByDepth}(gsIds, depths)
$$

即按 $d_i$ 升序（从近到远）对高斯球ID排序，以保证正确的 alpha blending 顺序。排序在单个 tile 内独立进行，各有效 tile 的排序结果在 (batch, camera) 视图内按 tile 索引升序紧凑写入，不同视图之间按 batch、camera 顺序依次排布。

排序优先在 UB 中一次性完成；当单个 tile 的高斯球数量超过 UB 单次排序容量时，自动切换到基于 workspace 的分段归并排序。A5（Ascend 950）路径的归并仅比较相邻子序列，为稳定排序，深度值相同时保持原始相对顺序。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnGaussianSortGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGaussianSort"接口执行计算。

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

- **参数说明：**
  
  <table style="width: 100%">
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>lbSched</td>
      <td>输入</td>
      <td>负载均衡调度信息。每个 (batch, camera) 视图内依次存放 num_bins 个核心偏移量、tile_num 个调度tile索引和 tile_num 个tile偏移量。</td>
      <td>不支持空tensor。第三维 schedule_num = num_bins + 2 * tile_num。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size, camera_num, schedule_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gaussianCnt</td>
      <td>输入</td>
      <td>每个tile中的有效高斯球计数，仅前 gaussianCnt 个高斯球参与排序。</td>
      <td>不支持空tensor。元素取值需满足 0 ≤ gaussianCnt ≤ gaussian_num。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num) 或 (batch_size, camera_num, tile_num, 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gsIds</td>
      <td>输入</td>
      <td>高斯球ID列表。</td>
      <td>不支持空tensor。以 FLOAT 存储的高斯球ID，内核按截断方式转换为 INT32 使用，取值应为可用 FLOAT 精确表示的整数ID。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>输入</td>
      <td>高斯球深度值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sortedOffset</td>
      <td>输入</td>
      <td>各 (batch, camera) 视图内有效高斯球总数的累积前缀和（exclusive），用于确定各视图在输出中的写入偏移。</td>
      <td>不支持空tensor。第 k 个元素为前 k 个视图的高斯球数之和，输出总长度为 sortedOffset 的最后一个元素。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size * camera_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>maxTileGauss</td>
      <td>输入</td>
      <td>每个tile中最大高斯球数量。</td>
      <td>正整数，需不小于单个tile内的最大高斯球数（max(gaussianCnt)），用于确定workspace大小。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sortedGsIds</td>
      <td>输出</td>
      <td>排序后的高斯球ID输出，各 (batch, camera) 视图的有效高斯球ID按 tile 索引升序紧凑排布。</td>
      <td>不支持空tensor。shape 为 (totalGauss)，totalGauss = 所有 (batch, camera) 视图内有效高斯球数之和。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(totalGauss)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/aclnn_return_code.md)。
  
  第一段接口会完成入参校验，出现以下场景时报错：
  
  <table style="width: 100%">
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>输入输出tensor的数据类型、数据格式或shape不在支持的范围内，或maxTileGauss不合法。</td>
            </tr>
        </tbody>
    </table>

## aclnnGaussianSort

- **参数说明：**
  
  <table style="width: 100%">
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGaussianSortGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>
- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](https://gitcode.com/cann/ops-nn/blob/master/docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 数据类型：lbSched、sortedOffset 为 INT64；gaussianCnt、sortedGsIds 为 INT32；gsIds、depths 为 FLOAT。数据格式均为 ND。
- gsIds 以 FLOAT 存储高斯球ID，内核按截断方式转换为 INT32 使用，取值应为可用 FLOAT 精确表示的整数ID（通常为非负整数）。
- 每个tile仅前 gaussianCnt 个高斯球参与排序，其余元素被忽略；gaussianCnt 为 0 的tile不参与排序、不写出。
- 排序语义：按深度值升序（从近到远）排列，输出用于保证正确的 alpha blending 顺序。
- A5（Ascend 950）：受 Sort 指令 repeat 上限约束，单次 UB 排序最多支持 255 × 32 = 8160 个高斯球（实际容量同时受 UB 大小限制）；当单个tile的高斯球数量超过该容量时，使用基于 workspace 的分段归并排序。该路径的归并仅比较相邻子序列，为稳定排序，深度值相同时保持高斯球原始相对顺序。
- maxTileGauss 必须不小于单个tile内的最大高斯球数（max(gaussianCnt)），否则 workspace 不足；workspace 按 32 对齐后的 maxTileGauss × sizeof(FLOAT) × 2 × blockDim 申请。
- blockDim = num_bins，由 lbSched 的长度推得；须满足 1 <= num_bins <= 当前产品的 AIV 核数。
- lbSched 的每个 (batch, camera) 视图内，前 num_bins 个元素为核心偏移量，随后 tile_num 个元素为调度tile索引，最后 tile_num 个元素为tile偏移量。
- 所有输入输出均不支持空tensor。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- camera_num：相机数量，表示参与计算的相机个数。
- tile_num：tile分区数量，表示图像划分的tile区块个数。
- gaussian_num：高斯球数量，表示每个tile中的高斯球个数。
- schedule_num：调度信息长度，schedule_num = num_bins + 2 * tile_num。
- num_bins：负载均衡核数，等于 schedule_num - 2 * tile_num。与 CalcRenderSimtFwd 串联时必须使用当前产品的 AIV 核数，即使 tile_num 更小。
- totalGauss：所有 (batch, camera) 视图内有效高斯球数之和，等于 sorted_offset 的最后一个元素。

## 调用示例

见tests/kernel_tests/test_gaussian_sort.cpp
