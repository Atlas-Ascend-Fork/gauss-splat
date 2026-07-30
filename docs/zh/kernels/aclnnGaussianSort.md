# aclnnGaussianSort

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_sort/op_kernel/gaussian_sort.cpp)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    ×    |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √    |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √    |
| Atlas 200I/500 A2 推理产品                  |    ×    |
| Atlas 推理系列产品                          |    ×    |
| Atlas 训练系列产品                          |    ×    |

## 功能说明

- 接口功能：`gaussian_sort`对高斯球进行深度排序，根据tile分区和深度值对高斯球ID进行排序，用于渲染时的alpha blending计算。
- 计算公式：

根据深度值对高斯球进行排序，输出排序后的高斯球ID序列：

$$
sorted\_gs\_ids = \text{SortByDepth}(gs\_ids, depths)
$$

排序顺序为从近到远（深度值从小到大），以保证正确的alpha blending顺序。

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
      <td>负载均衡调度信息。</td>
      <td>不支持空tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size, camera_num, schedule_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gaussianCnt</td>
      <td>输入</td>
      <td>每个tile中的高斯球计数。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, tile_num, 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gsIds</td>
      <td>输入</td>
      <td>高斯球ID列表。</td>
      <td>不支持空tensor。</td>
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
      <td>排序后的偏移量。</td>
      <td>不支持空tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ...)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>maxTileGauss</td>
      <td>输入</td>
      <td>每个tile中最大高斯球数量。</td>
      <td>正整数，用于确定workspace大小。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sortedGsIds</td>
      <td>输出</td>
      <td>排序后的高斯球ID输出。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ...)</td>
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
                <td>输入输出tensor的数据类型和数据格式不在支持的范围内，或maxTileGauss不合法。</td>
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

- lbSched和sortedOffset的数据类型为INT64。
- gaussianCnt和sortedGsIds的数据类型为INT32。
- gsIds和depths的数据类型为FLOAT。
- maxTileGauss决定了workspace的大小，需要根据实际高斯球数量合理设置。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- camera_num：相机数量，表示参与计算的相机个数。
- tile_num：tile分区数量，表示图像划分的tile区块个数。
- gaussian_num：高斯球数量，表示每个tile中的高斯球个数。
- schedule_num：调度数量，表示负载均衡调度的调度次数。

## 调用示例

见tests/kernel_tests/test_gaussian_sort.cpp
