# aclnnFlashGaussianBuildMask

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/flash_gaussian_build_mask/op_kernel/flash_gaussian_build_mask.cpp)

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

- 接口功能：`flash_gaussian_build_mask`构建Flash渲染所需的掩码信息，用于优化渲染效率。
- 计算公式：

根据高斯球位置、透明度等参数构建tile掩码：

$$
tile\_sum = \sum_{gaussians\_in\_tile} valid\_mask
$$

$$
tile\_depths = \sum_{gaussians\_in\_tile} depth \cdot valid\_mask
$$

掩码构建过程优化了后续渲染的并行计算效率。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnFlashGaussianBuildMaskGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnFlashGaussianBuildMask"接口执行计算。

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
      <td>means2d</td>
      <td>输入</td>
      <td>投影后的2D位置坐标。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>opacity</td>
      <td>输入</td>
      <td>高斯球透明度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>输入</td>
      <td>锥形参数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>covars2d</td>
      <td>输入</td>
      <td>2D协方差矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>输入</td>
      <td>高斯球深度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>输入</td>
      <td>每个tile的高斯球计数。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileGrid</td>
      <td>输入</td>
      <td>Tile网格信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(numTile, 2)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gatherMask</td>
      <td>输入</td>
      <td>Gather掩码信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(48, 20, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>imageWidth</td>
      <td>输入</td>
      <td>图像宽度。</td>
      <td>浮点数。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>imageHeight</td>
      <td>输入</td>
      <td>图像高度。</td>
      <td>浮点数。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSize</td>
      <td>输入</td>
      <td>Tile大小。</td>
      <td>正整数，通常为16。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSum</td>
      <td>输出</td>
      <td>每个tile的高斯球数量总和。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileDepths</td>
      <td>输出</td>
      <td>每个tile的深度信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gaussIndex</td>
      <td>输出</td>
      <td>高斯球索引信息。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, gaussian_num)</td>
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
                <td>输入输出tensor的数据类型和数据格式不在支持的范围内，或参数值不合法。</td>
            </tr>
        </tbody>
    </table>

## aclnnFlashGaussianBuildMask

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFlashGaussianBuildMaskGetWorkspaceSize获取。</td>
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

- means2d、opacity、conics、covars2d、depths、tileGrid、gatherMask、tileDepths、gaussIndex的数据类型为FLOAT。
- cnt、tileSum的数据类型为INT32。
- tileSize通常为16，用于tile分区计算。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- camera_num：相机数量，表示参与计算的相机个数。
- numTile：tile数量，表示图像划分的tile区块总数。

## 调用示例

见tests/kernel_tests/test_flash_gaussian_build_mask.cpp
