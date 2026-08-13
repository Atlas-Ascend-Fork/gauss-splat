# aclnnGaussianFilter

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_filter/op_kernel/gaussian_filter.cpp)

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

- 接口功能：`gaussian_filter`根据视锥体参数对高斯球进行筛选，剔除不在可见范围内的高斯球。
- 计算公式：

根据near_plane、far_plane、图像边界等参数筛选高斯球：

$$
filter[i] = \begin{cases} 1 & \text{if } depth_i \in [near\_plane, far\_plane] \text{ and } means2d_i \in image \\ 0 & \text{otherwise} \end{cases}
$$

筛选后的输出保持原始数据顺序，但仅保留可见的高斯球。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnGaussianFilterGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGaussianFilter"接口执行计算。

```Cpp
aclnnStatus aclnnGaussianFilterGetWorkspaceSize(
    const aclTensor *means,
    const aclTensor *colors,
    const aclTensor *det,
    const aclTensor *opacities,
    const aclTensor *means2d,
    const aclTensor *depths,
    const aclTensor *radius,
    const aclTensor *conics,
    const aclTensor *covars2d,
    const aclTensor *compensations,
    int64_t          width,
    int64_t          height,
    float            nearPlane,
    float            farPlane,
    const aclTensor *meansCulling,
    const aclTensor *colorsCulling,
    const aclTensor *means2dCulling,
    const aclTensor *depthsCulling,
    const aclTensor *radiusCulling,
    const aclTensor *covars2dCulling,
    const aclTensor *conicsCulling,
    const aclTensor *opacitiesCulling,
    const aclTensor *filter,
    const aclTensor *cnt,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGaussianFilter(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnGaussianFilterGetWorkspaceSize

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
      <td>means</td>
      <td>输入</td>
      <td>3D高斯球位置。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>colors</td>
      <td>输入</td>
      <td>高斯球颜色。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>det</td>
      <td>输入</td>
      <td>协方差行列式值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>opacities</td>
      <td>输入</td>
      <td>高斯球透明度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>means2d</td>
      <td>输入</td>
      <td>投影后2D位置。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>输入</td>
      <td>高斯球深度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>输入</td>
      <td>高斯球屏幕半径。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
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
      <td>compensations</td>
      <td>输入</td>
      <td>补偿值（可选）。</td>
      <td>可选输入，可传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>width</td>
      <td>输入</td>
      <td>图像宽度。</td>
      <td>正整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>height</td>
      <td>输入</td>
      <td>图像高度。</td>
      <td>正整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nearPlane</td>
      <td>输入</td>
      <td>近平面距离。</td>
      <td>浮点数，通常为0.01。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>farPlane</td>
      <td>输入</td>
      <td>远平面距离。</td>
      <td>浮点数，通常为100.0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meansCulling</td>
      <td>输出</td>
      <td>筛选后的3D位置。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>colorsCulling</td>
      <td>输出</td>
      <td>筛选后的颜色。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>means2dCulling</td>
      <td>输出</td>
      <td>筛选后的2D位置。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depthsCulling</td>
      <td>输出</td>
      <td>筛选后的深度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>radiusCulling</td>
      <td>输出</td>
      <td>筛选后的半径。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>covars2dCulling</td>
      <td>输出</td>
      <td>筛选后的2D协方差。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>conicsCulling</td>
      <td>输出</td>
      <td>筛选后的锥形参数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>opacitiesCulling</td>
      <td>输出</td>
      <td>筛选后的透明度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输出</td>
      <td>筛选标记位。</td>
      <td>不支持空tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ceil(gaussian_num/8))</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>输出</td>
      <td>筛选后的高斯球数量。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num)</td>
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

## aclnnGaussianFilter

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGaussianFilterGetWorkspaceSize获取。</td>
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

- 大部分输入输出tensor的数据类型为FLOAT。
- filter的数据类型为UINT8，cnt的数据类型为INT32。
- compensations为可选输入，可传入nullptr。
- width和height必须为正整数，nearPlane和farPlane为浮点数。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- camera_num：相机数量，表示参与计算的相机个数。

## 调用示例

见tests/kernel_tests/test_gaussian_filter.cpp
