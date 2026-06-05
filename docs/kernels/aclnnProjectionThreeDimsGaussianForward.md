# aclnnProjectionThreeDimsGaussianForward

[📄 查看源码](https://gitcode.com/cann/gauss-splat/tree/master/kernels/projection_three_dims_gaussian_forward)

## 产品支持情况


| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

## 功能说明

- 接口功能：`projection_three_dims_gaussian_forward`将3D高斯球投影到2D图像平面，计算投影后的位置、深度、锥形参数等。
- 计算公式：

将3D高斯球通过相机矩阵投影到2D平面：

$$
means2d = K \cdot (viewmat \cdot means)
$$

协方差矩阵的投影：

$$
covars2d = J \cdot viewmat \cdot covars \cdot viewmat^T \cdot J^T
$$

其中$J$是雅可比矩阵，$K$是相机内参矩阵，$viewmat$是视图变换矩阵。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnProjectionThreeDimsGaussianForward"接口执行计算。

```Cpp
aclnnStatus aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize(
    const aclTensor *means,
    const aclTensor *covars,
    const aclTensor *viewmats,
    const aclTensor *ks,
    int64_t          imageWidth,
    int64_t          imageHeight,
    float            eps2d,
    bool             calcCompensations,
    int64_t          cameraModel,
    const aclTensor *means2d,
    const aclTensor *depths,
    const aclTensor *conics,
    const aclTensor *compensations,
    const aclTensor *det,
    const aclTensor *radius,
    const aclTensor *covars2d,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnProjectionThreeDimsGaussianForward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize

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
      <td>3D高斯球的位置坐标。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>covars</td>
      <td>输入</td>
      <td>3D高斯球的协方差矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>viewmats</td>
      <td>输入</td>
      <td>视图变换矩阵（相机外参）。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ks</td>
      <td>输入</td>
      <td>相机内参矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>imageWidth</td>
      <td>输入</td>
      <td>图像宽度。</td>
      <td>正整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>imageHeight</td>
      <td>输入</td>
      <td>图像高度。</td>
      <td>正整数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps2d</td>
      <td>输入</td>
      <td>2D协方差的epsilon值，防止数值不稳定。</td>
      <td>推荐值为0.3。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>calcCompensations</td>
      <td>输入</td>
      <td>是否计算补偿值。</td>
      <td>true或false。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cameraModel</td>
      <td>输入</td>
      <td>相机模型类型。</td>
      <td>0表示针孔相机模型。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>means2d</td>
      <td>输出</td>
      <td>投影后的2D位置坐标。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>输出</td>
      <td>投影后的深度值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>输出</td>
      <td>锥形参数，用于渲染计算。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>输出</td>
      <td>补偿值输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>det</td>
      <td>输出</td>
      <td>协方差行列式值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>radius</td>
      <td>输出</td>
      <td>高斯球在屏幕上的半径。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>covars2d</td>
      <td>输出</td>
      <td>投影后的2D协方差矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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

## aclnnProjectionThreeDimsGaussianForward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnProjectionThreeDimsGaussianForwardGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 调用示例

见tests/kernel_tests/test_projection_three_dims_gaussian_forward.cpp
