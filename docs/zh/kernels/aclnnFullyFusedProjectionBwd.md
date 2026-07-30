# aclnnFullyFusedProjectionBwd

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fully_fused_projection_bwd/op_kernel/fully_fused_projection_bwd.cpp)

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

- 接口功能：`fully_fused_projection_bwd`计算全融合投影的反向传播，用于训练中计算各参数的梯度。
- 计算公式：

反向传播计算位置、四元数、缩放、颜色、透明度等参数的梯度：

$$
v\_means = \frac{\partial L}{\partial means}, \quad v\_quats = \frac{\partial L}{\partial quats}
$$

$$
v\_scales = \frac{\partial L}{\partial scales}, \quad v\_colors = \frac{\partial L}{\partial colors}
$$

$$
v\_opacities = \frac{\partial L}{\partial opacities}
$$

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnFullyFusedProjectionBwdGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnFullyFusedProjectionBwd"接口执行计算。

```Cpp
aclnnStatus aclnnFullyFusedProjectionBwdGetWorkspaceSize(
    const aclTensor *means,
    const aclTensor *quats,
    const aclTensor *scales,
    const aclTensor *conics,
    const aclTensor *viewmats,
    const aclTensor *Ks,
    const aclTensor *vMeans2d,
    const aclTensor *vDepths,
    const aclTensor *vConics,
    const aclTensor *vColorsCulling,
    const aclTensor *vOpacitiesCulling,
    const aclTensor *filter,
    const aclTensor *compensations,
    int64_t          width,
    int64_t          height,
    const aclTensor *vPW,
    const aclTensor *vQuats,
    const aclTensor *vScales,
    const aclTensor *vR,
    const aclTensor *vColors,
    const aclTensor *vOpacities,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFullyFusedProjectionBwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnFullyFusedProjectionBwdGetWorkspaceSize

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
      <td>quats</td>
      <td>输入</td>
      <td>四元数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>缩放因子。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
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
      <td>viewmats</td>
      <td>输入</td>
      <td>视图变换矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(camera_num, 4, 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>Ks</td>
      <td>输入</td>
      <td>相机内参矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(camera_num, 3, 3)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vMeans2d</td>
      <td>输入</td>
      <td>2D位置的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vDepths</td>
      <td>输入</td>
      <td>深度的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vConics</td>
      <td>输入</td>
      <td>锥形参数的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vColorsCulling</td>
      <td>输入</td>
      <td>筛选后颜色的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOpacitiesCulling</td>
      <td>输入</td>
      <td>筛选后透明度的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输入</td>
      <td>筛选标记位。</td>
      <td>不支持空tensor。</td>
      <td>UINT8</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>输入</td>
      <td>补偿值（可选）。</td>
      <td>可选输入，可传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
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
      <td>vPW</td>
      <td>输出</td>
      <td>投影矩阵的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vQuats</td>
      <td>输出</td>
      <td>四元数的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vScales</td>
      <td>输出</td>
      <td>缩放因子的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vR</td>
      <td>输出</td>
      <td>旋转矩阵的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vColors</td>
      <td>输出</td>
      <td>颜色的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOpacities</td>
      <td>输出</td>
      <td>透明度的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
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

## aclnnFullyFusedProjectionBwd

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFullyFusedProjectionBwdGetWorkspaceSize获取。</td>
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
- filter的数据类型为UINT8。
- compensations为可选输入，可传入nullptr。
- width和height必须为正整数。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- camera_num：相机数量，表示参与计算的相机个数。

## 调用示例

见tests/kernel_tests/test_fully_fused_projection_bwd.cpp
