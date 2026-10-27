# aclnnGaussianFilter

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_filter/op_kernel/arch22/gaussian_filter.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/gaussian_filter/op_kernel/arch35/gaussian_filter.cpp)

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

- 接口功能：`gaussian_filter` 根据视锥体参数对每个 (batch, camera) 视图下的高斯球进行筛选，剔除不在可见范围内的高斯球，并将筛选结果沿高斯球维度压缩为紧凑排布（compact）。同时输出筛选标记（按位压缩位图与浮点掩码）、每个视图内有效高斯球数量，以及紧凑位置到原始密集下标的映射；并支持配合 `vi`、`sigma` 的深度修正。
- 计算公式：

对每个 (batch, camera) 视图下第 $i$ 个高斯球，记：

$$
d_i = depths[b,c,i], \quad m_i = means2d[b,c,:,i], \quad r_i = radius[b,c,:,i]
$$

筛选标记为：

$$
filter[b,c,i] = \begin{cases}
1 & \text{if } i < gaussian\_num \ \wedge\ det[b,c,i] > 0 \ \wedge\ near\_plane < d_i < far\_plane \\
  & \wedge\ vi[b,c,i] > 0 \\
  & \wedge\ (m_{x,i} + r_{x,i} > 0) \wedge (m_{x,i} - r_{x,i} < width) \\
  & \wedge\ (m_{y,i} + r_{y,i} > 0) \wedge (m_{y,i} - r_{y,i} < height) \\
0 & \text{otherwise}
\end{cases}
$$

其中 `vi > 0` 条件在 A5（Ascend 950）路径生效。

每个视图内有效高斯球数量：

$$
cnt[b,c] = \sum_{i} filter[b,c,i]
$$

紧凑排布与下标映射（$\{\, i \mid filter[b,c,i]=1 \,\}$ 按原始顺序排列）：

$$
culling[b,c,:,0:cnt] = args[b,c,:,i] \big|_{filter[b,c,i]=1}, \quad denseIndices[b,c,k] = \text{第 } k \text{ 个有效高斯球的原始下标}
$$

其中 `args` 为各参数（`means`、`colors`、`quats`、`scales`、`means2d`、`depths`、`radius`、`covars2d`、`conics`、`opacities`、`vi`、`compensations`），分别写入对应的 `*Culling` 输出；`filter` 为按位压缩的 UINT8 位图，`filterFloat` 为 0/1 浮点掩码。当开启 `supportFeatures` bit0（深度修正）时，`viCulling = vi × max(1 − depths / sigma, 0)`，否则 `viCulling = vi`。

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
    const aclTensor *quats,
    const aclTensor *scales,
    const aclTensor *vi,
    const aclTensor *sigma,
    int64_t          width,
    int64_t          height,
    float            nearPlane,
    float            farPlane,
    int64_t          supportFeatures,
    const aclTensor *meansCulling,
    const aclTensor *colorsCulling,
    const aclTensor *means2dCulling,
    const aclTensor *depthsCulling,
    const aclTensor *radiusCulling,
    const aclTensor *covars2dCulling,
    const aclTensor *conicsCulling,
    const aclTensor *opacitiesCulling,
    const aclTensor *quatsCulling,
    const aclTensor *scalesCulling,
    const aclTensor *viCulling,
    const aclTensor *compensationsCulling,
    const aclTensor *filter,
    const aclTensor *cnt,
    const aclTensor *filterFloat,
    const aclTensor *denseIndices,
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
      <td>不支持空tensor。通道数 color_dims 通常为3，启用扩展特征时可为7。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, color_dims, gaussian_num)</td>
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
      <td>透明度补偿值。</td>
      <td>可选输入，可传入nullptr；不为空时 opacitiesCulling 按透明度×补偿加权，并输出 compensationsCulling。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quats</td>
      <td>输入</td>
      <td>四元数。</td>
      <td>可选输入，可传入nullptr；不为空时参与筛选并输出 quatsCulling。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>缩放因子。</td>
      <td>可选输入，可传入nullptr；不为空时参与筛选并输出 scalesCulling。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vi</td>
      <td>输入</td>
      <td>逐视图深度修正权重（LCWSR 中每个高斯球的逐视图权重）。</td>
      <td>可选输入，可传入nullptr；不为空时筛选条件要求 vi &gt; 0，并输出 viCulling。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>输入</td>
      <td>深度修正尺度参数（标量）。</td>
      <td>可选输入，可传入nullptr；启用深度修正（supportFeatures bit0）时使用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1,)</td>
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
      <td>supportFeatures</td>
      <td>输入</td>
      <td>功能开关位掩码。</td>
      <td>默认为0；bit0：启用深度修正（viCulling 按 vi×(1−depths/sigma) 加权），其余位预留。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meansCulling</td>
      <td>输出</td>
      <td>筛选后的3D位置（沿高斯球维压缩为紧凑排布）。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>colorsCulling</td>
      <td>输出</td>
      <td>筛选后的颜色。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, color_dims, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>means2dCulling</td>
      <td>输出</td>
      <td>筛选后的2D位置。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depthsCulling</td>
      <td>输出</td>
      <td>筛选后的深度。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>radiusCulling</td>
      <td>输出</td>
      <td>筛选后的半径。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>covars2dCulling</td>
      <td>输出</td>
      <td>筛选后的2D协方差。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>conicsCulling</td>
      <td>输出</td>
      <td>筛选后的锥形参数。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>opacitiesCulling</td>
      <td>输出</td>
      <td>筛选后的透明度。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quatsCulling</td>
      <td>输出</td>
      <td>筛选后的四元数。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalesCulling</td>
      <td>输出</td>
      <td>筛选后的缩放因子。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>viCulling</td>
      <td>输出</td>
      <td>筛选后的深度修正权重。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。启用深度修正时按 vi×(1−depths/sigma) 加权。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>compensationsCulling</td>
      <td>输出</td>
      <td>筛选后的透明度补偿值。</td>
      <td>不支持空tensor；仅前 cnt 个位置有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输出</td>
      <td>筛选标记位（按位压缩的 UINT8 位图）。</td>
      <td>不支持空tensor。第 i 位为1表示第 i 个高斯球通过筛选。</td>
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
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>filterFloat</td>
      <td>输出</td>
      <td>筛选标记的浮点掩码（0/1）。</td>
      <td>不支持空tensor。第 i 个元素为1表示第 i 个高斯球通过筛选。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>denseIndices</td>
      <td>输出</td>
      <td>紧凑位置到原始密集高斯球下标的映射索引。</td>
      <td>不支持空tensor。第 k 个元素为紧凑排布第 k 个有效高斯球的原始下标。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num)</td>
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
                <td>输入输出tensor的数据类型、数据格式或shape不在支持的范围内，或参数值/属性值不合法。</td>
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

- 数据类型：`means`、`colors`、`det`、`opacities`、`means2d`、`depths`、`radius`、`conics`、`covars2d`、`compensations`、`quats`、`scales`、`vi`、`sigma` 及全部 `*Culling` 输出、`filterFloat` 均为 FLOAT；`filter` 为 UINT8；`cnt` 为 FLOAT；`denseIndices` 为 INT32。数据格式均为 ND。
- `compensations`、`quats`、`scales`、`vi`、`sigma` 为可选输入，可传入 nullptr。
- `width`、`height` 必须为正整数；`nearPlane`、`farPlane` 为浮点数；`supportFeatures` 为位掩码（默认0）。
- `supportFeatures` 位定义：
  - bit0（1）：启用深度修正，`viCulling = vi × max(1 − depths / sigma, 0)`；关闭时 `viCulling = vi`。
  - 其余位预留。
- 筛选结果为沿高斯球维压缩后的紧凑排布：各 `*Culling` 输出与 `denseIndices` 的最后一维仍为 `gaussian_num`，但每个 (batch, camera) 视图仅前 `cnt` 个位置有效。
- 各产品路径实际写出（有效）的输出不同：
  - Atlas A2/A3：输出 `meansCulling`、`colorsCulling`、`means2dCulling`、`depthsCulling`、`radiusCulling`、`covars2dCulling`、`conicsCulling`、`opacitiesCulling`、`filter`、`cnt`；`quatsCulling`、`scalesCulling`、`viCulling`、`compensationsCulling`、`filterFloat`、`denseIndices` 为预留输出。
  - A5（Ascend 950）：输出全部16个输出；`filter` 为按位压缩位图，`filterFloat` 为对应的浮点掩码。
- 所有输出均为必选输出，第一段接口需全部传入有效tensor。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- camera_num：相机数量，表示参与计算的相机个数。
- color_dims：颜色通道数，通常为3，启用扩展特征时为7。

## 调用示例

见tests/kernel_tests/test_gaussian_filter.cpp
