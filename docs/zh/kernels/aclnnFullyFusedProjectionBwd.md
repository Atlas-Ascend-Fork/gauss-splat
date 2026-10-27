# aclnnFullyFusedProjectionBwd

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fully_fused_projection_bwd/op_kernel/arch22/fully_fused_projection_bwd.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/fully_fused_projection_bwd/op_kernel/arch35/fully_fused_projection_bwd.cpp)

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

- 接口功能：`fully_fused_projection_bwd`计算全融合投影的反向传播，用于训练中计算各参数的梯度。输入为前向投影及筛选（culling）后输出的梯度，输出为高斯球位置、四元数、缩放、颜色、透明度的梯度，以及相机位姿（视图矩阵、内参）和深度修正参数（`vi`、`sigma`）的梯度。
- 计算公式：

位置、四元数、缩放的梯度：

$$
v\_pW = \frac{\partial L}{\partial means}, \quad v\_quats = \frac{\partial L}{\partial quats}
$$

$$
v\_scales = \frac{\partial L}{\partial scales}
$$

颜色、透明度的梯度（透明度梯度会按 `compensations` 加权后在相机维上累加）：

$$
v\_colors = \frac{\partial L}{\partial colors}, \quad v\_opacities = \frac{\partial L}{\partial opacities}
$$

相机位姿与深度修正参数的梯度：

$$
v\_R = \frac{\partial L}{\partial R}, \quad v\_viewmats = \frac{\partial L}{\partial viewmat}, \quad v\_Ks = \frac{\partial L}{\partial K}
$$

$$
dL\_dvi = \frac{\partial L}{\partial vi}, \quad dL\_dsigma = \frac{\partial L}{\partial \sigma}
$$

其中 $L$ 为损失，$R$ 为视图变换矩阵的旋转部分，$\sigma$ 为深度修正尺度（`sigma`），$vi$ 为逐视图深度修正权重。

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
    const aclTensor *vViCulling,
    const aclTensor *opacities,
    const aclTensor *vi,
    const aclTensor *sigma,
    const aclTensor *depths,
    const aclTensor *denseIndices,
    const aclTensor *cnt,
    int64_t          width,
    int64_t          height,
    int64_t          maxCompactCount,
    int64_t          supportFeatures,
    const aclTensor *vPW,
    const aclTensor *vQuats,
    const aclTensor *vScales,
    const aclTensor *vR,
    const aclTensor *vColors,
    const aclTensor *vOpacities,
    const aclTensor *vViewmats,
    const aclTensor *vKs,
    const aclTensor *dLdvi,
    const aclTensor *dLdsigma,
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
      <td>不支持空tensor。开启筛选（supportFeatures bit2）时为按(batch,camera)合并的筛选后紧凑排布。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num) 或 (batch_size*camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quats</td>
      <td>输入</td>
      <td>四元数。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 4, gaussian_num) 或 (batch_size*camera_num, 4, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales</td>
      <td>输入</td>
      <td>缩放因子。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, 3, gaussian_num) 或 (batch_size*camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>输入</td>
      <td>锥形参数（2D协方差矩阵的逆）。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num) 或 (batch_size*camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>viewmats</td>
      <td>输入</td>
      <td>视图变换矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4) 或 (batch_size*camera_num, 16)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>Ks</td>
      <td>输入</td>
      <td>相机内参矩阵。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3) 或 (batch_size*camera_num, 9)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vMeans2d</td>
      <td>输入</td>
      <td>筛选后的2D位置梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num) 或 (batch_size*camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vDepths</td>
      <td>输入</td>
      <td>筛选后的深度梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vConics</td>
      <td>输入</td>
      <td>筛选后的锥形参数梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num) 或 (batch_size*camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vColorsCulling</td>
      <td>输入</td>
      <td>筛选后的颜色梯度。</td>
      <td>不支持空tensor。启用 alpha max 与梯度绝对值输出（supportFeatures bit3）时通道数为6，否则为3。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, color_dims, gaussian_num) 或 (batch_size*camera_num, color_dims, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOpacitiesCulling</td>
      <td>输入</td>
      <td>筛选后的透明度梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>输入</td>
      <td>筛选标记位。</td>
      <td>不支持空tensor。Atlas A2/A3 为按位压缩的UINT8位图，A5（Ascend 950）为FLOAT掩码。</td>
      <td>UINT8（A2/A3）/ FLOAT（A5）</td>
      <td>ND</td>
      <td>(batch_size, camera_num, ceil(gaussian_num/8))（A2/A3）或 (batch_size, camera_num, gaussian_num)（A5）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>compensations</td>
      <td>输入</td>
      <td>透明度补偿值。</td>
      <td>可选输入，可传入nullptr；不为空时用于给透明度梯度加权。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vViCulling</td>
      <td>输入</td>
      <td>筛选后的深度修正权重 `vi` 的梯度。</td>
      <td>可选输入，可传入nullptr；启用深度修正时使用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>opacities</td>
      <td>输入</td>
      <td>筛选后的原始透明度。</td>
      <td>可选输入，可传入nullptr。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vi</td>
      <td>输入</td>
      <td>筛选后的深度修正权重（LCWSR 中每个高斯球的逐视图权重）。</td>
      <td>可选输入，可传入nullptr；启用深度修正时使用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>输入</td>
      <td>深度修正尺度参数（标量）。</td>
      <td>可选输入，可传入nullptr；启用深度修正时使用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>输入</td>
      <td>筛选后的高斯球深度。</td>
      <td>可选输入，可传入nullptr；启用深度修正时使用。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, gaussian_num) 或 (batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>denseIndices</td>
      <td>输入</td>
      <td>筛选后紧凑位置到原始密集高斯球下标的映射索引。</td>
      <td>可选输入，可传入nullptr；开启筛选紧凑排布（supportFeatures bit2）时使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size*camera_num, max_compact_count)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>输入</td>
      <td>每个(batch, camera)视图内筛选后有效的高斯球数量。</td>
      <td>可选输入，可传入nullptr；开启筛选紧凑排布时使用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size*camera_num,)</td>
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
      <td>maxCompactCount</td>
      <td>输入</td>
      <td>每个视图筛选后高斯球数量的最大值，即紧凑缓冲区的列数。</td>
      <td>正整数，默认为1；开启筛选紧凑排布时需与 cnt 的最大值匹配。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>supportFeatures</td>
      <td>输入</td>
      <td>功能开关位掩码。</td>
      <td>默认为0；bit0：启用深度修正；bit1：cbs 后端；bit2：投影反向已筛选（输入为紧凑排布，batch/camera 维合并为 batch_size*camera_num）；bit3：额外输出 alpha max 与梯度绝对值（color_dims 由3变为6）。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vPW</td>
      <td>输出</td>
      <td>高斯球位置的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 3)（A2/A3）或 (batch_size, 3, gaussian_num)（A5）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vQuats</td>
      <td>输出</td>
      <td>四元数的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 4)（A2/A3）或 (batch_size, 4, gaussian_num)（A5）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vScales</td>
      <td>输出</td>
      <td>缩放因子的梯度。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num, 3)（A2/A3）或 (batch_size, 3, gaussian_num)（A5）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vR</td>
      <td>输出</td>
      <td>旋转矩阵的梯度。</td>
      <td>不支持空tensor。（A5 路径为预留输出）</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vColors</td>
      <td>输出</td>
      <td>颜色的梯度。启用 bit3 时额外输出 alpha max 与梯度绝对值。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, color_dims, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vOpacities</td>
      <td>输出</td>
      <td>透明度的梯度（在相机维上累加后）。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vViewmats</td>
      <td>输出</td>
      <td>视图变换矩阵的梯度（用于位姿优化）。</td>
      <td>不支持空tensor。（预留输出）</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 4, 4)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vKs</td>
      <td>输出</td>
      <td>相机内参矩阵的梯度。</td>
      <td>不支持空tensor。（预留输出）</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, 3)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dLdvi</td>
      <td>输出</td>
      <td>深度修正权重 `vi` 的梯度。</td>
      <td>不支持空tensor；启用深度修正时有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size*camera_num, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dLdsigma</td>
      <td>输出</td>
      <td>深度修正尺度 `sigma` 的梯度（标量）。</td>
      <td>不支持空tensor；启用深度修正时有效，仅第0个元素有效。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(1,)</td>
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

- 数据类型：`means`、`quats`、`scales`、`conics`、`viewmats`、`Ks`、`vMeans2d`、`vDepths`、`vConics`、`vColorsCulling`、`vOpacitiesCulling`、`compensations`、`vViCulling`、`opacities`、`vi`、`sigma`、`depths` 及全部输出均为 FLOAT；`denseIndices`、`cnt` 为 INT32；`filter` 在 Atlas A2/A3 上为 UINT8（位图），在 A5（Ascend 950）上为 FLOAT。数据格式均为 ND。
- `compensations`、`vViCulling`、`opacities`、`vi`、`sigma`、`depths`、`denseIndices`、`cnt` 为可选输入，可传入 nullptr。
- `width`、`height` 必须为正整数；`maxCompactCount` 为正整数（默认1）；`supportFeatures` 为位掩码（默认0）。
- `supportFeatures` 位定义：
  - bit0（1）：启用深度修正，使用 `vi`、`sigma`、`depths`、`vViCulling`，并输出 `dLdvi`、`dLdsigma`。
  - bit1（2）：cbs 后端。
  - bit2（4）：投影反向已筛选，输入（`means`、`quats`、`scales`、`conics`、`viewmats`、`Ks` 及各梯度）为筛选后的紧凑排布，(batch, camera) 维合并为 `batch_size*camera_num`，并需提供 `denseIndices`、`cnt`。
  - bit3（8）：额外输出 alpha max 与梯度绝对值，`vColorsCulling` 与 `vColors` 的通道数（color_dims）由3变为6。
- 当 bit2 置位时，`denseIndices` 与 `cnt` 用于将紧凑位置映射回原始密集高斯球下标；`cnt` 元素的取值范围为 `[0, gaussian_num]`，`maxCompactCount` 应不小于 `cnt` 的最大值。
- 各产品路径实际写出（有效）的输出不同：
  - Atlas A2/A3：输出 `vPW`、`vQuats`、`vScales`、`vR`、`vColors`、`vOpacities`；`vViewmats`、`vKs`、`dLdvi`、`dLdsigma` 为预留输出（不写出）。
  - A5（Ascend 950）：输出 `vPW`、`vQuats`、`vScales`、`vColors`、`vOpacities`、`dLdvi`、`dLdsigma`；`vR`、`vViewmats`、`vKs` 为预留输出（不写出）。
- 所有输出均为必选输出，第一段接口需全部传入有效tensor。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- camera_num：相机数量，表示参与计算的相机个数。
- color_dims：颜色通道数，默认为3，启用 `supportFeatures` bit3 时为6。

## 调用示例

见tests/kernel_tests/test_fully_fused_projection_bwd.cpp
