# aclnnSphericalHarmonicsForward

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_forward/op_kernel/arch22/spherical_harmonics_forward.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_forward/op_kernel/arch35/spherical_harmonics_forward.cpp)

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

- 接口功能：`spherical_harmonics_forward`计算球谐函数的前向传播，用于3D Gaussian Splatting中的颜色渲染。
- 计算公式：

球谐函数将方向向量映射到颜色空间，公式如下：

$$
output = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_{lm}(dirs)
$$

其中$Y_{lm}$是$l$阶$m$次的球谐基函数，$c_{lm}$是球谐系数，$dirs$是方向向量。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSphericalHarmonicsForwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSphericalHarmonicsForward"接口执行计算。

```Cpp
aclnnStatus aclnnSphericalHarmonicsForwardGetWorkspaceSize(
    const aclTensor *dirs,
    const aclTensor *coeffs,
    int64_t          degreesToUse,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSphericalHarmonicsForward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnSphericalHarmonicsForwardGetWorkspaceSize

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
      <td>dirs</td>
      <td>输入</td>
      <td>方向向量，表示光线方向。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>coeffs</td>
      <td>输入</td>
      <td>球谐系数，用于颜色计算。</td>
      <td>不支持空tensor。系数数量K与degrees_to_use相关，通道数C支持1~4，通常为3。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, K, C, gaussian_num)，其中K=(degrees_to_use+1)*(degrees_to_use+1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>degreesToUse</td>
      <td>输入</td>
      <td>使用的球谐阶数。</td>
      <td>支持0~4，推荐使用2、3、4。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>计算得到的颜色输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, C, gaussian_num)</td>
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
                <td>dirs、coeffs、output的数据类型和数据格式不在支持的范围内，或degrees_to_use不在0~4范围内，或coeffs的通道维C不在1~4范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnSphericalHarmonicsForward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSphericalHarmonicsForwardGetWorkspaceSize获取。</td>
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

- 数据类型：`dirs`、`coeffs`、`output` 均为 FLOAT；数据格式均为 ND。
- `dirs`、`coeffs`、`output` 均不支持空tensor，支持非连续Tensor。
- `degrees_to_use` 支持 0~4；`coeffs` 第二维 `K` 必须等于 `(degrees_to_use+1)*(degrees_to_use+1)`，通道维 `C` 支持 1~4，通常为 3。
- `dirs` 与 `coeffs` 的 batchSize 与 gaussian_num 必须一致；`output` 的 shape 为 `(batchSize, C, gaussian_num)`。
- 方向向量在算子内部进行单位化，调用方无需预先归一化。
- gaussian_num 切分到各 AIV 核进行并行计算，`degrees_to_use` 越大 K 越大，单核可处理的高斯球数量相应减少。
- Atlas A2/A3 与 A5（Ascend 950）均支持本算子。

**维度变量说明：**

- batchSize：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- K：球谐系数个数，等于 $(degrees\_to\_use+1)^2$，取值范围为 1（degree=0）~25（degree=4）。
- C：颜色通道数，支持 1~4，通常为 3。

## 调用示例

见tests/kernel_tests/test_spherical_harmonics_forward.cpp
