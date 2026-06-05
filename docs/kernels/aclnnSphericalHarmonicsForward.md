# aclnnSphericalHarmonicsForward

[📄 查看源码](https://gitcode.com/cann/gauss-splat/tree/master/kernels/spherical_harmonics_forward)

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

- 接口功能：`spherical_harmonics_forward`计算球谐函数的前向传播，用于3D Gaussian Splatting中的颜色渲染。
- 计算公式：

球谐函数将方向向量映射到颜色空间，公式如下：

$$
output = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_{lm}(dirs)
$$

其中$Y_{lm}$是$l$阶$m$次的球谐基函数，$c_{lm}$是球谐系数，$dirs$是方向向量。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSphericalHarmonicsForwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSphericalHarmonicsForward"接口执行计算。

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
      <td>(3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>coeffs</td>
      <td>输入</td>
      <td>球谐系数，用于颜色计算。</td>
      <td>不支持空tensor。系数数量与degrees_to_use相关。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(K, 3, gaussian_num)，其中K=(degrees_to_use+1)*(degrees_to_use+1)</td>
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
      <td>(3, gaussian_num)</td>
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
                <td>dirs、coeffs、output的数据类型和数据格式不在支持的范围内，或degrees_to_use不在0~4范围内。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 调用示例

见tests/kernel_tests/test_spherical_harmonics_forward.cpp
