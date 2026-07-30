# aclnnSphericalHarmonicsBwd

[📄 查看源码](https://gitcode.com/cann/gauss-splat/blob/master/kernels/spherical_harmonics_bwd/op_kernel/spherical_harmonics_bwd.cpp)

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

- 接口功能：`spherical_harmonics_bwd`计算球谐函数的反向传播，用于3D Gaussian Splatting训练中的梯度计算。
- 计算公式：

反向传播计算方向和系数的梯度：

$$
v\_dirs = \frac{\partial L}{\partial dirs}, \quad v\_coeffs = \frac{\partial L}{\partial coeffs}
$$

基于前向公式 $output = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} \cdot Y_{lm}(dirs)$ 进行反向梯度计算。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSphericalHarmonicsBwdGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSphericalHarmonicsBwd"接口执行计算。

```Cpp
aclnnStatus aclnnSphericalHarmonicsBwdGetWorkspaceSize(
    const aclTensor *dirs,
    const aclTensor *coeffs,
    const aclTensor *vColors,
    int64_t          degree,
    const aclTensor *vDirs,
    const aclTensor *vCoeffs,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSphericalHarmonicsBwd(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnSphericalHarmonicsBwdGetWorkspaceSize

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
      <td>不支持空tensor。K必须等于(degree+1)*(degree+1)。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, K, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vColors</td>
      <td>输入</td>
      <td>颜色的梯度输入。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>degree</td>
      <td>输入</td>
      <td>使用的球谐阶数。</td>
      <td>支持0~3。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vDirs</td>
      <td>输出</td>
      <td>方向的梯度输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vCoeffs</td>
      <td>输出</td>
      <td>系数的梯度输出。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batchSize, K, 3, gaussian_num)</td>
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
                <td>dirs、coeffs、vColors、vDirs、vCoeffs的数据类型和数据格式不在支持的范围内，或degree不在0~3范围内。</td>
            </tr>
        </tbody>
    </table>

## aclnnSphericalHarmonicsBwd

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSphericalHarmonicsBwdGetWorkspaceSize获取。</td>
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

## 调用示例

见tests/kernel_tests/test_spherical_harmonics_bwd.cpp
