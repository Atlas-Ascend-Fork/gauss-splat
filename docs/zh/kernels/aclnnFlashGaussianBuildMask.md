# aclnnFlashGaussianBuildMask

[📄 查看源码 (arch22)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/flash_gaussian_build_mask/op_kernel/arch22/flash_gaussian_build_mask.cpp) | [📄 查看源码 (arch35)](https://gitcode.com/cann/gauss-splat/blob/master/kernels/flash_gaussian_build_mask/op_kernel/arch35/flash_gaussian_build_mask.cpp)

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

- 接口功能：`flash_gaussian_build_mask` 对每个 (batch, camera) 视图下的高斯球按 tile 归属进行三级过滤与有序压缩，构建 Flash 渲染所需的 tile 掩码（每个 tile 内有效高斯球的数量、索引与深度列表），用于优化后续渲染的并行计算效率。
- 计算公式：

对每个 (batch, camera, tile)：

$$
tile\_sum[b,c,t] = \sum_{g} valid\_mask[b,c,t,g]
$$

$$
gauss\_index[b,c,t,0:tile\_sum] = \{\, g \mid valid\_mask[b,c,t,g] = 1 \,\}
$$

$$
tile\_depths[b,c,t,0:tile\_sum] = depths[b,c,g] \big|_{g \in gauss\_index[b,c,t]}
$$

其中 $valid\_mask$ 由三级过滤确定：先按 tile 的 Y 轴范围预过滤，再做包围盒粗过滤，最后做中心点/Tile 相交与椭圆相交精过滤。

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
    double           imageWidth,
    double           imageHeight,
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
      <td>可选输入，支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 2, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>opacity</td>
      <td>输入</td>
      <td>高斯球透明度。</td>
      <td>可选输入，支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>conics</td>
      <td>输入</td>
      <td>锥形参数。</td>
      <td>可选输入，支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>covars2d</td>
      <td>输入</td>
      <td>2D协方差矩阵。</td>
      <td>可选输入，支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 3, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>depths</td>
      <td>输入</td>
      <td>高斯球深度。</td>
      <td>可选输入，支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, 1, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cnt</td>
      <td>输入</td>
      <td>每个视图内有效的高斯球计数。</td>
      <td>可选输入，支持空tensor；元素取值需满足 0 ≤ cnt ≤ gaussian_num。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num) 或 (batch_size, camera_num, 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileGrid</td>
      <td>输入</td>
      <td>Tile网格信息。</td>
      <td>可选输入，支持空tensor；numTile = ceil(image_height / tile_size) * ceil(image_width / tile_size)。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(numTile, 2)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gatherMask</td>
      <td>输入</td>
      <td>核间/级间 Gather 中转区，不参与输出数学计算。</td>
      <td>可选输入，支持空tensor；block_num 为按平台适配的工作区行数（A2 系列通常为 48，A5 系列为 56/64），每个核独占一行。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(block_num, 20, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>imageWidth</td>
      <td>输入</td>
      <td>图像宽度。</td>
      <td>浮点数，取值范围 [1.0, 4294967295.0]。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>imageHeight</td>
      <td>输入</td>
      <td>图像高度。</td>
      <td>浮点数，取值范围 [1.0, 4294967295.0]。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSize</td>
      <td>输入</td>
      <td>Tile大小。</td>
      <td>正整数，tile 划分粒度，需与 tileGrid 的 numTile 满足 numTile = ceil(image_height / tile_size) * ceil(image_width / tile_size)。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tileSum</td>
      <td>输出</td>
      <td>每个tile内有效高斯球的数量总和。</td>
      <td>必选输出，支持空tensor（batch_size/camera_num/gaussian_num 任一为 0 时输出为空）。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tileDepths</td>
      <td>输出</td>
      <td>每个tile内按序排列的高斯球深度列表。</td>
      <td>必选输出，支持空tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>(batch_size, camera_num, numTile, gaussian_num)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gaussIndex</td>
      <td>输出</td>
      <td>每个tile内按序排列的高斯球索引列表。</td>
      <td>必选输出，支持空tensor。</td>
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
                <td>输入输出tensor的数据类型、数据格式或shape不在支持的范围内，或参数值/属性值不合法（例如 cnt 元素取值不在 [0, gaussian_num] 范围内）。</td>
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

- 数据类型：means2d、opacity、conics、covars2d、depths、tileGrid、gatherMask、tileDepths、gaussIndex 为 FLOAT；cnt、tileSum 为 INT32。数据格式均为 ND。
- 8 个输入均为可选输入，支持空tensor：当 batch_size、camera_num 或 gaussian_num 任一为 0 时进入空分支，仅将 tileSum 清零，tileDepths、gaussIndex 为 0 元素空列表且不写回。
- cnt 支持 rank 2 的 (batch_size, camera_num) 或 rank 3 的 (batch_size, camera_num, 1)；每个元素取值需满足 0 ≤ cnt[b,c] ≤ gaussian_num，越界时第一段接口返回 161002。
- means2d、opacity、conics、covars2d、depths 的通道维分别为 2、1、3、3、1，且其 batch_size、camera_num、gaussian_num 需与 cnt、gatherMask 保持一致。
- tileGrid 的 shape 为 (numTile, 2)，numTile 需满足 numTile = ceil(image_height / tile_size) * ceil(image_width / tile_size)，取值范围 [1, 65536]。
- gatherMask 的 shape 为 (block_num, 20, gaussian_num)，block_num 为调用侧按平台适配的工作区行数（A2 系列通常为 48，A5 系列为 56/64）。
- 属性约束：image_width、image_height 取值范围 [1.0, 4294967295.0]；tile_size 为正整数。
- gaussian_num 的取值范围为 [0, 2^24 - 1]。
- tileSize 通常为16，用于tile分区计算。

**维度变量说明：**

- batch_size：批次大小，表示一次处理的数据批次数量。
- gaussian_num：高斯球数量，表示每个批次中3D高斯球的个数。
- camera_num：相机数量，表示参与计算的相机个数。
- numTile：tile数量，等于 ceil(image_height / tile_size) * ceil(image_width / tile_size)。
- block_num：gatherMask 第一维行数（gather 工作区行数），由平台适配。

## 调用示例

见tests/kernel_tests/test_flash_gaussian_build_mask.cpp
