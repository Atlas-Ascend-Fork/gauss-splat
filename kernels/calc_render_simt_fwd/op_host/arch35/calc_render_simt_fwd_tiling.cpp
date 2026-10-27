/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "calc_render_simt_fwd_tiling.h"

namespace optiling {
namespace {
constexpr size_t kInputCount = 4;
constexpr size_t kTensorCount = 9;
// HostTiling.md §3 固定维契约：N_PIXEL_FIXED = 1024（32×32 tile，T1）、
// GS_ATTR_NUM = 10（gs 属性布局）。
constexpr int64_t kPixelCount = 1024;
constexpr int64_t kGsAttrNum = 10;
// HostTiling.md §3 行 5：各维 ∈ [0, 2^31−1]（含 NGauss < 2^31，T8 前提）。
constexpr int64_t kMaxDim = 2147483647;
constexpr std::array<const char*, kTensorCount> kNames = {
    "gs", "tileCoords", "offsets", "gsIds", "color", "depth", "lastCumsum", "error", "lastIds"};
constexpr std::array<ge::DataType, kTensorCount> kDtypes = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT64, ge::DT_INT64,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32};
constexpr std::array<size_t, kTensorCount> kRanks = {2, 3, 1, 1, 3, 2, 2, 2, 2};
using Shapes = std::array<const gert::StorageShape*, kTensorCount>;

ge::graphStatus Fail(const char* reason)
{
    return ge::GRAPH_FAILED;
}

// 校验顺序 HostTiling.md §3：dtype → format → rank（维度值检查后置，交
// ValidateInputs 设计锚点与维域兜底，保证 tileNum<0 命中设计锚点串）。
bool CheckTensorMetadata(gert::TilingContext* context, Shapes& shapes)
{
    for (size_t i = 0; i < kTensorCount; ++i) {
        const auto* desc = i < kInputCount ? context->GetInputDesc(i) : context->GetOutputDesc(i - kInputCount);
        shapes[i] = i < kInputCount ? context->GetInputShape(i) : context->GetOutputShape(i - kInputCount);
        if (desc == nullptr || shapes[i] == nullptr) {
            return false;
        }
        if (desc->GetDataType() != kDtypes[i]) {
            return false;
        }
        if (desc->GetStorageFormat() != ge::FORMAT_ND || desc->GetOriginFormat() != ge::FORMAT_ND) {
            return false;
        }
        const auto& storage = shapes[i]->GetStorageShape();
        const auto& origin = shapes[i]->GetOriginShape();
        if (storage.GetDimNum() != kRanks[i] || origin.GetDimNum() != kRanks[i]) {
            return false;
        }
        for (size_t d = 0; d < kRanks[i]; ++d) {
            if (origin.GetDim(d) != storage.GetDim(d)) {
                return false;
            }
        }
    }
    return true;
}

// HostTiling.md §3 ValidateInputs（T9 合同外兜底：断言 + 报错返回，锚点串与设计逐字一致）
static ge::graphStatus ValidateInputs(const gert::TilingContext* context, int64_t tileNum,
                                      int64_t nPixel, int64_t attrDim)
{
    if (nPixel != kPixelCount /* 1024 */ || attrDim != kGsAttrNum /* 10 */) {
        return ge::GRAPH_FAILED;  // shape_mismatch
    }
    if (tileNum < 0) {
        return ge::GRAPH_FAILED;  // shape_mismatch
    }
    return ge::GRAPH_SUCCESS;
}

// HostTiling.md §3 行 5：各维 ∈ [0, 2^31−1]（非负 + 上界兜底；NGauss < 2^31，T8）。
bool CheckDimDomain(const Shapes& shapes)
{
    for (size_t i = 0; i < kTensorCount; ++i) {
        const auto& shape = shapes[i]->GetStorageShape();
        for (size_t d = 0; d < kRanks[i]; ++d) {
            if (shape.GetDim(d) < 0 || shape.GetDim(d) > kMaxDim) {
                return false;
            }
        }
    }
    return true;
}

// HostTiling.md §3 行 3/行 5：tileCoords 坐标轴维 = 2；五路输出 host 预分配契约
// color [3,tileNum,1024]、depth/lastCumsum/error/lastIds [tileNum,1024]。
bool CheckShapes(const Shapes& shapes, int64_t tiles)
{
    const auto& coords = shapes[1]->GetStorageShape();
    if (coords.GetDim(1) != 2) {
        Fail("tileCoords coord-axis dim must be 2");
        return false;
    }
    const auto& color = shapes[4]->GetStorageShape();
    if (color.GetDim(0) != 3 || color.GetDim(1) != tiles || color.GetDim(2) != kPixelCount) {
        Fail("color shape must be [3,tileNum,1024]");
        return false;
    }
    for (size_t i = 5; i < kTensorCount; ++i) {
        const auto& shape = shapes[i]->GetStorageShape();
        if (shape.GetDim(0) != tiles || shape.GetDim(1) != kPixelCount) {
            return false;
        }
    }
    return true;
}
} // namespace

static ge::graphStatus CalcRenderSimtFwdTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) return Fail("context is null");
    if (context->GetComputeNodeInfo() == nullptr) return Fail("context compute-node metadata is null");
    Shapes shapes{};
    if (!CheckTensorMetadata(context, shapes)) return ge::GRAPH_FAILED;

    // 1) 读输入 shape（仅 shape，无值依赖，HostTiling.md §2）
    const auto& gs = shapes[0]->GetStorageShape();
    const auto& coords = shapes[1]->GetStorageShape();
    const int64_t nPixel  = coords.GetDim(2);
    const int64_t tileNum = coords.GetDim(0);
    const int64_t attrDim = gs.GetDim(1);
    const int64_t nGauss  = gs.GetDim(0);

    // 2) 合同校验（§3：nPixel==1024、attrDim==10、tileNum>=0 设计锚点；各维 ∈
    //    [0,2^31-1]；坐标轴维/输出 shape 一致性。任一失败报错返回，不进入 tiling 计算）
    auto ret = ValidateInputs(context, tileNum, nPixel, attrDim);
    if (ret != ge::GRAPH_SUCCESS) { return ret; }
    if (!CheckDimDomain(shapes)) return ge::GRAPH_FAILED;
    if (!CheckShapes(shapes, tileNum)) return ge::GRAPH_FAILED;

    // 3) 平台信息动态查询 + 核数计算（§5：blockDim = GetCoreNumAiv()，0 兜底 1，
    //    禁止硬编码；核间 tile 分摊由 offsets 调度表数据驱动，host 不重算）
    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) return Fail("platform is null");
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t queriedCores = platform.GetCoreNumAiv();
    const uint32_t blockDim = queriedCores == 0 ? 1U : queriedCores;

    // §3 行 5：len(offsets) == vecNum + 2*TileNum 兜底断言（aclnn L2 / 调用方
    // 保证，Tiling 侧兜底；vecNum = blockDim 生效值）
    const int64_t offsetLength = static_cast<int64_t>(blockDim) + 2 * tileNum;
    if (shapes[2]->GetStorageShape().GetDim(0) != offsetLength) {
        return ge::GRAPH_FAILED;
    }

    // 4) UB 动态查询 + DCache 预留（§4：SIMT 不能占用全部 UB；host 侧只透传
    //    ubSize 字段，容量校验在 kernel 侧 Init/UbBuffInit）
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto* raw = context->GetRawTilingData();
    if (raw == nullptr || raw->GetCapacity() < sizeof(CalcRenderSimtFwdTilingData) || raw->GetData() == nullptr) {
        return Fail("tiling buffer is null or capacity below 48 bytes");
    }
    auto* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) return Fail("workspace slot is null");
    if (context->SetBlockDim(blockDim) != ge::GRAPH_SUCCESS ||
        context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_RESERVED)) != ge::GRAPH_SUCCESS) {
        return Fail("context cannot set blockDim/UB launch outputs");
    }

    // 5) 填 TilingData：6 字段全量赋值 + 日志打印（§8；空输入不短路，§11）
    //    The common POD is shared with Device. Zero all bytes, including trailing
    //    padding, before assigning fields; no macro-generated class is reinterpreted.
    auto* td = static_cast<CalcRenderSimtFwdTilingData*>(raw->GetData());
    std::memset(td, 0, sizeof(*td));
    td->nPixel = nPixel;
    td->tileNum = tileNum;
    td->nGauss = nGauss;
    td->totalGauss = shapes[3]->GetStorageShape().GetDim(0);   // gsIds 长度（Task 30 段界防御）
    td->ubSize = ubSize;
    td->blockDim = blockDim;
    raw->SetDataSize(sizeof(*td));

    // 6) SetBlockDim + SetTilingKey + workspace=0（§8：SetBlockDim 与 tiling.blockDim
    //    同源同值；单分支固定 tilingKey=0；无跨核交换）
    if (context->SetTilingKey(0) != ge::GRAPH_SUCCESS) {
        return Fail("context cannot set tilingKey");
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus Parse(gert::TilingParseContext* context)
{
    return context != nullptr ? ge::GRAPH_SUCCESS : Fail("context for tiling parse is null");
}
struct CalcRenderSimtFwdCompileInfo {};
IMPL_OP_OPTILING(CalcRenderSimtFwd).Tiling(CalcRenderSimtFwdTilingFunc).TilingParse<CalcRenderSimtFwdCompileInfo>(Parse);
} // namespace optiling

// [REF_SAMPLE] Original sample retained for reference only.
// [REF_SAMPLE] /**
// [REF_SAMPLE]  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
// [REF_SAMPLE]  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// [REF_SAMPLE]  * CANN Open Software License Agreement Version 2.0 (the "License").
// [REF_SAMPLE]  * Please refer to the License for details. You may not use this file except in compliance with the License.
// [REF_SAMPLE]  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// [REF_SAMPLE]  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// [REF_SAMPLE]  * See LICENSE in the root of the software repository for the full text of the License.
// [REF_SAMPLE]  */
// [REF_SAMPLE] 
// [REF_SAMPLE] /*!
// [REF_SAMPLE]  * \file calc_render_simt_fwd_tiling.cpp
// [REF_SAMPLE]  * \brief
// [REF_SAMPLE]  */
// [REF_SAMPLE] 
// [REF_SAMPLE] #include "register/op_def_registry.h"
// [REF_SAMPLE] #include "op_common/log/log.h"
// [REF_SAMPLE] #include "op_common/op_host/util/math_util.h"
// [REF_SAMPLE] #include "op_common/op_host/util/platform_util.h"
// [REF_SAMPLE] #include "graph/utils/type_utils.h"
// [REF_SAMPLE] #include "../../op_kernel/arch35/calc_render_simt_fwd_tiling_data.h"
// [REF_SAMPLE] #include "../../op_kernel/arch35/calc_render_simt_fwd_tiling_key.h"
// [REF_SAMPLE] 
// [REF_SAMPLE] namespace optiling {
// [REF_SAMPLE] 
// [REF_SAMPLE] using Ops::Base::CeilDiv;
// [REF_SAMPLE] using Ops::Base::CeilAlign;
// [REF_SAMPLE] using Ops::Base::FloorDiv;
// [REF_SAMPLE] using Ops::Base::FloorAlign;
// [REF_SAMPLE] using Ops::Base::GetUbBlockSize;
// [REF_SAMPLE] 
// [REF_SAMPLE] constexpr uint32_t WS_SYS_SIZE = 0U;
// [REF_SAMPLE] constexpr int64_t TYPE_SIZE = 4;
// [REF_SAMPLE] constexpr size_t WORKSPACE_NUM = 1;
// [REF_SAMPLE] constexpr int64_t SINGLE_BUF_TENSOR_COUNT = 3;
// [REF_SAMPLE] constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 6;
// [REF_SAMPLE] // 双缓冲阈值：数据量大于此值时启用双缓冲
// [REF_SAMPLE] constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// [REF_SAMPLE] 
// [REF_SAMPLE] static const gert::Shape g_vec_1_shape = {1};
// [REF_SAMPLE] 
// [REF_SAMPLE] static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
// [REF_SAMPLE]     if (in_shape.GetDimNum() == 0) {
// [REF_SAMPLE]         return g_vec_1_shape;
// [REF_SAMPLE]     }
// [REF_SAMPLE]     return in_shape;
// [REF_SAMPLE] }
// [REF_SAMPLE] 
// [REF_SAMPLE] // 获取平台信息如ubSize, coreNum
// [REF_SAMPLE] static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
// [REF_SAMPLE] {
// [REF_SAMPLE]     // 获取ubsize coreNum
// [REF_SAMPLE]     fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
// [REF_SAMPLE]     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
// [REF_SAMPLE]     *coreNum = ascendcPlatform.GetCoreNumAiv();
// [REF_SAMPLE]     OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
// [REF_SAMPLE]     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
// [REF_SAMPLE]     OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
// [REF_SAMPLE]     return ge::GRAPH_SUCCESS;
// [REF_SAMPLE] }
// [REF_SAMPLE] 
// [REF_SAMPLE] // 获取属性，shape信息
// [REF_SAMPLE] static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalIdx, ge::DataType* dataType)
// [REF_SAMPLE] {
// [REF_SAMPLE]     // 获取输入shape信息
// [REF_SAMPLE]     auto inputX = context->GetInputShape(0);
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
// [REF_SAMPLE]     // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
// [REF_SAMPLE]     auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
// [REF_SAMPLE]     auto inputY = context->GetInputShape(1);
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, inputY);
// [REF_SAMPLE]     auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());
// [REF_SAMPLE]     auto outZ = context->GetOutputShape(0);
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
// [REF_SAMPLE]     auto outShapeZ = EnsureNotScalar(outZ->GetStorageShape());
// [REF_SAMPLE] 
// [REF_SAMPLE]     // shape校验：确保输入和输出shape一致
// [REF_SAMPLE]     OP_CHECK_IF(
// [REF_SAMPLE]         inputShapeX.GetShapeSize() != inputShapeY.GetShapeSize() ||
// [REF_SAMPLE]             inputShapeX.GetShapeSize() != outShapeZ.GetShapeSize(),
// [REF_SAMPLE]         OP_LOGE(
// [REF_SAMPLE]             context, "CalcRenderSimtFwd: input and output shape size mismatch: x=%ld, y=%ld, z=%ld",
// [REF_SAMPLE]             inputShapeX.GetShapeSize(), inputShapeY.GetShapeSize(), outShapeZ.GetShapeSize()),
// [REF_SAMPLE]         return ge::GRAPH_FAILED);
// [REF_SAMPLE] 
// [REF_SAMPLE]     // 获取shape dim值
// [REF_SAMPLE]     *totalIdx = inputShapeX.GetShapeSize();
// [REF_SAMPLE]     // dtype校验
// [REF_SAMPLE]     const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32};
// [REF_SAMPLE]     auto inputDesc = context->GetInputDesc(0);
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
// [REF_SAMPLE]     *dataType = inputDesc->GetDataType();
// [REF_SAMPLE]     OP_CHECK_IF(supportedDtype.count(*dataType) == 0, OP_LOGE(context, "invalid dtype"), return ge::GRAPH_FAILED);
// [REF_SAMPLE]     return ge::GRAPH_SUCCESS;
// [REF_SAMPLE] }
// [REF_SAMPLE] 
// [REF_SAMPLE] // 即使 workspace 大小为 0，也必须调用 GetWorkspaceSizes 声明 workspace slot，否则运行时不会为 kernel 分配 workspace 指针
// [REF_SAMPLE] static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
// [REF_SAMPLE] {
// [REF_SAMPLE]     size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
// [REF_SAMPLE]     currentWorkspace[0] = WS_SYS_SIZE;
// [REF_SAMPLE]     return ge::GRAPH_SUCCESS;
// [REF_SAMPLE] }
// [REF_SAMPLE] 
// [REF_SAMPLE] // tiling 分发入口
// [REF_SAMPLE] static ge::graphStatus CalcRenderSimtFwdTilingFunc(gert::TilingContext* context)
// [REF_SAMPLE] {
// [REF_SAMPLE]     // 1、获取平台运行信息
// [REF_SAMPLE]     uint64_t ubSize;
// [REF_SAMPLE]     int64_t coreNum;
// [REF_SAMPLE]     OP_CHECK_IF(
// [REF_SAMPLE]         GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
// [REF_SAMPLE]         return ge::GRAPH_FAILED);
// [REF_SAMPLE]     // 2、获取shape、属性信息
// [REF_SAMPLE]     int64_t totalIdx;
// [REF_SAMPLE]     ge::DataType dataType;
// [REF_SAMPLE]     OP_CHECK_IF(
// [REF_SAMPLE]         GetShapeAttrsInfo(context, &totalIdx, &dataType) != ge::GRAPH_SUCCESS,
// [REF_SAMPLE]         OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
// [REF_SAMPLE]     // 3、获取WorkspaceSize信息
// [REF_SAMPLE]     OP_CHECK_IF(
// [REF_SAMPLE]         GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
// [REF_SAMPLE]         return ge::GRAPH_FAILED);
// [REF_SAMPLE] 
// [REF_SAMPLE]     // 4、设置tiling信息, TilingData空间由kernel侧的REGISTER_TILING_DEFAULT决定
// [REF_SAMPLE]     CalcRenderSimtFwdTilingData* tiling = context->GetTilingData<CalcRenderSimtFwdTilingData>();
// [REF_SAMPLE]     OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
// [REF_SAMPLE]     OP_CHECK_IF(
// [REF_SAMPLE]         memset_s(tiling, sizeof(CalcRenderSimtFwdTilingData), 0, sizeof(CalcRenderSimtFwdTilingData)) != EOK,
// [REF_SAMPLE]         OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
// [REF_SAMPLE]     
// [REF_SAMPLE]     // 空Tensor 检查，避免后续除零风险
// [REF_SAMPLE]     if (totalIdx == 0) {
// [REF_SAMPLE]         context->SetBlockDim(1);
// [REF_SAMPLE]         ASCENDC_TPL_SEL_PARAM(context, 0);
// [REF_SAMPLE]         OP_LOGD(context, "TilingKey: bufferMode=SINGLE (empty tensor path) | dtype=%s",
// [REF_SAMPLE]                 ge::TypeUtils::DataTypeToAscendString(dataType).GetString());
// [REF_SAMPLE]         return ge::GRAPH_SUCCESS;
// [REF_SAMPLE]     }
// [REF_SAMPLE] 
// [REF_SAMPLE]     // 多核切分：将总元素按核数均分，按 DMA 最小粒度 (32B) 对齐
// [REF_SAMPLE]     // blockFactor 必须按 blockAlign 向上对齐，否则相邻核 CopyOut 写 GM 时
// [REF_SAMPLE]     // DMA 实际写入范围会超过 blockLen，覆盖相邻核的数据
// [REF_SAMPLE]     int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
// [REF_SAMPLE]     tiling->totalNum = totalIdx;
// [REF_SAMPLE]     tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
// [REF_SAMPLE]     int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);
// [REF_SAMPLE] 
// [REF_SAMPLE]     // UB 切分：按可用 UB 大小和 tensor 数量均分
// [REF_SAMPLE]     // 双缓冲时 UB tensor 数量：2输入×2(双缓冲) + 1输出×2(双缓冲) = 6
// [REF_SAMPLE]     // 单缓冲时 UB tensor 数量：2输入×1 + 1输出×1 = 3
// [REF_SAMPLE]     // 公式：(UB总大小 / 类型大小) / tensor数量，然后按 UB 块大小对齐
// [REF_SAMPLE]     uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
// [REF_SAMPLE]     int64_t bufferNum = useDoubleBuffer ? DOUBLE_BUF_TENSOR_COUNT : SINGLE_BUF_TENSOR_COUNT;
// [REF_SAMPLE]     tiling->ubFactor = Ops::Base::FloorAlign(Ops::Base::FloorDiv((static_cast<int64_t>(ubSize) / TYPE_SIZE), bufferNum), ubBlockSize);
// [REF_SAMPLE] 
// [REF_SAMPLE]     OP_CHECK_IF(tiling->ubFactor <= 0,
// [REF_SAMPLE]                 OP_LOGE(context, "ubFactor=%ld invalid (ubSize=%lu, bufferNum=%ld)",
// [REF_SAMPLE]                         tiling->ubFactor, ubSize, bufferNum),
// [REF_SAMPLE]                 return ge::GRAPH_FAILED);
// [REF_SAMPLE] 
// [REF_SAMPLE]     context->SetBlockDim(usedCoreNum);
// [REF_SAMPLE] 
// [REF_SAMPLE]     // 5、设置 TilingKey(必需，重要)
// [REF_SAMPLE]     // 参数顺序与calc_render_simt_fwd_tiling_key.h中 ASCENDC_TPL_ARGS_DECL 定义一致
// [REF_SAMPLE]     // 参数会映射到calc_render_simt_fwd_arch35.cpp中kernel 入口模板参数：calc_render_simt_fwd<BUFFER_MODE>
// [REF_SAMPLE]     ASCENDC_TPL_SEL_PARAM(context, useDoubleBuffer);
// [REF_SAMPLE]     OP_LOGD(context,
// [REF_SAMPLE]             "TilingKey: bufferMode=%s | dtype=%s | TilingData: totalNum=%ld, blockFactor=%ld, ubFactor=%ld, usedCoreNum=%ld",
// [REF_SAMPLE]             useDoubleBuffer ? "DOUBLE" : "SINGLE",
// [REF_SAMPLE]             ge::TypeUtils::DataTypeToAscendString(dataType).GetString(),
// [REF_SAMPLE]             tiling->totalNum, tiling->blockFactor, tiling->ubFactor, usedCoreNum);
// [REF_SAMPLE]     return ge::GRAPH_SUCCESS;
// [REF_SAMPLE] }
// [REF_SAMPLE] 
// [REF_SAMPLE] static ge::graphStatus TilingParseForCalcRenderSimtFwd([[maybe_unused]] gert::TilingParseContext* context)
// [REF_SAMPLE] {
// [REF_SAMPLE]     return ge::GRAPH_SUCCESS;
// [REF_SAMPLE] }
// [REF_SAMPLE] 
// [REF_SAMPLE] struct CalcRenderSimtFwdCompileInfo {}; // 必须定义, 入图场景依赖
// [REF_SAMPLE] 
// [REF_SAMPLE] // tiling注册入口.
// [REF_SAMPLE] IMPL_OP_OPTILING(CalcRenderSimtFwd).Tiling(CalcRenderSimtFwdTilingFunc).TilingParse<CalcRenderSimtFwdCompileInfo>(TilingParseForCalcRenderSimtFwd);
// [REF_SAMPLE] 
// [REF_SAMPLE] } // namespace optiling
