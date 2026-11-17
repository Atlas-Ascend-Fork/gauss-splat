/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fused_adam_multi_lr_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cmath>
#include <limits>

namespace {
const uint32_t PARAM_PTR_INDEX = 0;
const uint32_t GRAD_PTR_INDEX = 1;
const uint32_t EXP_AVG_PTR_INDEX = 2;
const uint32_t EXP_AVG_SQ_PTR_INDEX = 3;
const uint32_t PARAM_OUT_INDEX = 0;
const uint32_t EXP_AVG_OUT_INDEX = 1;
const uint32_t EXP_AVG_SQ_OUT_INDEX = 2;
const uint32_t DYNAMIC_TENSOR_INDEX_0 = 0;
const uint32_t NUM_ATTR_INDEX = 6;
const uint32_t NUM_DYNAMIC_OUTPUTS = 3;

const uint32_t ALIGN_VALUE = 8;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.8;
const uint32_t BN_NUM = 6;
const uint32_t MAX_TENSOR_NUM = 64;
}

namespace optiling {

struct PerTensorTiling {
    uint32_t paramNum;
    uint32_t totalTaskNum;
    uint32_t tailNum;
    uint32_t taskNumPerScore;
    uint32_t taskNumPerLcore;
    uint32_t numScore;
    uint32_t numLcore;
    uint32_t blockDim;
    int32_t taskNumPerLoop;
};

static PerTensorTiling ComputePerTensorTiling(uint32_t paramNum, uint32_t blockDim, uint64_t ubTotalSize)
{
    PerTensorTiling t{};

    // Defensive guard: callers must pass a non-zero blockDim (the launch grid
    // size); returning the zero-initialized tiling keeps the division below safe.
    if (blockDim == 0) {
        return t;
    }

    uint32_t totalTaskNum;
    if (static_cast<uint32_t>(paramNum % ALIGN_VALUE) == 0) {
        totalTaskNum = paramNum;
    } else {
        totalTaskNum = (static_cast<uint32_t>(paramNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    uint32_t tailNum = totalTaskNum - paramNum;
    uint32_t taskNumPerScore = (totalTaskNum / blockDim / ALIGN_VALUE) * ALIGN_VALUE;
    uint32_t taskNumPerLcore = taskNumPerScore + ALIGN_VALUE;
    uint32_t numScore = (blockDim * taskNumPerLcore - totalTaskNum) / ALIGN_VALUE;
    uint32_t numLcore = blockDim - numScore;

    uint32_t actualBlockDim = blockDim;
    if (taskNumPerScore == 0) {
        actualBlockDim = actualBlockDim - numScore;
    }
    if (taskNumPerLcore == 0) {
        actualBlockDim = actualBlockDim - numLcore;
    }

    int32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (BN_NUM * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    t.paramNum = paramNum;
    t.totalTaskNum = totalTaskNum;
    t.tailNum = tailNum;
    t.taskNumPerScore = taskNumPerScore;
    t.taskNumPerLcore = taskNumPerLcore;
    t.numScore = numScore;
    t.numLcore = numLcore;
    t.blockDim = actualBlockDim;
    t.taskNumPerLoop = taskNumPerLoop;
    return t;
}

struct FusedAdamMultiLrAttrs {
    float beta1;
    float beta2;
    float eps;
    float weightDecay;
    int64_t step;
    int64_t num;
    const float* lrData;
};

struct FusedAdamMultiLrTilingArrays {
    uint32_t paramNums[MAX_TENSOR_NUM];
    uint32_t totalTaskNums[MAX_TENSOR_NUM];
    uint32_t tailNums[MAX_TENSOR_NUM];
    uint32_t taskNumPerScores[MAX_TENSOR_NUM];
    uint32_t taskNumPerLcores[MAX_TENSOR_NUM];
    uint32_t numScores[MAX_TENSOR_NUM];
    uint32_t numLcores[MAX_TENSOR_NUM];
    uint32_t blockDims[MAX_TENSOR_NUM];
    int32_t taskNumPerLoops[MAX_TENSOR_NUM];
    float stepSizes[MAX_TENSOR_NUM];
};

static ge::graphStatus LoadFusedAdamMultiLrAttrs(gert::TilingContext* context, FusedAdamMultiLrAttrs& attrs)
{
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    attrs.beta1 = *(attrsPtr->GetAttrPointer<float>(1));
    attrs.beta2 = *(attrsPtr->GetAttrPointer<float>(2));
    attrs.eps = *(attrsPtr->GetAttrPointer<float>(3));
    attrs.weightDecay = *(attrsPtr->GetAttrPointer<float>(4));
    attrs.step = *(attrsPtr->GetAttrPointer<int64_t>(5));
    attrs.num = *(attrsPtr->GetAttrPointer<int64_t>(6));

    if (attrs.num <= 0 || static_cast<uint32_t>(attrs.num) > MAX_TENSOR_NUM) {
        return ge::GRAPH_FAILED;
    }

    // lr is a ListFloat attribute (index 0)
    auto lrList = attrsPtr->GetListFloat(0);
    if (lrList == nullptr || lrList->GetSize() < static_cast<uint32_t>(attrs.num)) {
        return ge::GRAPH_FAILED;
    }
    attrs.lrData = lrList->GetData();
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetTensorElemCount(gert::TilingContext* context, uint32_t tensorIdx)
{
    auto paramShape = context->GetDynamicInputShape(PARAM_PTR_INDEX, tensorIdx);
    if (paramShape == nullptr) {
        return 0;
    }
    uint64_t elemCount = 1;
    auto storageShape = paramShape->GetStorageShape();
    for (uint32_t i = 0; i < storageShape.GetDimNum(); i++) {
        elemCount *= static_cast<uint64_t>(storageShape.GetDim(i));
    }
    return elemCount;
}

static ge::graphStatus FillPerTensorTilings(gert::TilingContext* context, const FusedAdamMultiLrAttrs& attrs,
                                            uint32_t globalBlockDim, uint64_t ubTotalSize,
                                            FusedAdamMultiLrTilingArrays& arrays)
{
    uint32_t tensorNum = static_cast<uint32_t>(attrs.num);
    float biasCorrection1 = 1.0f - std::pow(attrs.beta1, static_cast<float>(attrs.step));

    for (uint32_t tIdx = 0; tIdx < tensorNum; tIdx++) {
        uint64_t elemCount64 = GetTensorElemCount(context, tIdx);
        if (elemCount64 == 0 || elemCount64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        uint32_t elemCount = static_cast<uint32_t>(elemCount64);

        PerTensorTiling pt = ComputePerTensorTiling(elemCount, globalBlockDim, ubTotalSize);

        arrays.paramNums[tIdx] = pt.paramNum;
        arrays.totalTaskNums[tIdx] = pt.totalTaskNum;
        arrays.tailNums[tIdx] = pt.tailNum;
        arrays.taskNumPerScores[tIdx] = pt.taskNumPerScore;
        arrays.taskNumPerLcores[tIdx] = pt.taskNumPerLcore;
        arrays.numScores[tIdx] = pt.numScore;
        arrays.numLcores[tIdx] = pt.numLcore;
        arrays.blockDims[tIdx] = pt.blockDim;
        arrays.taskNumPerLoops[tIdx] = pt.taskNumPerLoop;

        // Per-tensor stepSize = lr[tIdx] / biasCorrection1
        arrays.stepSizes[tIdx] = attrs.lrData[tIdx] / biasCorrection1;
    }
    return ge::GRAPH_SUCCESS;
}

static void SetTilingScalars(FusedAdamMultiLrTilingData& tiling, const FusedAdamMultiLrAttrs& attrs, uint64_t ubTotalSize)
{
    float biasCorrection2 = 1.0f - std::pow(attrs.beta2, static_cast<float>(attrs.step));
    float invBiasCorrection2 = 1.0f / std::sqrt(biasCorrection2);

    tiling.set_ubTotalSize(ubTotalSize);
    tiling.set_beta1(attrs.beta1);
    tiling.set_beta2(attrs.beta2);
    tiling.set_eps(attrs.eps);
    tiling.set_weightDecay(attrs.weightDecay);
    tiling.set_oneMinusBeta1(1.0f - attrs.beta1);
    tiling.set_oneMinusBeta2(1.0f - attrs.beta2);
    tiling.set_invBiasCorrection2(invBiasCorrection2);
    tiling.set_tensorNum(static_cast<uint32_t>(attrs.num));
}

static void SetTilingArrays(FusedAdamMultiLrTilingData& tiling, FusedAdamMultiLrTilingArrays& arrays)
{
    tiling.set_paramNums(arrays.paramNums);
    tiling.set_totalTaskNums(arrays.totalTaskNums);
    tiling.set_tailNums(arrays.tailNums);
    tiling.set_taskNumPerScores(arrays.taskNumPerScores);
    tiling.set_taskNumPerLcores(arrays.taskNumPerLcores);
    tiling.set_numScores(arrays.numScores);
    tiling.set_numLcores(arrays.numLcores);
    tiling.set_blockDims(arrays.blockDims);
    tiling.set_taskNumPerLoops(arrays.taskNumPerLoops);
    tiling.set_stepSizes(arrays.stepSizes);
}

static uint32_t GetLaunchBlockDim(const FusedAdamMultiLrTilingArrays& arrays, uint32_t tensorNum)
{
    // Launch with the max blockDim across all tensors: each tensor's active
    // cores are selected in-kernel via blockDims[tIdx]; launching with a
    // smaller grid (e.g. the first tensor's) would skip elements of larger
    // tensors processed by higher-indexed cores.
    uint32_t launchBlockDim = 0;
    for (uint32_t tIdx = 0; tIdx < tensorNum; tIdx++) {
        if (arrays.blockDims[tIdx] > launchBlockDim) {
            launchBlockDim = arrays.blockDims[tIdx];
        }
    }
    return launchBlockDim;
}

static ge::graphStatus TilingForFusedAdamMultiLr(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    FusedAdamMultiLrAttrs attrs;
    if (LoadFusedAdamMultiLrAttrs(context, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint64_t ubTotalSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotalSize);
    uint32_t maxBlockDim = platformInfo.GetCoreNumAiv();
    if (maxBlockDim == 0) {
        return ge::GRAPH_FAILED;
    }

    // All tensors share the same core count for simplicity
    FusedAdamMultiLrTilingData tiling;
    FusedAdamMultiLrTilingArrays arrays = {};
    if (FillPerTensorTilings(context, attrs, maxBlockDim, ubTotalSize, arrays) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetTilingScalars(tiling, attrs, ubTotalSize);
    SetTilingArrays(tiling, arrays);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(GetLaunchBlockDim(arrays, static_cast<uint32_t>(attrs.num)));
    context->SetTilingKey(1);

    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t num = *(attrsPtr->GetAttrPointer<int64_t>(NUM_ATTR_INDEX));
    if (num <= 0) {
        return ge::GRAPH_FAILED;
    }

    // Instance k of each of the 3 DYNAMIC outputs mirrors the shape of param
    // instance k. Iterate the real tensor count (num), not the flattened
    // output-instance count (NUM_DYNAMIC_OUTPUTS * num).
    for (int64_t k = 0; k < num; k++) {
        const gert::Shape* paramShape = context->GetDynamicInputShape(PARAM_PTR_INDEX, k);
        if (paramShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        for (uint32_t outIdx = 0; outIdx < NUM_DYNAMIC_OUTPUTS; outIdx++) {
            // CANN 9.1.0 has no per-(output, instance) API; dynamic output
            // instances are flattened in IR order: outIdx * num + k.
            size_t flatIdx = static_cast<size_t>(outIdx) * static_cast<size_t>(num) + static_cast<size_t>(k);
            gert::Shape* outShape = context->GetOutputShape(flatIdx);
            if (outShape == nullptr) {
                return ge::GRAPH_FAILED;
            }
            *outShape = *paramShape;
        }
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t num = *(attrsPtr->GetAttrPointer<int64_t>(NUM_ATTR_INDEX));
    if (num <= 0) {
        return ge::GRAPH_FAILED;
    }
    for (int64_t k = 0; k < num; k++) {
        const auto inputDataType = context->GetDynamicInputDataType(PARAM_PTR_INDEX, k);
        for (uint32_t outIdx = 0; outIdx < NUM_DYNAMIC_OUTPUTS; outIdx++) {
            // Same flat layout as InferShape: outIdx * num + k.
            size_t flatIdx = static_cast<size_t>(outIdx) * static_cast<size_t>(num) + static_cast<size_t>(k);
            context->SetOutputDataType(flatIdx, inputDataType);
        }
    }
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class FusedAdamMultiLr : public OpDef {
public:
    explicit FusedAdamMultiLr(const char* name) : OpDef(name)
    {
        this->Input("param")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("grad")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("exp_avg")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("exp_avg_sq")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("param_out")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("exp_avg_out")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("exp_avg_sq_out")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("lr").ListFloat();
        this->Attr("beta1").Float();
        this->Attr("beta2").Float();
        this->Attr("eps").Float();
        this->Attr("weight_decay").Float();
        this->Attr("step").Int();
        this->Attr("num").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForFusedAdamMultiLr);
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(FusedAdamMultiLr);
}