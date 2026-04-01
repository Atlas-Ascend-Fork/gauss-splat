/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spherical_harmonics_forward_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
const uint32_t DIRS_PTR_INDEX = 0;
const uint32_t COEFFS_PTR_INDEX = 1;
const uint32_t OUTPUT_PTR_INDEX = 0;

const uint32_t L2_SH_DEGREE = 2;
const uint32_t L3_SH_DEGREE = 3;
const uint32_t L4_SH_DEGREE = 4;
const uint32_t L2_SH_BUFFER_NUM = 6;
const uint32_t L3_SH_BUFFER_NUM = 10;
const uint32_t L4_SH_BUFFER_NUM = 16;

const uint32_t TASK_NUM_INDEX = 1;
const uint32_t COEFF_NUM_INDEX = 0;
const uint32_t DEGREE_USED_INDEX = 0;
const uint32_t ALIGN_VALUE = 8;

const uint32_t BN_NUM = 6;
const uint32_t BC_NUM = 4;
const int32_t FLOAT_SIZE = 4;
const float UB_RATIO = 0.8;
}

namespace optiling {
static ge::graphStatus TilingForSphericalHarmonicsForward(gert::TilingContext* context)
{
    SphericalHarmonicsForwardTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dirsTensorPtr = context->GetInputTensor(DIRS_PTR_INDEX);
    auto coeffsTensorPtr = context->GetInputTensor(COEFFS_PTR_INDEX);
    if (dirsTensorPtr == nullptr || coeffsTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto dirsShape = context->GetInputShape(DIRS_PTR_INDEX);
    auto coeffsShape = context->GetInputShape(COEFFS_PTR_INDEX);
    if (dirsShape == nullptr || coeffsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t taskNum = dirsShape->GetStorageShape().GetDim(TASK_NUM_INDEX);
    uint32_t coeffNum = coeffsShape->GetStorageShape().GetDim(COEFF_NUM_INDEX);

    auto attrsPtr = context->GetAttrs();
    if (attrsPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t degreeUsed = *(attrsPtr->GetAttrPointer<uint32_t>(DEGREE_USED_INDEX));
    uint32_t degreeNum = (degreeUsed + 1) * (degreeUsed + 1);

    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint64_t ubTotalSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubTotalSize);
    uint32_t blockDim = platformInfo.GetCoreNumAiv();
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalTaskNum;
    if (static_cast<uint32_t>((taskNum) % ALIGN_VALUE) == 0) {
        totalTaskNum = taskNum;
    } else {
        totalTaskNum = (static_cast<uint32_t>(taskNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
    }

    uint32_t tailNum = totalTaskNum - taskNum;
    uint32_t taskNumPerScore = (totalTaskNum / blockDim / ALIGN_VALUE) * ALIGN_VALUE;
    uint32_t taskNumPerLcore = taskNumPerScore + ALIGN_VALUE;
    uint32_t numScore = (blockDim * taskNumPerLcore - totalTaskNum) / ALIGN_VALUE;
    uint32_t numLcore = blockDim - numScore;

    if (taskNumPerScore == 0) {
        blockDim = blockDim - numScore;
    }
    if (taskNumPerLcore == 0) {
        blockDim = blockDim - numLcore;
    }

    uint32_t degreeBufferNum = 1;
    if (degreeUsed == L2_SH_DEGREE) {
        degreeBufferNum = L2_SH_BUFFER_NUM;
    } else if (degreeUsed == L3_SH_DEGREE) {
        degreeBufferNum = L3_SH_BUFFER_NUM;
    } else if (degreeUsed == L4_SH_DEGREE) {
        degreeBufferNum = L4_SH_BUFFER_NUM;
    }
    uint32_t bufferSize = BN_NUM + degreeBufferNum + BC_NUM * degreeNum;
    uint32_t taskNumPerLoop = static_cast<int32_t>((ubTotalSize * UB_RATIO) / (bufferSize * FLOAT_SIZE));
    taskNumPerLoop = static_cast<int32_t>((taskNumPerLoop + ALIGN_VALUE - 1) / ALIGN_VALUE * ALIGN_VALUE);

    tiling.set_taskNum(taskNum);
    tiling.set_coeffNum(coeffNum);
    tiling.set_degreeUsed(degreeUsed);
    tiling.set_degreeNum(degreeNum);
    tiling.set_totalTaskNum(totalTaskNum);
    tiling.set_tailNum(tailNum);
    tiling.set_taskNumPerScore(taskNumPerScore);
    tiling.set_taskNumPerLcore(taskNumPerLcore);
    tiling.set_numScore(numScore);
    tiling.set_numLcore(numLcore);
    tiling.set_taskNumPerLoop(taskNumPerLoop);
    tiling.set_blockDim(blockDim);
    tiling.set_ubTotalSize(ubTotalSize);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
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

    const gert::Shape* dirsShape = context->GetInputShape(DIRS_PTR_INDEX);
    const gert::Shape* coeffsShape = context->GetInputShape(COEFFS_PTR_INDEX);
    if (dirsShape == nullptr || coeffsShape == nullptr) {
            return ge::GRAPH_FAILED;
        }

    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_PTR_INDEX);
    uint32_t taskNum = dirsShape->GetDim(TASK_NUM_INDEX);
    uint32_t coeffNum = coeffsShape->GetDim(COEFF_NUM_INDEX);
    *outputShape = {taskNum, 3};
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class SphericalHarmonicsForward : public OpDef {
public:
    explicit SphericalHarmonicsForward(const char* name) : OpDef(name)
    {
        this->Input("dirs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("coeffs")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("degrees_to_use").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingForSphericalHarmonicsForward);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SphericalHarmonicsForward);
}
