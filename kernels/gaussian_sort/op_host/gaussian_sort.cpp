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
 
/*!
 * \file gaussian_sort.cpp
 * \brief gaussian sort op host
 */

#include "gaussian_sort_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t MASK_PTR_INDEX = 0;
constexpr uint32_t DEPTHS_PTR_INDEX = 3;
constexpr uint32_t TILE_NUM_INDEX = 0;
constexpr uint32_t GAUSSIAN_NUM_INDEX = 1;
constexpr uint32_t SIZE_OF_FLOAT = 4;
constexpr uint32_t BLOCK_SIZE = 256;
constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_FLOAT;
constexpr uint32_t MASK_TENSOR_NUM = 6;
constexpr uint32_t SORT_TENSOR_NUM = 8;
constexpr uint32_t WS_TENSOR_NUM = 4;
}  // namespace

namespace optiling {
static ge::graphStatus TilingForGaussianSort(gert::TilingContext* context)
{
    GaussianSortTilingData tiling;
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto allInMaskTensorPtr = context->GetInputTensor(MASK_PTR_INDEX);
    auto depthsTensorPtr = context->GetInputTensor(DEPTHS_PTR_INDEX);
    if (allInMaskTensorPtr == nullptr || depthsTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto allInMaskShape = context->GetInputShape(MASK_PTR_INDEX);
    auto depthsShape = context->GetInputShape(DEPTHS_PTR_INDEX);
    if (allInMaskShape == nullptr || depthsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = context->GetPlatformInfo();
    if (platform == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platform);
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = platformInfo.GetCoreNumAiv();
    if (blockDim == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t tileNum = allInMaskShape->GetStorageShape().GetDim(TILE_NUM_INDEX);
    uint32_t nGauss = allInMaskShape->GetStorageShape().GetDim(GAUSSIAN_NUM_INDEX);
    // mask阶段，切分策略 - 按tileNum均匀分配到各vector
    // 确定参与计算的核数
    blockDim = (blockDim < tileNum) ? blockDim : tileNum;
    // 核间tileNum切分计算整核尾核
    uint32_t formerNum = tileNum % blockDim;
    // 尾核处理Tile的个数
    uint32_t tailTileNum = tileNum / blockDim;
    // 整核处理Tile的个数
    uint32_t formerTileNum = (formerNum == 0) ? tailTileNum : tailTileNum + 1;
    // 核内分批处理高斯球切分
    // 搬运补齐
    uint32_t ubMaxBlockNum = static_cast<uint32_t>(ubSize / MASK_TENSOR_NUM / BLOCK_SIZE);
    uint32_t ubMaxNum = ubMaxBlockNum * ALIGN_NUM;
    uint32_t nGaussAlign = (nGauss % ALIGN_NUM == 0) ? nGauss : ((nGauss + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t maskNumPerLoop = (nGaussAlign < ubMaxNum) ? nGaussAlign : ubMaxNum;
    uint32_t maskLoopNum = nGaussAlign / maskNumPerLoop;  // 循环次数
    uint32_t maskTailNum = maskNumPerLoop;                // 处理尾块
    if (nGaussAlign % maskNumPerLoop != 0) {
        maskLoopNum += 1;
        maskTailNum = nGaussAlign % maskNumPerLoop;
    }
    // 补齐元素个数
    uint32_t maskAlignedNum = nGaussAlign - nGauss;
    // sort阶段，核内大小数排序UB支持最大数计算
    uint32_t maxSortNum = ubSize / (SORT_TENSOR_NUM * SIZE_OF_FLOAT);

    tiling.set_nGauss(nGauss);
    tiling.set_formerNum(formerNum);
    tiling.set_formerTileNum(formerTileNum);
    tiling.set_tailTileNum(tailTileNum);
    tiling.set_maskLoopNum(maskLoopNum);
    tiling.set_maskNumPerLoop(maskNumPerLoop);
    tiling.set_maskTailNum(maskTailNum);
    tiling.set_maskAlignedNum(maskAlignedNum);
    tiling.set_maxSortNum(maxSortNum);

    tiling.set_ubSize(ubSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    // workspace 空间申请，动态shape，此处采用nGaussAlign，可优化为maxMaskNum
    size_t userWorkspaceSize = nGaussAlign * SIZE_OF_FLOAT * WS_TENSOR_NUM * blockDim;
    size_t systemWorkspaceSize = platformInfo.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = systemWorkspaceSize + userWorkspaceSize;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class GaussianSort : public OpDef {
public:
    explicit GaussianSort(const char* name) : OpDef(name)
    {
        this->Input("all_in_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tile_sums")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("tile_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("depths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("sorted_gs_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingForGaussianSort);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GaussianSort);
}  // namespace ops
