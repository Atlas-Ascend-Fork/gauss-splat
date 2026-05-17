/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
using namespace AscendC;
using namespace std;

constexpr uint32_t MEANS2D_TAIL_DIM = 2;
constexpr uint32_t CONICS_TAIL_DIM = 3;
constexpr uint32_t COVARS2D_TAIL_DIM = 3;
constexpr uint32_t TILE_GRID_TAIL_DIM = 2;
constexpr uint32_t INPUT_BUFFER_NUM = 11;
constexpr uint32_t COMPUTE_BUFFER_NUM = 6;
constexpr uint32_t COMPARE_BUFFER_NUM = 5;

constexpr uint32_t CONICS_OFFSET_2 = 2;
constexpr uint32_t COVARS2D_OFFSET_2 = 2;
constexpr uint32_t INPUT_OFFSET_2 = 2;
constexpr uint32_t INPUT_OFFSET_3 = 3;
constexpr uint32_t INPUT_OFFSET_4 = 4;
constexpr uint32_t INPUT_OFFSET_5 = 5;
constexpr uint32_t INPUT_OFFSET_6 = 6;
constexpr uint32_t INPUT_OFFSET_7 = 7;
constexpr uint32_t INPUT_OFFSET_8 = 8;
constexpr uint32_t INPUT_OFFSET_9 = 9;
constexpr uint32_t INPUT_OFFSET_10 = 10;

constexpr uint32_t COMPUTE_OFFSET_2 = 2;
constexpr uint32_t COMPUTE_OFFSET_3 = 3;
constexpr uint32_t COMPUTE_OFFSET_4 = 4;
constexpr uint32_t COMPUTE_OFFSET_5 = 5;
constexpr uint32_t COMPARE_OFFSET_2 = 2;
constexpr uint32_t COMPARE_OFFSET_3 = 3;
constexpr uint32_t COMPARE_OFFSET_4 = 4;

constexpr uint32_t GATHERMASK_OFFSET_2 = 2;
constexpr uint32_t GATHERMASK_OFFSET_3 = 3;
constexpr uint32_t GATHERMASK_OFFSET_4 = 4;
constexpr uint32_t GATHERMASK_OFFSET_5 = 5;
constexpr uint32_t GATHERMASK_OFFSET_6 = 6;
constexpr uint32_t GATHERMASK_OFFSET_7 = 7;
constexpr uint32_t GATHERMASK_OFFSET_8 = 8;
constexpr uint32_t GATHERMASK_OFFSET_9 = 9;
constexpr uint32_t GATHERMASK_OFFSET_10 = 10;
constexpr uint32_t GATHERMASK_OFFSET_11 = 11;
constexpr uint32_t GATHERMASK_OFFSET_12 = 12;
constexpr uint32_t GATHERMASK_OFFSET_13 = 13;
constexpr uint32_t GATHERMASK_OFFSET_14 = 14;
constexpr uint32_t GATHERMASK_OFFSET_15 = 15;
constexpr uint32_t GATHERMASK_OFFSET_16 = 16;
constexpr uint32_t GATHERMASK_OFFSET_17 = 17;
constexpr uint32_t GATHERMASK_OFFSET_18 = 18;
constexpr uint32_t GATHERMASK_OFFSET_19 = 19;

constexpr uint32_t ALIGN_VALUE = 64;
constexpr int32_t MAX_NUM_TILE_PER_CORE = 1024;
constexpr int32_t FLOAT_SIZE = 4;
constexpr uint32_t BLOCK_NUM = 48;
constexpr uint32_t GATHER_NUM = 20;

constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float TWO_FLOAT_VALUE = 2.0f;
constexpr float LN2 = 0.69314718055f;
constexpr float LN2_COEFF = 8.0f;
constexpr float DELTA_COEFF = 4.0f;

constexpr float ALPHA_FACTOR_VALUE = -0.5 * 0.3525f;
constexpr float LINEAR_APPROX_VALUE_A = 0.3525f;
constexpr float LINEAR_APPROX_VALUE_B = 0.7729f;

#define Ceil8(num) (static_cast<uint32_t>(((num) + 7) & ~uint64_t(7)))
#define Ceil64(num) (static_cast<uint32_t>(((num) + 63) & ~uint64_t(63)))

class FlashGaussianBuildMask {
public:
    __aicore__ inline FlashGaussianBuildMask() {}

    __aicore__ inline void GetTilingData(const FlashGaussianBuildMaskTilingData* tiling_data)
    {
        tileNumPerScore = tiling_data->tileNumPerScore;
        tileNumPerLcore = tiling_data->tileNumPerLcore;
        numScore = tiling_data->numScore;
        numLcore = tiling_data->numLcore;
        blockDim = tiling_data->blockDim;
        taskNumPerLoop = tiling_data->taskNumPerLoop;
        numTile = tiling_data->numTile;
        batchSize = tiling_data->batchSize;
        cameraNum = tiling_data->cameraNum;
        gaussNum = tiling_data->gaussNum;
        tileSize = tiling_data->tileSize;
        imageWidth = tiling_data->imageWidth;
        imageHeight = tiling_data->imageHeight;
        ubTotalSize = tiling_data->ubTotalSize;
    }

    __aicore__ inline void PreInit(const FlashGaussianBuildMaskTilingData* tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        GetTilingData(tiling_data);

        this->blockIndex = GetBlockIdx();
        if (this->blockIndex < numLcore) {
            tileNumPerCore = tileNumPerLcore;
            tileStartIndex = this->blockIndex * tileNumPerCore;
        } else {
            tileNumPerCore = tileNumPerScore;
            tileStartIndex = numLcore * tileNumPerLcore + (this->blockIndex - numLcore) * tileNumPerCore;
        }
        gatherBlockStartAddr = this->blockIndex * gaussNum * GATHER_NUM;
        gatherStartIndex = 0;
    }

    __aicore__ inline void GlobalBufferInit(GM_ADDR means2d, GM_ADDR opacity, GM_ADDR conics, GM_ADDR covars2d,
                                            GM_ADDR depths, GM_ADDR cnt, GM_ADDR tile_grid, GM_ADDR gathermask,
                                            GM_ADDR tile_sum, GM_ADDR tile_depths, GM_ADDR gauss_index)
    {
        uint64_t baseGaussSize = batchSize * cameraNum * gaussNum;
        means2dGM.SetGlobalBuffer((__gm__ DTYPE_MEANS2D*)means2d, baseGaussSize * MEANS2D_TAIL_DIM);
        opacityGM.SetGlobalBuffer((__gm__ DTYPE_OPACITY*)opacity, baseGaussSize);
        conicsGM.SetGlobalBuffer((__gm__ DTYPE_CONICS*)conics, baseGaussSize * CONICS_TAIL_DIM);
        covars2dGM.SetGlobalBuffer((__gm__ DTYPE_COVARS2D*)covars2d, baseGaussSize * COVARS2D_TAIL_DIM);
        cntGM.SetGlobalBuffer((__gm__ DTYPE_CNT*)cnt, batchSize * cameraNum);
        tilegridGM.SetGlobalBuffer((__gm__ DTYPE_TILE_GRID*)tile_grid, numTile * TILE_GRID_TAIL_DIM);
        depthsGM.SetGlobalBuffer((__gm__ DTYPE_DEPTHS*)depths, baseGaussSize);
        gathermaskGM.SetGlobalBuffer((__gm__ DTYPE_GATHER_MASK*)gathermask, BLOCK_NUM * GATHER_NUM * gaussNum);
        gaussindexGM.SetGlobalBuffer((__gm__ DTYPE_GAUSS_INDEX*)gauss_index, baseGaussSize * numTile);
        tiledepthsGM.SetGlobalBuffer((__gm__ DTYPE_TILE_DEPTHS*)tile_depths, baseGaussSize * numTile);
        tilesumGM.SetGlobalBuffer((__gm__ DTYPE_TILE_SUM*)tile_sum, batchSize * cameraNum * numTile);
        // 两次gather结果存放：
        //  第一次：means2d + opactiy + conics + rect + depths + gatherIndex = 12
        //  第二次：means2d + opactiy + conics + depths + gatherIndex = 8
    }

    __aicore__ inline void LocalBufferInit(TPipe* pipe)
    {
        uint64_t baseBufferSize = taskNumPerLoop * FLOAT_SIZE;
        this->_pipe = pipe;
        this->_pipe->InitBuffer(InputTensorBuffer, baseBufferSize * INPUT_BUFFER_NUM);
        this->_pipe->InitBuffer(MaskTensorBuffer, baseBufferSize);
        this->_pipe->InitBuffer(ComputingTensorBuffer, baseBufferSize * COMPUTE_BUFFER_NUM);
        this->_pipe->InitBuffer(CompareTensorBuffer, taskNumPerLoop * COMPARE_BUFFER_NUM);
        this->_pipe->InitBuffer(GatherIndexTensorBuffer, baseBufferSize);
        this->_pipe->InitBuffer(TileSumTensorBuffer, tileNumPerCore);

        InputTensor = InputTensorBuffer.Get<DTYPE_MEANS2D>();
        MaskTensor = MaskTensorBuffer.Get<float>();
        CompareTensor = CompareTensorBuffer.Get<uint8_t>();
        ComputingTensor = ComputingTensorBuffer.Get<float>();
        GatherIndexTensor = GatherIndexTensorBuffer.Get<float>();
        TileSumTensor = TileSumTensorBuffer.Get<int32_t>();
        Duplicate(GatherIndexTensor, ZERO_FLOAT_VALUE, taskNumPerLoop);
    }

    __aicore__ inline void Init(GM_ADDR means2d, GM_ADDR opacity, GM_ADDR conics, GM_ADDR covars2d, GM_ADDR depths,
                                GM_ADDR cnt, GM_ADDR tile_grid, GM_ADDR gathermask, GM_ADDR tile_sum,
                                GM_ADDR tile_depths, GM_ADDR gauss_index,
                                const FlashGaussianBuildMaskTilingData* tiling_data, TPipe* pipe)
    {
        PreInit(tiling_data);
        GlobalBufferInit(means2d, opacity, conics, covars2d, depths, cnt, tile_grid, gathermask, tile_sum, tile_depths,
                         gauss_index);
        LocalBufferInit(pipe);
    }

    __aicore__ inline void Process()
    {
        float tileGridXArr[MAX_NUM_TILE_PER_CORE];
        float tileGridYArr[MAX_NUM_TILE_PER_CORE];
        for (uint32_t tileLoopIndex = 0; tileLoopIndex < tileNumPerCore; tileLoopIndex++) {
            uint64_t tileCopyinIndex = (tileStartIndex + tileLoopIndex) * TILE_GRID_TAIL_DIM;
            tileGridYArr[tileLoopIndex] = tilegridGM.GetValue(tileCopyinIndex);
            tileGridXArr[tileLoopIndex] = tilegridGM.GetValue(tileCopyinIndex + 1);
        }
        float minTileGridY = tileGridYArr[0];
        float maxTileGridY = tileGridYArr[tileNumPerCore - 1] + tileSize;

        for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (uint32_t cameraIdx = 0; cameraIdx < cameraNum; cameraIdx++) {
                cntGaussNum = cntGM.GetValue(batchIdx * cameraNum + cameraIdx);
                ComputingCurGaussNum(static_cast<uint32_t>(cntGaussNum));
                AxisYFilter(batchIdx, cameraIdx, minTileGridY, maxTileGridY);
                ComputingForSingleTile(batchIdx, cameraIdx, tileGridXArr, tileGridYArr);
            }
        }
    }

    __aicore__ inline uint64_t GaussIndexGather(uint32_t taskLoopIndex, uint32_t stageId)
    {
        int32_t vecStartIndex = taskLoopIndex * taskNumPerLoop;
        uint64_t rsvdGSCnt = 0;
        uint64_t tailCount = 0;

        if (taskLoopIndex == taskLoop - 1) {
            tailCount = taskNumPerCurLoop - tailNum;
        } else {
            tailCount = taskNumPerCurLoop;
        }

        GatherMaskParams gatherMaskParams = {1, 1, 32, 1};
        if (stageId == 0) {
            CreateVecIndex(MaskTensor, static_cast<float>(vecStartIndex), taskNumPerCurLoop);
            GatherMask(GatherIndexTensor, MaskTensor, CompareTensor.ReinterpretCast<uint32_t>(), true, tailCount,
                       gatherMaskParams, rsvdGSCnt);
        } else {
            GatherMask(MaskTensor, GatherIndexTensor, CompareTensor.ReinterpretCast<uint32_t>(), true, tailCount,
                       gatherMaskParams, rsvdGSCnt);
        }
        return rsvdGSCnt;
    }

    __aicore__ inline void GaussIndexCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(
                GatherIndexTensor,
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_11 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        } else {
            DataCopy(
                GatherIndexTensor,
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_19 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        }
    }

    __aicore__ inline void GaussIndexCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_11 + gatherStartIndex],
                     GatherIndexTensor, copyLength);
        } else {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_19 + gatherStartIndex],
                     GatherIndexTensor, copyLength);
        }
    }

    __aicore__ inline void Means2dGather(GatherMaskParams gatherMaskParams, uint64_t rsvdGSCnt)
    {
        GatherMask(ComputingTensor, InputTensor, CompareTensor.ReinterpretCast<uint32_t>(), true, taskNumPerCurLoop,
                   gatherMaskParams, rsvdGSCnt);
        GatherMask(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop],
                   CompareTensor.ReinterpretCast<uint32_t>(), true, taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
    }

    __aicore__ inline void Means2dCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(InputTensor, gathermaskGM[gatherBlockStartAddr + taskLoopIndex * taskNumPerLoop], copyLength);
            DataCopy(InputTensor[taskNumPerCurLoop],
                     gathermaskGM[gatherBlockStartAddr + gaussNum + taskLoopIndex * taskNumPerLoop], copyLength);
        } else {
            DataCopy(
                InputTensor,
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_12 + taskLoopIndex * taskNumPerLoop],
                copyLength);
            DataCopy(
                InputTensor[taskNumPerCurLoop],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_13 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        }
    }

    __aicore__ inline void Means2dCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex], ComputingTensor, copyLength);
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum],
                     ComputingTensor[taskNumPerCurLoop], copyLength);
        } else {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_12],
                     ComputingTensor, copyLength);
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_13],
                     ComputingTensor[taskNumPerCurLoop], copyLength);
        }
    }

    __aicore__ inline void OpacityGather(GatherMaskParams gatherMaskParams, uint64_t rsvdGSCnt)
    {
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
    }

    __aicore__ inline void OpacityCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_2 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        } else {
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_14 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        }
    }

    __aicore__ inline void OpacityCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_2],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], copyLength);
        } else {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_14],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], copyLength);
        }
    }

    __aicore__ inline void ConicsGather(GatherMaskParams gatherMaskParams, uint64_t rsvdGSCnt)
    {
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
    }

    __aicore__ inline void ConicsCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_3 + taskLoopIndex * taskNumPerLoop],
                copyLength);
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_4 + taskLoopIndex * taskNumPerLoop],
                copyLength);
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_5 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        } else {
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_15 + taskLoopIndex * taskNumPerLoop],
                copyLength);
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_16 + taskLoopIndex * taskNumPerLoop],
                copyLength);
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_17 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        }
    }

    __aicore__ inline void ConicsCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_3],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], copyLength);
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_4],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], copyLength);
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_5],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], copyLength);
        } else {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_15],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], copyLength);
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_16],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], copyLength);
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_17],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], copyLength);
        }
    }

    __aicore__ inline void RectGather(GatherMaskParams gatherMaskParams, uint64_t rsvdGSCnt)
    {
        GatherMask(ComputingTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6],
                   CompareTensor.ReinterpretCast<uint32_t>(), true, taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
        GatherMask(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7],
                   CompareTensor.ReinterpretCast<uint32_t>(), true, taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
    }

    __aicore__ inline void RectCopyIn(uint64_t taskLoopIndex, uint64_t copyLength)
    {
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6],
                 gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_6 + taskLoopIndex * taskNumPerLoop],
                 copyLength);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7],
                 gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_7 + taskLoopIndex * taskNumPerLoop],
                 copyLength);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8],
                 gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_8 + taskLoopIndex * taskNumPerLoop],
                 copyLength);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9],
                 gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_9 + taskLoopIndex * taskNumPerLoop],
                 copyLength);
    }

    __aicore__ inline void RectCopyOut(uint64_t copyLength)
    {
        DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_6],
                 ComputingTensor, copyLength);
        DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_7],
                 ComputingTensor[taskNumPerCurLoop], copyLength);
        DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_8],
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], copyLength);
        DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_9],
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], copyLength);
    }

    __aicore__ inline void DepthGather(GatherMaskParams gatherMaskParams, uint64_t rsvdGSCnt)
    {
        GatherMask(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4],
                   InputTensor[taskNumPerCurLoop * INPUT_OFFSET_10], CompareTensor.ReinterpretCast<uint32_t>(), true,
                   taskNumPerCurLoop, gatherMaskParams, rsvdGSCnt);
    }

    __aicore__ inline void DepthCopyIn(uint64_t taskLoopIndex, uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_10],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_10 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        } else {
            DataCopy(
                InputTensor[taskNumPerCurLoop * INPUT_OFFSET_10],
                gathermaskGM[gatherBlockStartAddr + gaussNum * GATHERMASK_OFFSET_18 + taskLoopIndex * taskNumPerLoop],
                copyLength);
        }
    }

    __aicore__ inline void DepthCopyOut(uint64_t copyLength, uint32_t stageId)
    {
        if (stageId == 0) {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_10],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], copyLength);
        } else {
            DataCopy(gathermaskGM[gatherBlockStartAddr + gatherStartIndex + gaussNum * GATHERMASK_OFFSET_18],
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], copyLength);
        }
    }

    __aicore__ inline void AxisYFilter(uint32_t batchIdx, uint32_t cameraIdx, float minTileGridY, float maxTileGridY)
    {
        for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
            GetComputeLength(taskLoopIndex);
            CopyInForBound(batchIdx, cameraIdx, taskLoopIndex);
            GetRect(taskLoopIndex);
            CopyInForFlashGS(batchIdx, cameraIdx, taskLoopIndex);
            FilterByGridY(minTileGridY, maxTileGridY);
            Copyout2yFilterWS(batchIdx, cameraIdx, taskLoopIndex);
        }
        yFilterGaussNum = gatherStartIndex;
        ComputingCurGaussNum(yFilterGaussNum);
        gatherStartIndex = 0;
    }

    __aicore__ inline void FilterByGridY(float minTileGridY, float maxTileGridY)
    {
        // rectmaxy > minValue & rectminy < maxValue
        CompareScalar(CompareTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], maxTileGridY, CMPMODE::LT,
                      taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], minTileGridY,
                      CMPMODE::GT, taskNumPerCurLoop);
        And(CompareTensor, CompareTensor, CompareTensor[taskNumPerCurLoop], taskNumPerCurLoop);
    }

    __aicore__ inline void Copyout2yFilterWS(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t rsvdGSCnt = 0;
        uint64_t realGSCnt = GaussIndexGather(taskLoopIndex, 0);
        uint64_t copyLength = Ceil8(realGSCnt);
        GatherMaskParams gatherMaskParams = {1, 1, 32, 1};
        // copyout: means2d, opacity
        Means2dGather(gatherMaskParams, rsvdGSCnt);
        OpacityGather(gatherMaskParams, rsvdGSCnt);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        ConicsGather(gatherMaskParams, rsvdGSCnt);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        Means2dCopyOut(copyLength, 0);
        OpacityCopyOut(copyLength, 0);
        ConicsCopyOut(copyLength, 0);
        GaussIndexCopyOut(copyLength, 0);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        // copyout: rect, depths
        RectGather(gatherMaskParams, rsvdGSCnt);
        DepthGather(gatherMaskParams, rsvdGSCnt);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        RectCopyOut(copyLength);
        DepthCopyOut(copyLength, 0);

        gatherStartIndex += realGSCnt;
    }

    __aicore__ inline void ComputingCurGaussNum(uint32_t totalGaussNum)
    {
        uint32_t alignedGaussNum;
        if ((totalGaussNum % ALIGN_VALUE) != 0) {
            alignedGaussNum = (static_cast<uint32_t>(totalGaussNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
        } else {
            alignedGaussNum = totalGaussNum;
        }
        tailNum = alignedGaussNum - totalGaussNum;
        taskLoop = static_cast<int32_t>((totalGaussNum + taskNumPerLoop - 1) / taskNumPerLoop);
    }

    __aicore__ inline void ComputingForSingleTile(uint32_t batchIdx, uint32_t cameraIdx, float* tileGridXArr,
                                                  float* tileGridYArr)
    {
        for (uint32_t tileLoopIndex = 0; tileLoopIndex < tileNumPerCore; tileLoopIndex++) {
            cntGaussNum = yFilterGaussNum;
            ComputingCurGaussNum(cntGaussNum);
            for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                GetComputeLength(taskLoopIndex);
                CopyInTileGrid(tileLoopIndex, tileGridXArr, tileGridYArr);
                CopyInFromYFilter(taskLoopIndex);
                ComputingBound();
                BoundFilter(taskLoopIndex);
            }
            cntGaussNum = gatherStartIndex;
            ComputingCurGaussNum(cntGaussNum);
            gatherStartIndex = 0;
            for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                GetComputeLength(taskLoopIndex);
                CopyInTileGrid(tileLoopIndex, tileGridXArr, tileGridYArr);
                CopyInFromBoundFilter(taskLoopIndex);
                BlockContainsCenter();
                BlockIntersectEllipse();
                FlagFilter(batchIdx, cameraIdx, tileLoopIndex, taskLoopIndex);
            }
            TileSumTensor.SetValue(tileLoopIndex, gatherStartIndex);
            cntGaussNum = gatherStartIndex;
            ComputingCurGaussNum(cntGaussNum);
            gatherStartIndex = 0;
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        uint64_t tilesumIndex = batchIdx * cameraNum * numTile + cameraIdx * numTile + tileStartIndex;
        DataCopyParams dataCopyParams{1, static_cast<uint16_t>(tileNumPerCore * sizeof(int32_t)), 0, 0};
        DataCopyPad(tilesumGM[tilesumIndex], TileSumTensor, dataCopyParams);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void GetComputeLength(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == taskLoop - 1) {
            taskNumPerCurLoop = (cntGaussNum + tailNum) - taskLoopIndex * taskNumPerLoop;
        } else {
            taskNumPerCurLoop = taskNumPerLoop;
        }
    }

    __aicore__ inline void CopyInFromBoundFilter(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == 0) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        Means2dCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
        Duplicate(MaskTensor, ONE_FLOAT_VALUE, taskNumPerCurLoop);
        if (taskLoopIndex == taskLoop - 1) {
            for (uint32_t id = (taskNumPerCurLoop - tailNum); id < taskNumPerCurLoop; id++) {
                MaskTensor.SetValue(id, ZERO_FLOAT_VALUE);
            }
        }
        CompareScalar(CompareTensor, MaskTensor, ZERO_FLOAT_VALUE, CMPMODE::GT, taskNumPerCurLoop);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        OpacityCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
        ConicsCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        GaussIndexCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
        DepthCopyIn(taskLoopIndex, taskNumPerCurLoop, 1);
    }

    __aicore__ inline void CopyInFromYFilter(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == 0) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        RectCopyIn(taskLoopIndex, taskNumPerCurLoop);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (taskLoopIndex > 0) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        }
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        Means2dCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
        OpacityCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
        ConicsCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
        GaussIndexCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
        DepthCopyIn(taskLoopIndex, taskNumPerCurLoop, 0);
    }

    __aicore__ inline void CopyInForBound(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t baseCopyinIndex = batchIdx * cameraNum * gaussNum + cameraIdx * gaussNum;
        uint64_t taskCopyinIndex = taskLoopIndex * taskNumPerLoop;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], opacityGM[baseCopyinIndex + taskCopyinIndex],
                 taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
                 covars2dGM[baseCopyinIndex * COVARS2D_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
                 covars2dGM[baseCopyinIndex * COVARS2D_TAIL_DIM + taskCopyinIndex + gaussNum * COVARS2D_OFFSET_2],
                 taskNumPerCurLoop);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        DataCopy(InputTensor, means2dGM[baseCopyinIndex * MEANS2D_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop],
                 means2dGM[baseCopyinIndex * MEANS2D_TAIL_DIM + taskCopyinIndex + gaussNum], taskNumPerCurLoop);
    }

    __aicore__ inline void CopyInForFlashGS(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t baseCopyinIndex = batchIdx * cameraNum * gaussNum + cameraIdx * gaussNum;
        uint64_t taskCopyinIndex = taskLoopIndex * taskNumPerLoop;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
                 conicsGM[baseCopyinIndex * CONICS_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
                 conicsGM[baseCopyinIndex * CONICS_TAIL_DIM + taskCopyinIndex + gaussNum], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5],
                 conicsGM[baseCopyinIndex * CONICS_TAIL_DIM + taskCopyinIndex + gaussNum * CONICS_OFFSET_2],
                 taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_10], depthsGM[baseCopyinIndex + taskCopyinIndex],
                 taskNumPerCurLoop);
    }

    __aicore__ inline void CopyInTileGrid(uint32_t tileLoopIndex, float* tileGridXArr, float* tileGridYArr)
    {
        tileGridX = tileGridXArr[tileLoopIndex];
        tileGridY = tileGridYArr[tileLoopIndex];
    }

    __aicore__ inline void GetRect(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex > 0) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }

        // power = LN2 * 8 + LN2 * torch.log2(opacity) -> 0
        Log2(ComputingTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);
        Muls(ComputingTensor, ComputingTensor, LN2, taskNumPerCurLoop);
        Adds(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], ComputingTensor, LN2_COEFF * LN2, taskNumPerCurLoop);
        Muls(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2],
             TWO_FLOAT_VALUE, taskNumPerCurLoop);

        // w = (torch.sqrt(2*cov00[:, None]*power)+1).floor().squeeze()
        // h = (torch.sqrt(2*cov11[:, None]*power)+1).floor().squeeze()
        Mul(ComputingTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);

        Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor, taskNumPerCurLoop);
        Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor[taskNumPerCurLoop],
             taskNumPerCurLoop);

        Adds(ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ONE_FLOAT_VALUE,
             taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ONE_FLOAT_VALUE,
             taskNumPerCurLoop);

        Floor(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], ComputingTensor, taskNumPerCurLoop);
        Floor(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // rect_min_w = torch.clamp(means_x - w, 0, width - 1.0), rect_min_h = torch.clamp(means_y - h, 0, height - 1.0)
        Sub(ComputingTensor[taskNumPerCurLoop], InputTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
            taskNumPerCurLoop);
        Sub(ComputingTensor, InputTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
            taskNumPerCurLoop);

        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor[taskNumPerCurLoop],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor, ZERO_FLOAT_VALUE,
                 taskNumPerCurLoop);

        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
                 imageWidth - ONE_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
                 imageHeight - ONE_FLOAT_VALUE, taskNumPerCurLoop);

        // rect_max_w = torch.clamp(means_x + w, 0, width - 1.0)
        // rect_max_h = torch.clamp(means_y + h, 0, height - 1.0)
        Add(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], InputTensor,
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], taskNumPerCurLoop);
        Add(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], InputTensor[taskNumPerCurLoop],
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], taskNumPerCurLoop);

        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
                 ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
                 imageWidth - ONE_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
                 imageHeight - ONE_FLOAT_VALUE, taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingBound()
    {
        CompareScalar(CompareTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6], tileGridX, CMPMODE::GT,
                      taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], tileGridY,
                      CMPMODE::GT, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
                      InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], tileGridX + tileSize, CMPMODE::LT,
                      taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
                      InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], tileGridY + tileSize, CMPMODE::LT,
                      taskNumPerCurLoop);

        Select(MaskTensor, CompareTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6], tileGridX,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, taskNumPerCurLoop);
        Select(ComputingTensor, CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
               InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], tileGridX + tileSize, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], CompareTensor[taskNumPerCurLoop],
               InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], tileGridY, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
               InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], tileGridY + tileSize, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               taskNumPerCurLoop);

        Compare(CompareTensor[taskNumPerCurLoop], ComputingTensor, MaskTensor, CMPMODE::GT, taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor[taskNumPerCurLoop],
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], CMPMODE::GT, taskNumPerCurLoop);
        And(CompareTensor, CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
            taskNumPerCurLoop);  // all_in_mask
    }

    __aicore__ inline void BoundFilter(uint32_t taskLoopIndex)
    {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        uint64_t rsvdGSCnt = 0;
        uint64_t realGSCnt = GaussIndexGather(taskLoopIndex, 1);
        uint64_t copyLength = Ceil8(realGSCnt);
        GatherMaskParams gatherMaskParams = {1, 1, 32, 1};
        // copyout: means2d, opacity
        Adds(GatherIndexTensor, MaskTensor, ZERO_FLOAT_VALUE, copyLength);
        DepthGather(gatherMaskParams, rsvdGSCnt);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        DepthCopyOut(copyLength, 1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        Means2dGather(gatherMaskParams, rsvdGSCnt);
        OpacityGather(gatherMaskParams, rsvdGSCnt);
        ConicsGather(gatherMaskParams, rsvdGSCnt);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

        Means2dCopyOut(copyLength, 1);
        OpacityCopyOut(copyLength, 1);
        ConicsCopyOut(copyLength, 1);
        GaussIndexCopyOut(copyLength, 1);

        gatherStartIndex += realGSCnt;
    }

    __aicore__ inline void FlagFilter(uint32_t batchIdx, uint32_t cameraIdx, uint32_t tileLoopIndex,
                                      uint32_t taskLoopIndex)
    {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

        uint64_t rsvdGSCnt = 0;
        uint64_t realGSCnt = GaussIndexGather(taskLoopIndex, 1);
        uint64_t copyLength = Ceil8(realGSCnt);
        GatherMaskParams gatherMaskParams = {1, 1, 32, 1};
        DepthGather(gatherMaskParams, rsvdGSCnt);
        Adds(GatherIndexTensor, MaskTensor, ZERO_FLOAT_VALUE, copyLength);

        uint64_t batchCopyoutIndex = batchIdx * cameraNum * gaussNum * numTile;
        uint64_t cameraCopyoutIndex = cameraIdx * gaussNum * numTile;
        uint64_t tileCopyoutIndex = (tileStartIndex + tileLoopIndex) * gaussNum;

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        DataCopy(gaussindexGM[batchCopyoutIndex + cameraCopyoutIndex + tileCopyoutIndex + gatherStartIndex],
                 GatherIndexTensor, copyLength);
        DataCopy(tiledepthsGM[batchCopyoutIndex + cameraCopyoutIndex + tileCopyoutIndex + gatherStartIndex],
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], copyLength);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        gatherStartIndex += realGSCnt;
    }

    __aicore__ inline void BlockContainsCenter()
    {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        CompareScalar(CompareTensor[taskNumPerCurLoop], InputTensor, tileGridX, CMPMODE::GT, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], InputTensor[taskNumPerCurLoop], tileGridY,
                      CMPMODE::GT, taskNumPerCurLoop);

        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], InputTensor,
                      (tileGridX + tileSize - ONE_FLOAT_VALUE), CMPMODE::LE, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], InputTensor[taskNumPerCurLoop],
                      (tileGridY + tileSize - ONE_FLOAT_VALUE), CMPMODE::LE, taskNumPerCurLoop);

        And(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
            taskNumPerCurLoop);  // center_flag
    }

    __aicore__ inline void BlockIntersectEllipse()
    {
        // pix2x = pix_min_x + pix_max_x
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], InputTensor,
                      float(0.5f) * (tileGridX + tileGridX + tileSize - ONE_FLOAT_VALUE), CMPMODE::LT,
                      taskNumPerCurLoop);
        // x_pix_min = meansx_broad - pix_min_x
        Adds(ComputingTensor, InputTensor, float(-1.0f) * tileGridX, taskNumPerCurLoop);
        // x_pix_max = meansx_broad - pix_max_x
        Adds(ComputingTensor[taskNumPerCurLoop], InputTensor, float(-1.0f) * (tileGridX + tileSize - ONE_FLOAT_VALUE),
             taskNumPerCurLoop);
        // dx = torch.where(compation1, x_pix_min, x_pix_max)
        Select(MaskTensor, CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor,
               ComputingTensor[taskNumPerCurLoop], SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // b = -2 * conic_01[:, None] * dx
        Muls(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
             TWO_FLOAT_VALUE * float(-1.0f), taskNumPerCurLoop);
        Mul(ComputingTensor, ComputingTensor[taskNumPerCurLoop], MaskTensor, taskNumPerCurLoop);

        // c = conic_00[:, None] * dx * dx - w
        Mul(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], MaskTensor,
            taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor[taskNumPerCurLoop], MaskTensor,
            taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);

        // delta = b * b - 4 * a * c
        Mul(MaskTensor, ComputingTensor, ComputingTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5],
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], DELTA_COEFF, taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
            taskNumPerCurLoop);

        // t1 = (pix_min - mean[:, None]) * (2 * a) + b = b1 - y_pix_min * a1_double
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor[taskNumPerCurLoop],
             float(-1.0f) * tileGridY, taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5],
            taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor,
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // t2 = (pix_max - mean[:, None]) * (2 * a) + b
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor[taskNumPerCurLoop],
             float(-1.0f) * (tileGridY + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        Muls(MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor,
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], taskNumPerCurLoop);
        Sub(MaskTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // delta1_compare = delta1 >= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor[taskNumPerCurLoop],
                      ZERO_FLOAT_VALUE, CMPMODE::GE, taskNumPerCurLoop);
        // t11_compare = t11 <= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
                      ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ZERO_FLOAT_VALUE, CMPMODE::LE,
                      taskNumPerCurLoop);
        // t11_delta1_compare = t11 * t11 <= delta1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4],
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor[taskNumPerCurLoop], CMPMODE::LE,
                taskNumPerCurLoop);
        // t11_compare_total = t11_compare | t11_delta1_compare
        Or(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], taskNumPerCurLoop);

        // t21_compare = t21 >= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], MaskTensor, ZERO_FLOAT_VALUE, CMPMODE::GE,
                      taskNumPerCurLoop);
        // t21_delta1_compare = t21 * t21 <= delta1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor, MaskTensor, taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4],
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor[taskNumPerCurLoop], CMPMODE::LE,
                taskNumPerCurLoop);
        // t21_compare_total = t21_compare | t21_delta1_compare
        Or(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        // flag1
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], taskNumPerCurLoop);
        // center_flag | flag1
        Or(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop],
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop);

        // pix2x = pix_min_x + pix_max_x
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], InputTensor[taskNumPerCurLoop],
                      float(0.5f) * (tileGridY + tileGridY + tileSize - ONE_FLOAT_VALUE), CMPMODE::LT,
                      taskNumPerCurLoop);
        // x_pix_min = meansx_broad - pix_min_x
        Adds(ComputingTensor, InputTensor[taskNumPerCurLoop], float(-1.0f) * tileGridY, taskNumPerCurLoop);
        // x_pix_max = meansx_broad - pix_max_x
        Adds(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop],
             float(-1.0f) * (tileGridY + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        // dx = torch.where(compation1, x_pix_min, x_pix_max)
        Select(MaskTensor, CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor,
               ComputingTensor[taskNumPerCurLoop], SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);

        // b = -2 * conic_01[:, None] * dx
        Muls(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4],
             TWO_FLOAT_VALUE * float(-1.0f), taskNumPerCurLoop);
        Mul(ComputingTensor, ComputingTensor[taskNumPerCurLoop], MaskTensor, taskNumPerCurLoop);

        // c = conic_00[:, None] * dx * dx - w
        Mul(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], MaskTensor,
            taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor[taskNumPerCurLoop], MaskTensor,
            taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);

        // delta = b * b - 4 * a * c
        Mul(MaskTensor, ComputingTensor, ComputingTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], DELTA_COEFF, taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
            taskNumPerCurLoop);

        // t1 = (pix_min - mean[:, None]) * (2 * a) + b = b1 - y_pix_min * a1_double
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor, float(-1.0f) * tileGridX,
             taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3],
            taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor,
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // t2 = (pix_max - mean[:, None]) * (2 * a) + b
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor,
             float(-1.0f) * (tileGridX + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        Muls(MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor,
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], taskNumPerCurLoop);
        Sub(MaskTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // delta1_compare = delta1 >= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor[taskNumPerCurLoop],
                      ZERO_FLOAT_VALUE, CMPMODE::GE, taskNumPerCurLoop);
        // t11_compare = t11 <= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
                      ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ZERO_FLOAT_VALUE, CMPMODE::LE,
                      taskNumPerCurLoop);
        // t11_delta1_compare = t11 * t11 <= delta1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3],
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2],
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4],
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor[taskNumPerCurLoop], CMPMODE::LE,
                taskNumPerCurLoop);
        // t11_compare_total = t11_compare | t11_delta1_compare
        Or(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], taskNumPerCurLoop);

        // t21_compare = t21 >= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], MaskTensor, ZERO_FLOAT_VALUE, CMPMODE::GE,
                      taskNumPerCurLoop);
        // t21_delta1_compare = t21 * t21 <= delta1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor, MaskTensor, taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4],
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor[taskNumPerCurLoop], CMPMODE::LE,
                taskNumPerCurLoop);
        // t21_compare_total = t21_compare | t21_delta1_compare
        Or(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3],
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        // flag1
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2],
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], taskNumPerCurLoop);
        // center_flag | flag1
        Or(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop],
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop);

        // all_in_mask = all_in_mask & (center_flag | ellipse_isect_flag)
        And(CompareTensor, CompareTensor, CompareTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Duplicate(ComputingTensor, ONE_FLOAT_VALUE, taskNumPerCurLoop);
        Select(MaskTensor, CompareTensor, ComputingTensor, ZERO_FLOAT_VALUE, SELMODE::VSEL_TENSOR_SCALAR_MODE,
               taskNumPerCurLoop);
    }

private:
    TPipe* _pipe;
    TBuf<TPosition::VECCALC> InputTensorBuffer, MaskTensorBuffer;
    TBuf<TPosition::VECCALC> ComputingTensorBuffer, CompareTensorBuffer, GatherIndexTensorBuffer, TileSumTensorBuffer;
    LocalTensor<float> InputTensor, MaskTensor, ComputingTensor, GatherIndexTensor;
    LocalTensor<uint8_t> CompareTensor;
    LocalTensor<int32_t> TileSumTensor;

    GlobalTensor<DTYPE_MEANS2D> means2dGM;
    GlobalTensor<DTYPE_CNT> cntGM;
    GlobalTensor<DTYPE_OPACITY> opacityGM;
    GlobalTensor<DTYPE_CONICS> conicsGM;
    GlobalTensor<DTYPE_COVARS2D> covars2dGM;
    GlobalTensor<DTYPE_TILE_GRID> tilegridGM;
    GlobalTensor<DTYPE_DEPTHS> depthsGM;
    GlobalTensor<DTYPE_TILE_SUM> tilesumGM;
    GlobalTensor<DTYPE_TILE_DEPTHS> tiledepthsGM;
    GlobalTensor<DTYPE_GATHER_MASK> gathermaskGM;
    GlobalTensor<DTYPE_GAUSS_INDEX> gaussindexGM;

    float imageWidth, imageHeight, tileSize;
    float tileGridX, tileGridY;
    uint32_t batchSize, cameraNum, gaussNum, cntGaussNum, yFilterGaussNum, tailNum;
    uint32_t tileNumPerCore, tileNumPerScore, tileNumPerLcore, numScore, numLcore, blockDim;
    uint32_t taskNumPerLoop, taskNumPerCurLoop, tileStartIndex, taskLoop, numTile, asyncTaskNum;
    uint64_t blockIndex, ubTotalSize, gatherStartIndex, gatherBlockStartAddr;
};

extern "C" __global__ __aicore__ void flash_gaussian_build_mask(GM_ADDR means2d, GM_ADDR opacity, GM_ADDR conics,
                                                                GM_ADDR covars2d, GM_ADDR depths, GM_ADDR cnt,
                                                                GM_ADDR tile_grid, GM_ADDR gathermask, GM_ADDR tile_sum,
                                                                GM_ADDR tile_depths, GM_ADDR gauss_index,
                                                                GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        FlashGaussianBuildMask op;
        op.Init(means2d, opacity, conics, covars2d, depths, cnt, tile_grid, gathermask, tile_sum, tile_depths,
                gauss_index, &tiling_data, &pipe);
        op.Process();
    }
}