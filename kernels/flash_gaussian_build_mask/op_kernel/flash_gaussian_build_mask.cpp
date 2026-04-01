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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
using namespace AscendC;
using namespace std;

constexpr uint32_t MEANS2D_TAIL_DIM = 2;
constexpr uint32_t CONICS_TAIL_DIM = 3;
constexpr uint32_t COVARS2D_TAIL_DIM = 3;
constexpr uint32_t TILE_GRID_TAIL_DIM = 2;
constexpr uint32_t INPUT_BUFFER_NUM = 10;
constexpr uint32_t COMPUTE_BUFFER_NUM = 4;
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

constexpr uint32_t COMPUTE_OFFSET_2 = 2;
constexpr uint32_t COMPUTE_OFFSET_3 = 3;
constexpr uint32_t COMPARE_OFFSET_2 = 2;
constexpr uint32_t COMPARE_OFFSET_3 = 3;
constexpr uint32_t COMPARE_OFFSET_4 = 4;

constexpr uint32_t ALIGN_VALUE = 64;
constexpr int32_t MAX_NUM_TILE_PER_CORE = 1024;
constexpr int32_t FLOAT_SIZE = 4;
constexpr int32_t ONE_VALUE = 1;
constexpr int32_t ZERO_VALUE = 0;
constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float TWO_FLOAT_VALUE = 2.0f;
constexpr float LN2 = 0.69314718055f;
constexpr float LN2_COEFF = 8.0f;
constexpr float DELTA_COEFF = 4.0f;

class FlashGaussianBuildMask {
public:
    __aicore__ inline FlashGaussianBuildMask()
    {}

    __aicore__ inline void GetTilingData(const FlashGaussianBuildMaskTilingData *tiling_data)
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

    __aicore__ inline void PreInit(const FlashGaussianBuildMaskTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex = GetBlockIdx();

        GetTilingData(tiling_data);

        if (this->blockIndex < numLcore) {
            tileNumPerCore = tileNumPerLcore;
            tileStartIndex = this->blockIndex * tileNumPerCore;
        } else {
            tileNumPerCore = tileNumPerScore;
            tileStartIndex = numLcore * tileNumPerLcore + (this->blockIndex - numLcore) * tileNumPerCore;
        }
    }

    __aicore__ inline void Init(GM_ADDR means2d, GM_ADDR opacity, GM_ADDR conics, GM_ADDR covars2d, \
                                GM_ADDR cnt, GM_ADDR tile_grid, GM_ADDR mask, \
                                const FlashGaussianBuildMaskTilingData *tiling_data, TPipe* pipe)
    {
        PreInit(tiling_data);
        uint64_t baseGaussSize = batchSize * cameraNum * gaussNum;
        uint64_t baseBufferSize = taskNumPerLoop * FLOAT_SIZE;
        this->_pipe = pipe;

        means2dGM.SetGlobalBuffer((__gm__ DTYPE_MEANS2D *)means2d, baseGaussSize * MEANS2D_TAIL_DIM);
        opacityGM.SetGlobalBuffer((__gm__ DTYPE_OPACITY *)opacity, baseGaussSize);
        conicsGM.SetGlobalBuffer((__gm__ DTYPE_CONICS *)conics, baseGaussSize * CONICS_TAIL_DIM);
        covars2dGM.SetGlobalBuffer((__gm__ DTYPE_COVARS2D *)covars2d, baseGaussSize * COVARS2D_TAIL_DIM);
        cntGM.SetGlobalBuffer((__gm__ DTYPE_CNT *)cnt, batchSize * cameraNum);
        tilegridGM.SetGlobalBuffer((__gm__ DTYPE_TILE_GRID *)tile_grid, numTile * TILE_GRID_TAIL_DIM);
        maskGM.SetGlobalBuffer((__gm__ DTYPE_MASK *)mask, baseGaussSize * numTile);

        this->_pipe->InitBuffer(InputTensorBuffer, baseBufferSize * INPUT_BUFFER_NUM);
        this->_pipe->InitBuffer(MaskTensorBuffer, baseBufferSize);
        this->_pipe->InitBuffer(ComputingTensorBuffer, baseBufferSize * COMPUTE_BUFFER_NUM);
        this->_pipe->InitBuffer(CompareTensorBuffer, taskNumPerLoop * COMPARE_BUFFER_NUM);

        InputTensor = InputTensorBuffer.Get<DTYPE_MEANS2D>();
        MaskTensor = MaskTensorBuffer.Get<DTYPE_MASK>();
        CompareTensor = CompareTensorBuffer.Get<uint8_t>();
        ComputingTensor = ComputingTensorBuffer.Get<float>();
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

        for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (uint32_t cameraIdx = 0; cameraIdx < cameraNum; cameraIdx++) {
                ComputingCurGaussNum(batchIdx, cameraIdx);
                ComputingForSingleTile(batchIdx, cameraIdx, tileGridXArr, tileGridYArr);
            }
        }
    }

    __aicore__ inline void ComputingCurGaussNum(uint32_t batchIdx, uint32_t cameraIdx)
    {
        cntGaussNum = cntGM.GetValue(batchIdx * cameraNum + cameraIdx);
        uint32_t alignedGaussNum;
        if (static_cast<uint32_t>((cntGaussNum) % ALIGN_VALUE) != 0) {
            alignedGaussNum = (static_cast<uint32_t>(cntGaussNum / ALIGN_VALUE) + 1) * ALIGN_VALUE;
        } else {
            alignedGaussNum = cntGaussNum;
        }
        tailNum = alignedGaussNum - cntGaussNum;
        taskLoop = static_cast<int32_t>((cntGaussNum + taskNumPerLoop - 1) / taskNumPerLoop);
    }

    __aicore__ inline void ComputingForSingleTile(uint32_t batchIdx, uint32_t cameraIdx, \
                                                  float* tileGridXArr, float* tileGridYArr)
    {
        for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
            GetComputeLength(taskLoopIndex);
            if (taskLoopIndex == 0) {
                SyncCopyInForBound(batchIdx, cameraIdx, taskLoopIndex);
            }
            GetRect(taskLoopIndex);
            CopyInForFlashGS(batchIdx, cameraIdx, taskLoopIndex);
            for (uint32_t tileLoopIndex = 0; tileLoopIndex < tileNumPerCore; tileLoopIndex++) {
                CopyInTileGrid(tileLoopIndex, tileGridXArr, tileGridYArr);
                ComputingBound(tileLoopIndex);
                BlockContainsCenter();
                BlockIntersectEllipse(batchIdx, cameraIdx, taskLoopIndex, tileLoopIndex);
                if (taskLoopIndex == taskLoop - 1 && tailNum != 0) {
                    ProcessDirtyData();
                    SetAtomicAdd<float>();
                    CopyOut(batchIdx, cameraIdx, taskLoopIndex, tileLoopIndex); // 尾核&非对齐&最后一次循环时搬出
                    SetAtomicNone();
                } else {
                    CopyOut(batchIdx, cameraIdx, taskLoopIndex, tileLoopIndex); // 非尾核或对齐场景搬出
                }
            }
        }
    }

    __aicore__ inline void GetComputeLength(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == taskLoop - 1) {
            taskNumPerCurLoop = (cntGaussNum + tailNum) - taskLoopIndex * taskNumPerLoop;
        } else {
            taskNumPerCurLoop = taskNumPerLoop;
        }
    }

    __aicore__ inline void SyncCopyInForBound(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t baseCopyinIndex = batchIdx * cameraNum * gaussNum + cameraIdx * gaussNum;
        uint64_t taskCopyinIndex = taskLoopIndex * taskNumPerLoop;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], \
                 opacityGM[baseCopyinIndex + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], \
                 covars2dGM[baseCopyinIndex * COVARS2D_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], \
                 covars2dGM[baseCopyinIndex * COVARS2D_TAIL_DIM + taskCopyinIndex + gaussNum * COVARS2D_OFFSET_2], \
                 taskNumPerCurLoop);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        DataCopy(InputTensor, means2dGM[baseCopyinIndex * MEANS2D_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop], \
                 means2dGM[baseCopyinIndex * MEANS2D_TAIL_DIM + taskCopyinIndex + gaussNum], taskNumPerCurLoop);
    }

    __aicore__ inline void AsyncCopyInForBound(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t baseCopyinIndex = batchIdx * cameraNum * gaussNum + cameraIdx * gaussNum;
        uint64_t taskCopyinIndex = taskLoopIndex * taskNumPerLoop;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], \
                 opacityGM[baseCopyinIndex + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], \
                 covars2dGM[baseCopyinIndex * COVARS2D_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], \
                 covars2dGM[baseCopyinIndex * COVARS2D_TAIL_DIM + taskCopyinIndex + gaussNum * COVARS2D_OFFSET_2], \
                 taskNumPerCurLoop);

        DataCopy(InputTensor, means2dGM[baseCopyinIndex * MEANS2D_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop], \
                 means2dGM[baseCopyinIndex * MEANS2D_TAIL_DIM + taskCopyinIndex + gaussNum], taskNumPerCurLoop);
    }

    __aicore__ inline void CopyInForFlashGS(uint32_t batchIdx, uint32_t cameraIdx, uint32_t taskLoopIndex)
    {
        uint64_t baseCopyinIndex = batchIdx * cameraNum * gaussNum + cameraIdx * gaussNum;
        uint64_t taskCopyinIndex = taskLoopIndex * taskNumPerLoop;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], \
                 conicsGM[baseCopyinIndex * CONICS_TAIL_DIM + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], \
                 conicsGM[baseCopyinIndex * CONICS_TAIL_DIM + taskCopyinIndex + gaussNum], taskNumPerCurLoop);
        DataCopy(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], \
                 conicsGM[baseCopyinIndex * CONICS_TAIL_DIM + taskCopyinIndex + gaussNum * CONICS_OFFSET_2], \
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
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        }

        // power = LN2 * 8 + LN2 * torch.log2(opacity) -> 0
        Log2(ComputingTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);
        Muls(ComputingTensor, ComputingTensor, LN2, taskNumPerCurLoop);
        Adds(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], ComputingTensor, LN2_COEFF * LN2, taskNumPerCurLoop);
        Muls(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], \
             InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], TWO_FLOAT_VALUE, taskNumPerCurLoop);

        // w = (torch.sqrt(2*cov00[:, None]*power)+1).floor().squeeze()
        // h = (torch.sqrt(2*cov11[:, None]*power)+1).floor().squeeze()
        Mul(ComputingTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);

        Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
             ComputingTensor, taskNumPerCurLoop);
        Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);

        Adds(ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
             ONE_FLOAT_VALUE, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             ONE_FLOAT_VALUE, taskNumPerCurLoop);

        Floor(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], ComputingTensor, taskNumPerCurLoop);
        Floor(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);

        if (taskLoopIndex == 0) {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }

        // rect_min_w = torch.clamp(means_x - w, 0, width - 1.0), rect_min_h = torch.clamp(means_y - h, 0, height - 1.0)
        Sub(ComputingTensor[taskNumPerCurLoop], InputTensor, \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], taskNumPerCurLoop);
        Sub(ComputingTensor, InputTensor[taskNumPerCurLoop], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], taskNumPerCurLoop);

        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                 ComputingTensor[taskNumPerCurLoop], ZERO_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                 ComputingTensor, ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                 imageWidth - ONE_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                 imageHeight - ONE_FLOAT_VALUE, taskNumPerCurLoop);

        // rect_max_w = torch.clamp(means_x + w, 0, width - 1.0)
        // rect_max_h = torch.clamp(means_y + h, 0, height - 1.0)
        Add(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], InputTensor, \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], taskNumPerCurLoop);
        Add(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], InputTensor[taskNumPerCurLoop], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], taskNumPerCurLoop);

        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                 InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], ZERO_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                 InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                 imageWidth - ONE_FLOAT_VALUE, taskNumPerCurLoop);
        ClampMax(InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                 imageHeight - ONE_FLOAT_VALUE, taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingBound(uint32_t tileLoopIndex)
    {
        CompareScalar(CompareTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6], \
                      tileGridX, CMPMODE::GT, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop], \
                      InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], tileGridY, CMPMODE::GT, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
                      InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], tileGridX + tileSize, \
                      CMPMODE::LT, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
                      InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], tileGridY + tileSize, \
                      CMPMODE::LT, taskNumPerCurLoop);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        Select(MaskTensor, CompareTensor, InputTensor[taskNumPerCurLoop * INPUT_OFFSET_6], \
               tileGridX, SELMODE::VSEL_TENSOR_SCALAR_MODE, taskNumPerCurLoop);
        Select(ComputingTensor, CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
               InputTensor[taskNumPerCurLoop * INPUT_OFFSET_8], tileGridX + tileSize, \
               SELMODE::VSEL_TENSOR_SCALAR_MODE, taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], CompareTensor[taskNumPerCurLoop], \
               InputTensor[taskNumPerCurLoop * INPUT_OFFSET_7], tileGridY, \
               SELMODE::VSEL_TENSOR_SCALAR_MODE, taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
               InputTensor[taskNumPerCurLoop * INPUT_OFFSET_9], tileGridY + tileSize, \
               SELMODE::VSEL_TENSOR_SCALAR_MODE, taskNumPerCurLoop);

        Compare(CompareTensor[taskNumPerCurLoop], ComputingTensor, MaskTensor, CMPMODE::GT, taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor[taskNumPerCurLoop], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], CMPMODE::GT, taskNumPerCurLoop);
        And(CompareTensor, CompareTensor[taskNumPerCurLoop], \
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop); // all_in_mask
    }

    __aicore__ inline void BlockContainsCenter()
    {
        CompareScalar(CompareTensor[taskNumPerCurLoop], InputTensor, tileGridX, CMPMODE::GT, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
                      InputTensor[taskNumPerCurLoop], tileGridY, CMPMODE::GT, taskNumPerCurLoop);

        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
                      InputTensor, (tileGridX + tileSize - ONE_FLOAT_VALUE), CMPMODE::LE, taskNumPerCurLoop);
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], \
                      InputTensor[taskNumPerCurLoop], (tileGridY + tileSize - ONE_FLOAT_VALUE), \
                      CMPMODE::LE, taskNumPerCurLoop);

        And(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop], \
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop], \
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop); // center_flag
    }

    __aicore__ inline void BlockIntersectEllipse(uint32_t batchIdx, uint32_t cameraIdx, \
                                                 uint32_t taskLoopIndex, uint32_t tileLoopIndex)
    {
        // pix2x = pix_min_x + pix_max_x
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], InputTensor, \
                      float(0.5f) * (tileGridX + tileGridX + tileSize - ONE_FLOAT_VALUE), \
                      CMPMODE::LT, taskNumPerCurLoop);
        // x_pix_min = meansx_broad - pix_min_x
        Adds(ComputingTensor, InputTensor, float(-1.0f) * tileGridX, taskNumPerCurLoop);
        // x_pix_max = meansx_broad - pix_max_x
        Adds(ComputingTensor[taskNumPerCurLoop], InputTensor, \
             float(-1.0f) * (tileGridX + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        // dx = torch.where(compation1, x_pix_min, x_pix_max)
        Select(MaskTensor, CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
               ComputingTensor, ComputingTensor[taskNumPerCurLoop], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);

        if (tileLoopIndex == 0) {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        }

        // b = -2 * conic_01[:, None] * dx
        Muls(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], \
             TWO_FLOAT_VALUE * float(-1.0f), taskNumPerCurLoop);
        Mul(ComputingTensor, ComputingTensor[taskNumPerCurLoop], MaskTensor, taskNumPerCurLoop);

        // c = conic_00[:, None] * dx * dx - w
        Mul(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], \
            MaskTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor[taskNumPerCurLoop], \
            MaskTensor, taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);

        // delta = b * b - 4 * a * c
        Mul(MaskTensor, ComputingTensor, ComputingTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], DELTA_COEFF, taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            taskNumPerCurLoop);

        // t1 = (pix_min - mean[:, None]) * (2 * a) + b = b1 - y_pix_min * a1_double
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor[taskNumPerCurLoop], \
             float(-1.0f) * tileGridY, taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor, \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // t2 = (pix_max - mean[:, None]) * (2 * a) + b
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor[taskNumPerCurLoop], \
             float(-1.0f) * (tileGridY + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        Muls(MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor, \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], taskNumPerCurLoop);
        Sub(MaskTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // delta1_compare = delta1 >= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
                      ComputingTensor[taskNumPerCurLoop], ZERO_FLOAT_VALUE, CMPMODE::GE, taskNumPerCurLoop);
        // t11_compare = t11 <= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
                      ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ZERO_FLOAT_VALUE, \
                      CMPMODE::LE, taskNumPerCurLoop);
        // t11_delta1_compare = t11 * t11 <= delta1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                ComputingTensor[taskNumPerCurLoop], CMPMODE::LE, taskNumPerCurLoop);
        // t11_compare_total = t11_compare | t11_delta1_compare
        Or(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], taskNumPerCurLoop);

        // t21_compare = t21 >= 0.0
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], MaskTensor, \
                      ZERO_FLOAT_VALUE, CMPMODE::GE, taskNumPerCurLoop);
        // t21_delta1_compare = t21 * t21 <= delta1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor, MaskTensor, taskNumPerCurLoop);
        Compare(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputingTensor[taskNumPerCurLoop], \
                CMPMODE::LE, taskNumPerCurLoop);
        // t21_compare_total = t21_compare | t21_delta1_compare
        Or(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], \
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_4], taskNumPerCurLoop);
        // flag1
        And(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], \
            CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_3], taskNumPerCurLoop);
        // center_flag | flag1
        Or(CompareTensor[taskNumPerCurLoop], CompareTensor[taskNumPerCurLoop], \
           CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], taskNumPerCurLoop);

        // pix2x = pix_min_x + pix_max_x
        CompareScalar(CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], InputTensor[taskNumPerCurLoop], \
                      float(0.5f) * (tileGridY + tileGridY + tileSize - ONE_FLOAT_VALUE), \
                      CMPMODE::LT, taskNumPerCurLoop);
        // x_pix_min = meansx_broad - pix_min_x
        Adds(ComputingTensor, InputTensor[taskNumPerCurLoop], float(-1.0f) * tileGridY, taskNumPerCurLoop);
        // x_pix_max = meansx_broad - pix_max_x
        Adds(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop], \
             float(-1.0f) * (tileGridY + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        // dx = torch.where(compation1, x_pix_min, x_pix_max)
        Select(MaskTensor, CompareTensor[taskNumPerCurLoop * COMPARE_OFFSET_2], ComputingTensor, \
               ComputingTensor[taskNumPerCurLoop], SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);

        // b = -2 * conic_01[:, None] * dx
        Muls(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_4], \
             TWO_FLOAT_VALUE * float(-1.0f), taskNumPerCurLoop);
        Mul(ComputingTensor, ComputingTensor[taskNumPerCurLoop], MaskTensor, taskNumPerCurLoop);

        // c = conic_00[:, None] * dx * dx - w
        Mul(ComputingTensor[taskNumPerCurLoop], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_5], \
            MaskTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor[taskNumPerCurLoop], \
            MaskTensor, taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_2], taskNumPerCurLoop);

        // delta = b * b - 4 * a * c
        Mul(MaskTensor, ComputingTensor, ComputingTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], DELTA_COEFF, taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            taskNumPerCurLoop);

        // t1 = (pix_min - mean[:, None]) * (2 * a) + b = b1 - y_pix_min * a1_double
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor, \
             float(-1.0f) * tileGridX, taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputingTensor, \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        // t2 = (pix_max - mean[:, None]) * (2 * a) + b
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], InputTensor, \
             float(-1.0f) * (tileGridX + tileSize - ONE_FLOAT_VALUE), taskNumPerCurLoop);
        Muls(MaskTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             TWO_FLOAT_VALUE, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MaskTensor, \
            InputTensor[taskNumPerCurLoop * INPUT_OFFSET_3], taskNumPerCurLoop);
        Sub(MaskTensor, ComputingTensor, \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        asyncTaskNum = taskNumPerCurLoop;
        if ((taskLoopIndex < taskLoop - 1) && (tileLoopIndex == tileNumPerCore - 1)) {
            GetComputeLength(taskLoopIndex + 1);
            AsyncCopyInForBound(batchIdx, cameraIdx, taskLoopIndex + 1);
        }

        // delta1_compare = delta1 >= 0.0
        CompareScalar(CompareTensor[asyncTaskNum * COMPARE_OFFSET_2], \
                      ComputingTensor[asyncTaskNum], ZERO_FLOAT_VALUE, CMPMODE::GE, asyncTaskNum);
        // t11_compare = t11 <= 0.0
        CompareScalar(CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], \
                      ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_2], ZERO_FLOAT_VALUE, CMPMODE::LE, asyncTaskNum);
        // t11_delta1_compare = t11 * t11 <= delta1
        Mul(ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_3], ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_2], \
            ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_2], asyncTaskNum);
        Compare(CompareTensor[asyncTaskNum * COMPARE_OFFSET_4], ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_3], \
                ComputingTensor[asyncTaskNum], CMPMODE::LE, asyncTaskNum);
        // t11_compare_total = t11_compare | t11_delta1_compare
        Or(CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], \
           CompareTensor[asyncTaskNum * COMPARE_OFFSET_4], asyncTaskNum);
        And(CompareTensor[asyncTaskNum * COMPARE_OFFSET_2], CompareTensor[asyncTaskNum * COMPARE_OFFSET_2], \
            CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], asyncTaskNum);

        // t21_compare = t21 >= 0.0
        CompareScalar(CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], MaskTensor, \
                      ZERO_FLOAT_VALUE, CMPMODE::GE, asyncTaskNum);
        // t21_delta1_compare = t21 * t21 <= delta1
        Mul(ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_3], MaskTensor, MaskTensor, asyncTaskNum);
        Compare(CompareTensor[asyncTaskNum * COMPARE_OFFSET_4], ComputingTensor[asyncTaskNum * COMPUTE_OFFSET_3], \
                ComputingTensor[asyncTaskNum], CMPMODE::LE, asyncTaskNum);
        // t21_compare_total = t21_compare | t21_delta1_compare
        Or(CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], \
           CompareTensor[asyncTaskNum * COMPARE_OFFSET_4], asyncTaskNum);
        // flag1
        And(CompareTensor[asyncTaskNum * COMPARE_OFFSET_2], CompareTensor[asyncTaskNum * COMPARE_OFFSET_2], \
            CompareTensor[asyncTaskNum * COMPARE_OFFSET_3], asyncTaskNum);
        // center_flag | flag1
        Or(CompareTensor[asyncTaskNum], CompareTensor[asyncTaskNum], \
           CompareTensor[asyncTaskNum * COMPARE_OFFSET_2], asyncTaskNum);

        // all_in_mask = all_in_mask & (center_flag | ellipse_isect_flag)
        And(CompareTensor, CompareTensor, CompareTensor[asyncTaskNum], asyncTaskNum);
        Duplicate(ComputingTensor, ONE_FLOAT_VALUE, asyncTaskNum);
        Select(MaskTensor, CompareTensor, ComputingTensor, ZERO_FLOAT_VALUE, \
               SELMODE::VSEL_TENSOR_SCALAR_MODE, asyncTaskNum);
    }

    __aicore__ inline void ProcessDirtyData()
    {
        uint32_t taskNumPerTailCore = asyncTaskNum - tailNum;
        Duplicate(ComputingTensor[asyncTaskNum], ONE_FLOAT_VALUE, asyncTaskNum);
        for (uint32_t i = 0; i < tailNum; i++) {
            ComputingTensor[asyncTaskNum].SetValue(taskNumPerTailCore + i, ZERO_FLOAT_VALUE);
        }
        Mul(MaskTensor, MaskTensor, ComputingTensor[asyncTaskNum], asyncTaskNum);
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx, uint32_t cameraIdx, \
                                   uint32_t taskLoopIndex, uint32_t tileLoopIndex)
    {
        uint64_t batchCopyoutIndex = batchIdx * cameraNum * gaussNum * numTile;
        uint64_t cameraCopyoutIndex = cameraIdx * gaussNum * numTile;
        uint64_t tileCopyoutIndex = (tileStartIndex + tileLoopIndex) * gaussNum;
        uint64_t taskCopyoutIndex = taskLoopIndex * taskNumPerLoop;

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        DataCopy(maskGM[batchCopyoutIndex + cameraCopyoutIndex + tileCopyoutIndex + taskCopyoutIndex], \
                 MaskTensor, asyncTaskNum);

        if ((taskLoopIndex == taskLoop - 1) && (batchIdx == batchSize - 1) && \
            (cameraIdx == cameraNum - 1) && (tileLoopIndex == tileNumPerCore - 1)) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> InputTensorBuffer, MaskTensorBuffer;
    TBuf <TPosition::VECCALC> ComputingTensorBuffer, CompareTensorBuffer;
    LocalTensor<float> InputTensor, MaskTensor, ComputingTensor;
    LocalTensor<uint8_t> CompareTensor;

    GlobalTensor<DTYPE_MEANS2D> means2dGM;
    GlobalTensor<DTYPE_CNT> cntGM;
    GlobalTensor<DTYPE_OPACITY> opacityGM;
    GlobalTensor<DTYPE_CONICS> conicsGM;
    GlobalTensor<DTYPE_COVARS2D> covars2dGM;
    GlobalTensor<DTYPE_MASK> maskGM;
    GlobalTensor<DTYPE_TILE_GRID> tilegridGM;

    float imageWidth, imageHeight, tileSize;
    float tileGridX, tileGridY;
    uint32_t batchSize, cameraNum, gaussNum, cntGaussNum, tailNum;
    uint32_t tileNumPerCore, tileNumPerScore, tileNumPerLcore, numScore, numLcore, blockDim;
    uint32_t taskNumPerLoop, taskNumPerCurLoop, tileStartIndex, taskLoop, numTile, asyncTaskNum;
    uint64_t blockIndex, ubTotalSize;
};

extern "C" __global__ __aicore__ void flash_gaussian_build_mask(GM_ADDR means2d, GM_ADDR opacity, \
                                                                GM_ADDR conics, GM_ADDR covars2d, \
                                                                GM_ADDR cnt, GM_ADDR tile_grid, \
                                                                GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        FlashGaussianBuildMask op;
        op.Init(means2d, opacity, conics, covars2d, cnt, tile_grid, mask, &tiling_data, &pipe);
        op.Process();
    }
}