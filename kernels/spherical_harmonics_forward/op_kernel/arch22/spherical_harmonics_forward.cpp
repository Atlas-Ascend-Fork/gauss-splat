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

constexpr uint32_t L2_SH_DEGREE = 2;
constexpr uint32_t L3_SH_DEGREE = 3;
constexpr uint32_t L4_SH_DEGREE = 4;
constexpr uint32_t L2_SH_BUFFER_NUM = 6;
constexpr uint32_t L3_SH_BUFFER_NUM = 10;
constexpr uint32_t L4_SH_BUFFER_NUM = 16;

constexpr uint32_t L0_SH_COEFFS_NUM = 1;
constexpr uint32_t L1_SH_COEFFS_NUM = 4;
constexpr uint32_t L2_SH_COEFFS_NUM = 9;
constexpr uint32_t L3_SH_COEFFS_NUM = 16;
constexpr uint32_t L4_SH_COEFFS_NUM = 25;
constexpr uint32_t AXIS_NUM = 3;

constexpr uint32_t DIRS_OFFSET_2 = 2;
constexpr uint32_t L1RST_OFFSET_2 = 2;
constexpr uint32_t L1RST_OFFSET_3 = 3;
constexpr uint32_t L2RST_OFFSET_4 = 4;
constexpr uint32_t L2RST_OFFSET_5 = 5;
constexpr uint32_t L2RST_OFFSET_6 = 6;
constexpr uint32_t L2RST_OFFSET_7 = 7;
constexpr uint32_t L2RST_OFFSET_8 = 8;
constexpr uint32_t L3RST_OFFSET_9 = 9;
constexpr uint32_t L3RST_OFFSET_10 = 10;
constexpr uint32_t L3RST_OFFSET_11 = 11;
constexpr uint32_t L3RST_OFFSET_12 = 12;
constexpr uint32_t L3RST_OFFSET_13 = 13;
constexpr uint32_t L3RST_OFFSET_14 = 14;
constexpr uint32_t L3RST_OFFSET_15 = 15;
constexpr uint32_t L4RST_OFFSET_16 = 16;
constexpr uint32_t L4RST_OFFSET_17 = 17;
constexpr uint32_t L4RST_OFFSET_18 = 18;
constexpr uint32_t L4RST_OFFSET_19 = 19;
constexpr uint32_t L4RST_OFFSET_20 = 20;
constexpr uint32_t L4RST_OFFSET_21 = 21;
constexpr uint32_t L4RST_OFFSET_22 = 22;
constexpr uint32_t L4RST_OFFSET_23 = 23;
constexpr uint32_t L4RST_OFFSET_24 = 24;

constexpr uint32_t L2_SH_OFFSET_2 = 2;
constexpr uint32_t L2_SH_OFFSET_3 = 3;
constexpr uint32_t L2_SH_OFFSET_4 = 4;
constexpr uint32_t L2_SH_OFFSET_5 = 5;
constexpr uint32_t L3_SH_OFFSET_6 = 6;
constexpr uint32_t L3_SH_OFFSET_7 = 7;
constexpr uint32_t L3_SH_OFFSET_8 = 8;
constexpr uint32_t L3_SH_OFFSET_9 = 9;
constexpr uint32_t L4_SH_OFFSET_10 = 10;
constexpr uint32_t L4_SH_OFFSET_11 = 11;
constexpr uint32_t L4_SH_OFFSET_12 = 12;
constexpr uint32_t L4_SH_OFFSET_13 = 13;
constexpr uint32_t L4_SH_OFFSET_14 = 14;
constexpr uint32_t L4_SH_OFFSET_15 = 15;

constexpr int32_t FLOAT_SIZE = 4;
constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float TWO_FLOAT_VALUE = 2.0f;

constexpr float L0_M0_SH_PARAM = 0.2820947917738781f;
constexpr float L1_M0_SH_PARAM = -0.48860251190292f;
constexpr float L2_M0_SH_PARAM_1 = 0.9461746957575601f;
constexpr float L2_M0_SH_PARAM_2 = -0.3153915652525201f;
constexpr float L2_M1_SH_PARAM = -1.092548430592079f;
constexpr float L2_M2_SH_PARAM = 0.5462742152960395f;
constexpr float L3_M0_SH_PARAM_1 = 1.865881662950577f;
constexpr float L3_M0_SH_PARAM_2 = -1.119528997770346f;
constexpr float L3_M1_SH_PARAM_1 = -2.285228997322329f;
constexpr float L3_M1_SH_PARAM_2 = 0.4570457994644658f;
constexpr float L3_M2_SH_PARAM = 1.445305721320277f;
constexpr float L3_M3_SH_PARAM = -0.5900435899266435f;
constexpr float L4_M0_SH_PARAM_1 = 1.984313483298443f;
constexpr float L4_M0_SH_PARAM_2 = -1.006230589874905f;
constexpr float L4_M1_SH_PARAM_1 = -4.683325804901025f;
constexpr float L4_M1_SH_PARAM_2 = 2.007139630671868f;
constexpr float L4_M2_SH_PARAM_1 = 3.31161143515146f;
constexpr float L4_M2_SH_PARAM_2 = -0.47308734787878f;
constexpr float L4_M3_SH_PARAM = -1.770130769779931f;
constexpr float L4_M4_SH_PARAM = 0.6258357354491763f;

class SphericalHarmonicsForward {
public:
    __aicore__ inline SphericalHarmonicsForward()
    {}

    __aicore__ inline void GetTilingData(const SphericalHarmonicsForwardTilingData *tiling_data)
    {
        batchSize = tiling_data->batchSize;
        gaussianNum = tiling_data->gaussianNum;
        taskNum = tiling_data->taskNum;
        coeffNum = tiling_data->coeffNum;
        channelNum = tiling_data->channelNum;
        degreeNum = tiling_data->degreeNum;
        degreeUsed = tiling_data->degreeUsed;
        totalTaskNum = tiling_data->totalTaskNum;
        tailNum = tiling_data->tailNum;
        taskNumPerScore = tiling_data->taskNumPerScore;
        taskNumPerLcore = tiling_data->taskNumPerLcore;
        numScore = tiling_data->numScore;
        numLcore = tiling_data->numLcore;
        blockDim = tiling_data->blockDim;
        taskNumPerLoop = tiling_data->taskNumPerLoop;
        ubTotalSize = tiling_data->ubTotalSize;
    }

    __aicore__ inline void PreInit(const SphericalHarmonicsForwardTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex = GetBlockIdx();

        GetTilingData(tiling_data);
        if (this->blockIndex < numLcore) {
            taskNumPerCore = taskNumPerLcore;
            taskStartIndex = this->blockIndex * taskNumPerCore;
        } else {
            taskNumPerCore = taskNumPerScore;
            taskStartIndex = numLcore * taskNumPerLcore + (this->blockIndex - numLcore) * taskNumPerCore;
        }
        taskLoop = static_cast<int32_t>((taskNumPerCore + taskNumPerLoop - 1) / taskNumPerLoop);
    }

    __aicore__ inline void Init(GM_ADDR dirs, GM_ADDR coeffs, GM_ADDR output,
                                const SphericalHarmonicsForwardTilingData *tiling_data, TPipe* pipe)
    {
        PreInit(tiling_data);
        dirSize = batchSize * AXIS_NUM * gaussianNum;
        outputSize = batchSize * channelNum * gaussianNum;
        uint64_t BaseBufferSize = taskNumPerLoop * AXIS_NUM * FLOAT_SIZE;
        uint64_t OutputBaseBufferSize = taskNumPerLoop * channelNum * FLOAT_SIZE;
        this->_pipe = pipe;

        dirsGM.SetGlobalBuffer((__gm__ DTYPE_DIRS *)dirs, dirSize);

        // coeffs: [B, K, C, N], C in [1, 4]
        coeffsGM.SetGlobalBuffer((__gm__ DTYPE_COEFFS *)coeffs, \
                                batchSize * coeffNum * channelNum * gaussianNum);
        outputGM.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output, outputSize);
        this->_pipe->InitBuffer(DirsTensorBuffer, BaseBufferSize);
        this->_pipe->InitBuffer(CoeffsTensorBuffer, \
                                taskNumPerLoop * channelNum * degreeNum * FLOAT_SIZE);
        this->_pipe->InitBuffer(OutputTensorBuffer, OutputBaseBufferSize);
        this->_pipe->InitBuffer(ResultTensorBuffer, taskNumPerLoop * degreeNum * FLOAT_SIZE);
        if (degreeUsed == L2_SH_DEGREE) {
            this->_pipe->InitBuffer(ComputingTensorBuffer, taskNumPerLoop * L2_SH_BUFFER_NUM * FLOAT_SIZE);
        } else if (degreeUsed == L3_SH_DEGREE) {
            this->_pipe->InitBuffer(ComputingTensorBuffer, taskNumPerLoop * L3_SH_BUFFER_NUM * FLOAT_SIZE);
        } else if (degreeUsed == L4_SH_DEGREE) {
            this->_pipe->InitBuffer(ComputingTensorBuffer, taskNumPerLoop * L4_SH_BUFFER_NUM * FLOAT_SIZE);
        } else {
            this->_pipe->InitBuffer(ComputingTensorBuffer, taskNumPerLoop);
        }

        DirsTensor = DirsTensorBuffer.Get<DTYPE_DIRS>();
        CoeffsTensor = CoeffsTensorBuffer.Get<DTYPE_COEFFS>();
        OutputTensor = OutputTensorBuffer.Get<DTYPE_OUTPUT>();
        ResultTensor = ResultTensorBuffer.Get<float>();
        ComputingTensor = ComputingTensorBuffer.Get<float>();
    }

    __aicore__ inline void Process()
    {
        Duplicate(ResultTensor, ZERO_FLOAT_VALUE, ResultTensor.GetSize());
        for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                ComputingTaskNum(taskLoopIndex);
                CopyIn(batchIdx, taskLoopIndex);
                NormalizeDirs();
                ComputingSphericalHarmonics();
                ComputingOutput();
                CopyOut(batchIdx, taskLoopIndex);
            }
        }
    }

    __aicore__ inline void ComputingTaskNum(int32_t taskLoopIndex)
    {
        if (taskLoopIndex == taskLoop - 1) {
            taskNumPerCurLoop = taskNumPerCore - taskLoopIndex * taskNumPerLoop;
        } else {
            taskNumPerCurLoop = taskNumPerLoop;
        }
    }

    __aicore__ inline void CopyIn(uint32_t batchIdx, int32_t taskLoopIndex)
    {
        uint64_t taskCopyinIndex = taskStartIndex + taskLoopIndex * taskNumPerLoop;
        uint64_t batchOffset = batchIdx * AXIS_NUM * gaussianNum;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(DirsTensor, dirsGM[batchOffset + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(DirsTensor[taskNumPerCurLoop], dirsGM[batchOffset + gaussianNum + taskCopyinIndex], taskNumPerCurLoop);
        DataCopy(DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
                 dirsGM[batchOffset + gaussianNum * DIRS_OFFSET_2 + taskCopyinIndex], taskNumPerCurLoop);

        // coeffs GM layout: [B, K, C, N]
        // coeffs UB layout: [C, K, taskNumPerCurLoop]
        uint64_t coeffBatchOffset = batchIdx * coeffNum * channelNum * gaussianNum;
        uint64_t coeffChannelLength = degreeNum * taskNumPerCurLoop;
        for (int32_t j = 0; j < degreeNum; j++) {
            uint64_t coeffBase = coeffBatchOffset + gaussianNum * j * channelNum;
            for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx ++) {
                uint64_t localOffset = channelIdx * coeffChannelLength + j * taskNumPerCurLoop;
                uint64_t globalOffset = coeffBase + channelIdx * gaussianNum + taskCopyinIndex;
                DataCopy(CoeffsTensor[localOffset], coeffsGM[globalOffset], taskNumPerCurLoop);
            }
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void NormalizeDirs()
    {
        Mul(ComputingTensor, DirsTensor, DirsTensor, taskNumPerCurLoop * AXIS_NUM);
        Add(ComputingTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], taskNumPerCurLoop);
        Sqrt(ComputingTensor, ComputingTensor, taskNumPerCurLoop);

        Div(DirsTensor, DirsTensor, ComputingTensor, taskNumPerCurLoop);
        Div(DirsTensor[taskNumPerCurLoop], DirsTensor[taskNumPerCurLoop], ComputingTensor, taskNumPerCurLoop);
        Div(DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
            ComputingTensor, taskNumPerCurLoop);
    }

    __aicore__ inline void Level0SphericalHarmonicsComputing()
    {
        // result[..., 0] = 0.2820947917738781
        Duplicate(ResultTensor, L0_M0_SH_PARAM, taskNumPerCurLoop);
    }

    __aicore__ inline void Level1SphericalHarmonicsComputing()
    {
        Level0SphericalHarmonicsComputing();
        // result[..., 1] = fTmpA * y
        Muls(ResultTensor[taskNumPerCurLoop], DirsTensor[taskNumPerCurLoop], L1_M0_SH_PARAM, taskNumPerCurLoop);
        // result[..., 2] = -fTmpA * z
        Muls(ResultTensor[taskNumPerCurLoop * L1RST_OFFSET_2], DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
             (-1.0f) * L1_M0_SH_PARAM, taskNumPerCurLoop);
        // result[..., 3] = fTmpA * x
        Muls(ResultTensor[taskNumPerCurLoop * L1RST_OFFSET_3], DirsTensor, L1_M0_SH_PARAM, taskNumPerCurLoop);
    }

    __aicore__ inline void Level2SphericalHarmonicsComputing()
    {
        Level1SphericalHarmonicsComputing();
        // fC1 = x * x - y * y
        Mul(ComputingTensor[taskNumPerCurLoop], DirsTensor, DirsTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], DirsTensor[taskNumPerCurLoop], \
            DirsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], taskNumPerCurLoop);
        // fS1 = 2 * x * y
        Mul(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], DirsTensor, \
            DirsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], \
             ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], TWO_FLOAT_VALUE, taskNumPerCurLoop);
        // z2 = z * z
        Mul(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
            DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], taskNumPerCurLoop);
        // fTmpB = -1.092548430592079 * z
        Muls(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
             L2_M1_SH_PARAM, taskNumPerCurLoop);
        // 0.9461746957575601 * z2
        Muls(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_5], \
             ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], L2_M0_SH_PARAM_1, taskNumPerCurLoop);

        // result[..., 4] = fTmpA * fS1
        Muls(ResultTensor[taskNumPerCurLoop * L2RST_OFFSET_4], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], \
             L2_M2_SH_PARAM, taskNumPerCurLoop);
        // result[..., 5] = fTmpB * y
        Mul(ResultTensor[taskNumPerCurLoop * L2RST_OFFSET_5], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
            DirsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        // result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
        Adds(ResultTensor[taskNumPerCurLoop * L2RST_OFFSET_6], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_5], \
             L2_M0_SH_PARAM_2, taskNumPerCurLoop);
        // result[..., 7] = fTmpB * x
        Mul(ResultTensor[taskNumPerCurLoop * L2RST_OFFSET_7], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
            DirsTensor, taskNumPerCurLoop);
        // result[..., 8] = fTmpA * fC1
        Muls(ResultTensor[taskNumPerCurLoop * L2RST_OFFSET_8], ComputingTensor[taskNumPerCurLoop], \
             L2_M2_SH_PARAM, taskNumPerCurLoop);
    }

    __aicore__ inline void Level3SphericalHarmonicsComputing()
    {
        Level2SphericalHarmonicsComputing();
        // fTmpB = 1.445305721320277 * z
        Muls(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
             L3_M2_SH_PARAM, taskNumPerCurLoop);
        // fC2 = x * fC1 - y * fS1
        Mul(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], DirsTensor, \
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], DirsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], taskNumPerCurLoop);
        // fS2 = x * fS1 + y * fC1
        Mul(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], DirsTensor, \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], DirsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], taskNumPerCurLoop);
        // fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
        Muls(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], \
             L3_M1_SH_PARAM_1, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], \
             L3_M1_SH_PARAM_2, taskNumPerCurLoop);
        // (1.865881662950577 * z2 - 1.119528997770346)
        Muls(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_9], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], \
             L3_M0_SH_PARAM_1, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_9], ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_9], \
             L3_M0_SH_PARAM_2, taskNumPerCurLoop);
        // result[..., 9] = fTmpA * fS2
        Muls(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_9], ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], \
             L3_M3_SH_PARAM, taskNumPerCurLoop);
        // result[..., 10] = fTmpB * fS1
        Mul(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_10], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], taskNumPerCurLoop);
        // result[..., 11] = fTmpC * y
        Mul(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_11], DirsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], taskNumPerCurLoop);
        // result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
        Mul(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_12], DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_9], taskNumPerCurLoop);
        // result[..., 13] = fTmpC * x
        Mul(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_13], DirsTensor, \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_8], taskNumPerCurLoop);
        // result[..., 14] = fTmpB * fC1
        Mul(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_14], ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        // result[..., 15] = fTmpA * fC2
        Muls(ResultTensor[taskNumPerCurLoop * L3RST_OFFSET_15], ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], \
             L3_M3_SH_PARAM, taskNumPerCurLoop);
    }

    __aicore__ inline void Level4SphericalHarmonicsComputing()
    {
        Level3SphericalHarmonicsComputing();
        // fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
        Muls(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_10], \
             ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], L4_M1_SH_PARAM_1, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], \
             ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_10], L4_M1_SH_PARAM_2, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], \
            DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], taskNumPerCurLoop);
        // fTmpC = 3.31161143515146 * z2 - 0.47308734787878
        Muls(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], \
             ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], L4_M2_SH_PARAM_1, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], \
             ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], L4_M2_SH_PARAM_2, taskNumPerCurLoop);
        // fTmpB = -1.770130769779931 * z
        Muls(ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
             DirsTensor[taskNumPerCurLoop * DIRS_OFFSET_2], L4_M3_SH_PARAM, taskNumPerCurLoop);
        // fC3 = x * fC2 - y * fS2
        Mul(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_12], DirsTensor, \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_13], DirsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_12], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_12], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_13], taskNumPerCurLoop);
        // fS3 = x * fS2 + y * fC2
        Mul(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_13], DirsTensor, \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_14], DirsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_13], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_13], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_14], taskNumPerCurLoop);
        // result[..., 16] = fTmpA * fS3
        Muls(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_16], \
             ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_13], L4_M4_SH_PARAM, taskNumPerCurLoop);
        // result[..., 17] = fTmpB * fS2
        Mul(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_17], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_7], taskNumPerCurLoop);
        // result[..., 18] = fTmpC * fS1
        Mul(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_18], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_2], taskNumPerCurLoop);
        // result[..., 19] = fTmpD * y
        Mul(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_19], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_10], DirsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        // result[..., 20] = 1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346) +
        // -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
        Mul(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_15], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_3], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_14], \
             ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_15], L4_M0_SH_PARAM_1, taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_15], \
             ResultTensor[taskNumPerCurLoop * L2RST_OFFSET_6], L4_M0_SH_PARAM_2, taskNumPerCurLoop);
        Add(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_20], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_14], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_15], taskNumPerCurLoop);
        // result[..., 21] = fTmpD * x
        Mul(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_21], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_10], DirsTensor, taskNumPerCurLoop);
        // result[..., 22] = fTmpC * fC1
        Mul(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_22], \
            ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        // result[..., 23] = fTmpB * fC2
        Mul(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_23], \
            ComputingTensor[taskNumPerCurLoop * L2_SH_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * L3_SH_OFFSET_6], taskNumPerCurLoop);
        // result[..., 24] = fTmpA * fC3
        Muls(ResultTensor[taskNumPerCurLoop * L4RST_OFFSET_24], \
             ComputingTensor[taskNumPerCurLoop * L4_SH_OFFSET_12], L4_M4_SH_PARAM, taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingSphericalHarmonics()
    {
        if (degreeNum == L0_SH_COEFFS_NUM) {
            Level0SphericalHarmonicsComputing();
        } else if (degreeNum == L1_SH_COEFFS_NUM) {
            Level1SphericalHarmonicsComputing();
        } else if (degreeNum == L2_SH_COEFFS_NUM) {
            Level2SphericalHarmonicsComputing();
        } else if (degreeNum == L3_SH_COEFFS_NUM) {
            Level3SphericalHarmonicsComputing();
        } else {
            Level4SphericalHarmonicsComputing();
        }
    }

    __aicore__ inline void ComputingOutput()
    {
        // bases: [K, task], coeffs: [C, K, task]
        uint64_t coeffLength = taskNumPerCurLoop * degreeNum;
        for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx ++) {
            uint64_t coeffOffset = channelIdx * coeffLength;
            Mul(CoeffsTensor[coeffOffset], ResultTensor, CoeffsTensor[coeffOffset], coeffLength);
        }

        // Sum over K: [C, K, task] -> [C, task]
        for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx ++) {
            uint64_t coeffOffset = channelIdx * coeffLength;
            uint64_t outputOffset = channelIdx * taskNumPerCurLoop;
            if (degreeNum >= DIRS_OFFSET_2) {
                Add(OutputTensor[outputOffset], CoeffsTensor[coeffOffset], \
                    CoeffsTensor[coeffOffset + taskNumPerCurLoop], taskNumPerCurLoop);
                for (int32_t coeffIndex = DIRS_OFFSET_2; coeffIndex < degreeNum; coeffIndex ++) {
                    Add(OutputTensor[outputOffset], OutputTensor[outputOffset], \
                        CoeffsTensor[coeffOffset + taskNumPerCurLoop * coeffIndex],
                        taskNumPerCurLoop);
                }
            } else {
                Adds(OutputTensor[outputOffset], CoeffsTensor[coeffOffset], \
                    ZERO_FLOAT_VALUE, taskNumPerCurLoop);
            }
        }
    }

    __aicore__ inline void CopyOut(uint32_t batchIdx, int32_t taskLoopIndex)
    {
        uint64_t taskCopyoutIndex = (taskStartIndex + taskLoopIndex * taskNumPerLoop);
        uint64_t batchOffset = batchIdx * channelNum * gaussianNum;
        bool isTailBlock = taskLoopIndex == taskLoop - 1 && tailNum != 0 && \
                            this->blockIndex == blockDim - 1;
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        if (isTailBlock) {
            uint32_t validTaskNum = taskNumPerCurLoop - tailNum;
            DataCopyExtParams copyOutParams{
                1,
                static_cast<uint32_t>(validTaskNum * FLOAT_SIZE),
                0,
                0,
                0
            };
            for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
                DataCopyPad(outputGM[batchOffset + channelIdx * gaussianNum + taskCopyoutIndex], \
                            OutputTensor[channelIdx * taskNumPerCurLoop], copyOutParams);
            }
        } else {
            for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
                DataCopy(outputGM[batchOffset + channelIdx * gaussianNum + taskCopyoutIndex], \
                        OutputTensor[channelIdx * taskNumPerCurLoop], taskNumPerCurLoop);
            }
        }

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> DirsTensorBuffer, CoeffsTensorBuffer, OutputTensorBuffer;
    TBuf <TPosition::VECCALC> ResultTensorBuffer, ComputingTensorBuffer;
    LocalTensor<float> DirsTensor, CoeffsTensor, OutputTensor, ResultTensor, ComputingTensor;

    GlobalTensor<DTYPE_DIRS> dirsGM;
    GlobalTensor<DTYPE_COEFFS> coeffsGM;
    GlobalTensor<DTYPE_OUTPUT> outputGM;

    uint32_t batchSize, gaussianNum, taskNum, coeffNum, channelNum, degreeNum, degreeUsed, totalTaskNum, tailNum;
    uint32_t taskNumPerScore, taskNumPerLcore;
    uint32_t numScore, numLcore, blockDim;
    uint64_t blockIndex, ubTotalSize, dirSize, outputSize, bcSize, bcnSize;
    uint32_t taskNumPerLoop, taskNumPerCurLoop, taskNumPerCore, taskStartIndex, taskLoop;
};

extern "C" __global__ __aicore__ void spherical_harmonics_forward(GM_ADDR dirs, GM_ADDR coeffs, \
                                                                  GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        SphericalHarmonicsForward op;
        op.Init(dirs, coeffs, output, &tiling_data, &pipe);
        op.Process();
    }
}