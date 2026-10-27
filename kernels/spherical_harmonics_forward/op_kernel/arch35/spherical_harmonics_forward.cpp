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
#include "spherical_harmonics_forward_degree0_vf.h"
using namespace AscendC;
using namespace std;

constexpr uint32_t DEGREE0_PIPELINE_DUB_DEPTH = 2;
constexpr uint32_t DEGREE0_SUB_ALIGN = 8;

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
        // degree 1-4 pipeline: the depth-2 ping-pong splits a tile into two
        // half-size queue slots, so the physical queue footprint equals the
        // per-task budget. Temp14Buffer holds the small recurrence reserve.
        this->_pipe = pipe;

        // coeffs: [B, K, C, N], C in [1, 4]
        coeffsGM.SetGlobalBuffer((__gm__ DTYPE_COEFFS *)coeffs, \
                                batchSize * coeffNum * channelNum * gaussianNum);
        outputGM.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output, outputSize);

        if (degreeNum == L0_SH_COEFFS_NUM) {
            // degree-0 double-buffered fast path: only coeffs + output UB buffers,
            // each split into DEGREE0_PIPELINE_DUB_DEPTH ping-pong sub-tile slots.
            uint32_t degree0SubLen = taskNumPerLoop / DEGREE0_PIPELINE_DUB_DEPTH;
            degree0SubLen = ((degree0SubLen + DEGREE0_SUB_ALIGN - 1) / DEGREE0_SUB_ALIGN) * DEGREE0_SUB_ALIGN;
            this->_pipe->InitBuffer(CoeffsQue, DEGREE0_PIPELINE_DUB_DEPTH, \
                                    degree0SubLen * channelNum * degreeNum * FLOAT_SIZE);
            this->_pipe->InitBuffer(OutputQue, DEGREE0_PIPELINE_DUB_DEPTH, \
                                    degree0SubLen * channelNum * FLOAT_SIZE);
            return;
        }

        dirsGM.SetGlobalBuffer((__gm__ DTYPE_DIRS *)dirs, dirSize);
        // degree 1-4: depth-2 ping-pong carries coeffs/output (and dirs) through
        // VECIN/VECOUT queues; when the whole per-core range fits one tile
        // (taskLoop == 1) a plain single-pass (queueDepth 1) is used instead.
        uint32_t queueDepth = DEGREE0_PIPELINE_DUB_DEPTH;
        uint32_t degree14SubLen = taskNumPerLoop;
        if (taskLoop > 1) {
            degree14SubLen = taskNumPerLoop / DEGREE0_PIPELINE_DUB_DEPTH;
            degree14SubLen = ((degree14SubLen + DEGREE0_SUB_ALIGN - 1) / \
                              DEGREE0_SUB_ALIGN) * DEGREE0_SUB_ALIGN;
            if (degree14SubLen == 0) {
                degree14SubLen = taskNumPerLoop;
            }
        } else {
            queueDepth = 1;
        }
        this->_pipe->InitBuffer(DirsQue14, queueDepth, \
                                degree14SubLen * AXIS_NUM * FLOAT_SIZE);
        this->_pipe->InitBuffer(CoeffsQue14, queueDepth, \
                                degree14SubLen * channelNum * degreeNum * FLOAT_SIZE);
        this->_pipe->InitBuffer(OutputQue14, queueDepth, \
                                degree14SubLen * channelNum * FLOAT_SIZE);
    }

    __aicore__ inline void Process()
    {
        if (degreeNum == L0_SH_COEFFS_NUM) {
            ProcessDegree0();
            return;
        }
        // degree 1-4: depth-2 ping-pong over sub-tiles of the per-core range.
        uint32_t degree14SubLen = taskNumPerLoop;
        if (taskLoop > 1) {
            degree14SubLen = taskNumPerLoop / DEGREE0_PIPELINE_DUB_DEPTH;
            degree14SubLen = ((degree14SubLen + DEGREE0_SUB_ALIGN - 1) / \
                              DEGREE0_SUB_ALIGN) * DEGREE0_SUB_ALIGN;
            if (degree14SubLen == 0) {
                degree14SubLen = taskNumPerLoop;
            }
        }
        for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            uint32_t numSub = (taskNumPerCore + degree14SubLen - 1) / degree14SubLen;
            if (numSub == 0) {
                continue;
            }
            CopyIn(degree14SubLen, batchIdx, 0);
            for (uint32_t subIdx = 0; subIdx < numSub; subIdx++) {
                if (subIdx + 1 < numSub) {
                    CopyIn(degree14SubLen, batchIdx, subIdx + 1);
                }
                if (subIdx > 0) {
                    CopyOut(degree14SubLen, batchIdx, subIdx - 1);
                }
                uint32_t len = taskNumPerCore - subIdx * degree14SubLen;
                if (len > degree14SubLen) {
                    len = degree14SubLen;
                }
                LocalTensor<float> dirsIn = DirsQue14.DeQue<float>();
                LocalTensor<float> coeffsIn = CoeffsQue14.DeQue<float>();
                LocalTensor<float> outLocal = OutputQue14.AllocTensor<float>();
                ComputingOutput(dirsIn, coeffsIn, outLocal, len);
                DirsQue14.FreeTensor(dirsIn);
                CoeffsQue14.FreeTensor(coeffsIn);
                OutputQue14.EnQue(outLocal);
            }
            CopyOut(degree14SubLen, batchIdx, numSub - 1);
        }
    }

    __aicore__ inline void ProcessDegree0()
    {
        // Degree-0 double-buffered tile pipeline. The host sizes taskNumPerLoop
        // from the degree-0-only UB budget (coeffs + output = 6 units/task); the
        // kernel splits each tile into DEGREE0_PIPELINE_DUB_DEPTH ping-pong
        // sub-tiles carried by VECIN/VECOUT queues of the same depth, so the MTE2
        // copy-in of the next sub-tile overlaps the V compute / MTE3 copy-out of
        // the current one.
        uint32_t subLen = taskNumPerLoop / DEGREE0_PIPELINE_DUB_DEPTH;
        subLen = ((subLen + DEGREE0_SUB_ALIGN - 1) / DEGREE0_SUB_ALIGN) * DEGREE0_SUB_ALIGN;
        if (subLen == 0) {
            subLen = taskNumPerLoop;
        }
        for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            uint32_t numSub = (taskNumPerCore + subLen - 1) / subLen;
            for (uint32_t subIdx = 0; subIdx < numSub; subIdx++) {
                uint32_t len = taskNumPerCore - subIdx * subLen;
                if (len > subLen) {
                    len = subLen;
                }
                uint64_t base = taskStartIndex + static_cast<uint64_t>(subIdx) * subLen;
                bool isTail = subIdx == numSub - 1 && tailNum != 0 && \
                              this->blockIndex == blockDim - 1;
                LocalTensor<DTYPE_COEFFS> coeffsLocal = CoeffsQue.AllocTensor<DTYPE_COEFFS>();
                LocalTensor<DTYPE_OUTPUT> outputLocal = OutputQue.AllocTensor<DTYPE_OUTPUT>();
                CopyInDegree0(coeffsLocal, batchIdx, base, len);
                CoeffsQue.EnQue(coeffsLocal);
                LocalTensor<DTYPE_COEFFS> coeffsIn = CoeffsQue.DeQue<DTYPE_COEFFS>();
                ComputingDegree0(coeffsIn, outputLocal, len);
                CoeffsQue.FreeTensor(coeffsIn);
                OutputQue.EnQue(outputLocal);
                LocalTensor<DTYPE_OUTPUT> outputOut = OutputQue.DeQue<DTYPE_OUTPUT>();
                CopyOutDegree0(outputOut, batchIdx, base, len, isTail);
                OutputQue.FreeTensor(outputOut);
            }
        }
    }

    __aicore__ inline void CopyInDegree0(const LocalTensor<DTYPE_COEFFS> &coeffs, uint32_t batchIdx,
                                         uint64_t taskBase, uint32_t validLen)
    {
        uint64_t coeffBatchOffset = batchIdx * coeffNum * channelNum * gaussianNum;
        // Single cross-channel strided copy (one descriptor for all C channels):
        // coeffs GM [B, K=1, C, N] -> coeffs UB [C, validLen]; channel rows are
        // gaussianNum elements apart in GM and packed contiguously in UB.
        int64_t channelGap = (static_cast<int64_t>(gaussianNum) - static_cast<int64_t>(validLen));
        if (channelGap < 0) {
            channelGap = 0;
        }
        DataCopyExtParams copyInParams{
            static_cast<uint16_t>(channelNum),
            static_cast<uint32_t>(validLen * FLOAT_SIZE),
            channelGap * FLOAT_SIZE,
            0,
            0
        };
        DataCopyPad(coeffs, coeffsGM[coeffBatchOffset + taskBase], copyInParams,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    }

    __aicore__ inline void ComputingDegree0(const LocalTensor<DTYPE_COEFFS> &coeffs,
                                            const LocalTensor<DTYPE_OUTPUT> &out, uint32_t validLen)
    {
        // degree 0: out[c, n] = SH_C0 * coeffs[c, 0, n]; coeffs/output layout [C, K=1, validLen]
        // Register-resident VF body over the full contiguous [C, validLen] region.
        SphericalHarmonicsDegree0ScaleVf((__ubuf__ float *)out.GetPhyAddr(),
                                         (__ubuf__ float *)coeffs.GetPhyAddr(),
                                         static_cast<uint32_t>(channelNum * validLen),
                                         L0_M0_SH_PARAM);
    }

    __aicore__ inline void CopyOutDegree0(const LocalTensor<DTYPE_OUTPUT> &out, uint32_t batchIdx,
                                          uint64_t taskBase, uint32_t validLen, bool isTailBlock)
    {
        uint64_t batchOffset = batchIdx * channelNum * gaussianNum;
        if (isTailBlock) {
            uint32_t validTaskNum = validLen - tailNum;
            DataCopyExtParams copyOutParams{
                1,
                static_cast<uint32_t>(validTaskNum * FLOAT_SIZE),
                0,
                0,
                0
            };
            for (uint32_t channelIdx = 0; channelIdx < channelNum; channelIdx++) {
                DataCopyPad(outputGM[batchOffset + channelIdx * gaussianNum + taskBase], \
                            out[channelIdx * validLen], copyOutParams);
            }
        } else {
            // Single cross-channel strided copy (one descriptor for all C channels):
            // output UB [C, validLen] -> output GM [B, C, N].
            int64_t channelGap = (static_cast<int64_t>(gaussianNum) - static_cast<int64_t>(validLen));
            if (channelGap < 0) {
                channelGap = 0;
            }
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(channelNum),
                static_cast<uint32_t>(validLen * FLOAT_SIZE),
                0,
                channelGap * FLOAT_SIZE,
                0
            };
            DataCopyPad(outputGM[batchOffset + taskBase], out, copyOutParams);
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

    __aicore__ inline void CopyIn(uint32_t degree14SubLen, uint32_t batchIdx, uint32_t degree14SubIdx)
    {
        uint32_t len = taskNumPerCore - degree14SubIdx * degree14SubLen;
        if (len > degree14SubLen) {
            len = degree14SubLen;
        }
        uint64_t base = taskStartIndex + static_cast<uint64_t>(degree14SubIdx) * degree14SubLen;
        uint64_t dirsBatchOffset = static_cast<uint64_t>(batchIdx) * AXIS_NUM * gaussianNum;
        int64_t channelGap = static_cast<int64_t>(gaussianNum) - static_cast<int64_t>(len);
        if (channelGap < 0) {
            channelGap = 0;
        }

        LocalTensor<DTYPE_DIRS> dirsLocal = DirsQue14.AllocTensor<DTYPE_DIRS>();
        // dirs GM [B, 3, N] -> UB [3, len]: one cross-axis strided descriptor.
        DataCopyExtParams dirsParams{
            static_cast<uint16_t>(AXIS_NUM),
            static_cast<uint32_t>(len * FLOAT_SIZE),
            static_cast<uint32_t>(channelGap * FLOAT_SIZE),
            0,
            0
        };
        DataCopyPad(dirsLocal, dirsGM[dirsBatchOffset + base], dirsParams,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
        DirsQue14.EnQue(dirsLocal);

        LocalTensor<DTYPE_COEFFS> coeffsLocal = CoeffsQue14.AllocTensor<DTYPE_COEFFS>();
        // coeffs GM [B, K, C, N] -> UB [K, C, len]: exactly one merged descriptor
        // per k (blockCount = channelNum) instead of K * channelNum DataCopy calls.
        uint64_t coeffBatchOffset = static_cast<uint64_t>(batchIdx) * coeffNum * channelNum * gaussianNum;
        DataCopyExtParams coeffParams{
            static_cast<uint16_t>(channelNum),
            static_cast<uint32_t>(len * FLOAT_SIZE),
            static_cast<uint32_t>(channelGap * FLOAT_SIZE),
            0,
            0
        };
        for (uint32_t k = 0; k < degreeNum; k++) {
            uint64_t srcBase = coeffBatchOffset + \
                               static_cast<uint64_t>(gaussianNum) * k * channelNum + base;
            DataCopyPad(coeffsLocal[k * channelNum * len], coeffsGM[srcBase], coeffParams,
                        DataCopyPadExtParams<float>{false, 0, 0, 0});
        }
        CoeffsQue14.EnQue(coeffsLocal);
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

    // degree 1-4 per-sub-tile compute as a register-resident VF body: the three
    // dirs axes are L2-normalised and the real-SH basis is built in registers,
    // one __VEC_SCOPE__ over 64-fp32 tiles; out[c] = sum_k coeffs[k,c]*basis_k.
    // (On this SDK the LocalTensor level-2 binary vector APIs lower to
    // __simd_vf__ and fault in this kernel; the register-resident form is the
    // proven path, the same one the degree-0 fast path uses.)
    template <uint32_t DEG>
    __aicore__ inline void ComputeDegree14Vf(const LocalTensor<float> &dirsTensor,
                                             const LocalTensor<float> &coeffsTensor,
                                             const LocalTensor<float> &outTensor, uint32_t len)
    {
        __ubuf__ float *dirs = (__ubuf__ float *)dirsTensor.GetPhyAddr();
        __ubuf__ float *coeffs = (__ubuf__ float *)coeffsTensor.GetPhyAddr();
        __ubuf__ float *out = (__ubuf__ float *)outTensor.GetPhyAddr();
        const uint16_t tiles = (uint16_t)((len + 63u) >> 6);
        uint32_t remain = len;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vx, vy, vz, nrm, tb, tc;
            Reg::RegTensor<float> acc, cur, prod;
            Reg::RegTensor<float> z2, fC1, fS1, fC2, fS2, t2, fB, fC, fD, b6;
            Reg::MaskReg mask;
            for (uint16_t t = 0; t < tiles; ++t) {
                uint32_t off = (uint32_t)t * 64u;
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vx, dirs + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vy, dirs + len + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vz, dirs + 2u * len + off);
                Reg::Mul(tb, vx, vx, mask);
                Reg::Mul(tc, vy, vy, mask);
                Reg::Add(tb, tb, tc, mask);
                Reg::Mul(tc, vz, vz, mask);
                Reg::Add(tb, tb, tc, mask);
                Reg::Sqrt(nrm, tb, mask);
                Reg::Div(vx, vx, nrm, mask);
                Reg::Div(vy, vy, nrm, mask);
                Reg::Div(vz, vz, nrm, mask);
                const uint16_t channel = (uint16_t)channelNum;
                for (uint16_t c = 0; c < channel; ++c) {
                    __ubuf__ float *cb = coeffs + (uint32_t)c * len + off;
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(cur, cb);
                    Reg::Muls(acc, cur, L0_M0_SH_PARAM, mask);
                    Reg::Muls(cur, vy, L1_M0_SH_PARAM, mask);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + channelNum * len);
                    Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                    Reg::Muls(cur, vz, (-1.0f) * L1_M0_SH_PARAM, mask);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 2u * channelNum * len);
                    Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                    Reg::Muls(cur, vx, L1_M0_SH_PARAM, mask);
                    Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 3u * channelNum * len);
                    Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                    if constexpr (DEG >= 2) {
                        Reg::Mul(z2, vz, vz, mask);
                        Reg::Mul(tb, vx, vx, mask);
                        Reg::Mul(tc, vy, vy, mask);
                        Reg::Sub(fC1, tb, tc, mask);
                        Reg::Mul(tb, vx, vy, mask);
                        Reg::Muls(fS1, tb, TWO_FLOAT_VALUE, mask);
                        Reg::Muls(tb, z2, L2_M0_SH_PARAM_1, mask);
                        Reg::Adds(b6, tb, L2_M0_SH_PARAM_2, mask);
                        Reg::Muls(fB, vz, L2_M1_SH_PARAM, mask);
                        Reg::Muls(cur, fS1, L2_M2_SH_PARAM, mask);
                        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 4u * channelNum * len);
                        Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                        Reg::Mul(cur, fB, vy, mask);
                        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 5u * channelNum * len);
                        Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 6u * channelNum * len);
                        Reg::Mul(prod, prod, b6, mask); Reg::Add(acc, acc, prod, mask);
                        Reg::Mul(cur, fB, vx, mask);
                        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 7u * channelNum * len);
                        Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                        Reg::Muls(cur, fC1, L2_M2_SH_PARAM, mask);
                        Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 8u * channelNum * len);
                        Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                        if constexpr (DEG >= 3) {
                            Reg::Muls(fB, vz, L3_M2_SH_PARAM, mask);
                            Reg::Mul(tb, vx, fC1, mask);
                            Reg::Mul(tc, vy, fS1, mask);
                            Reg::Sub(fC2, tb, tc, mask);
                            Reg::Mul(tb, vx, fS1, mask);
                            Reg::Mul(tc, vy, fC1, mask);
                            Reg::Add(fS2, tb, tc, mask);
                            Reg::Muls(t2, z2, L3_M0_SH_PARAM_1, mask);
                            Reg::Adds(t2, t2, L3_M0_SH_PARAM_2, mask);
                            Reg::Muls(fC, z2, L3_M1_SH_PARAM_1, mask);
                            Reg::Adds(fC, fC, L3_M1_SH_PARAM_2, mask);
                            Reg::Muls(cur, fS2, L3_M3_SH_PARAM, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 9u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            Reg::Mul(cur, fB, fS1, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 10u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            Reg::Mul(cur, fC, vy, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 11u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            Reg::Mul(cur, vz, t2, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 12u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            Reg::Mul(cur, fC, vx, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 13u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            Reg::Mul(cur, fB, fC1, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 14u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            Reg::Muls(cur, fC2, L3_M3_SH_PARAM, mask);
                            Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 15u * channelNum * len);
                            Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                            if constexpr (DEG >= 4) {
                                Reg::Muls(fB, vz, L4_M3_SH_PARAM, mask);
                                Reg::Mul(cur, fB, fS2, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 17u * channelNum * len);
                                Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Mul(cur, fB, fC2, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 23u * channelNum * len);
                                Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Muls(fC, z2, L4_M2_SH_PARAM_1, mask);
                                Reg::Adds(fC, fC, L4_M2_SH_PARAM_2, mask);
                                Reg::Mul(cur, fC, fS1, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 18u * channelNum * len);
                                Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Mul(cur, fC, fC1, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 22u * channelNum * len);
                                Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Muls(fD, z2, L4_M1_SH_PARAM_1, mask);
                                Reg::Adds(fD, fD, L4_M1_SH_PARAM_2, mask);
                                Reg::Mul(fD, fD, vz, mask);
                                Reg::Mul(cur, fD, vy, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 19u * channelNum * len);
                                Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Mul(cur, fD, vx, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 21u * channelNum * len);
                                Reg::Mul(prod, prod, cur, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Mul(fD, t2, z2, mask);
                                Reg::Muls(fD, fD, L4_M0_SH_PARAM_1, mask);
                                Reg::Muls(fC, b6, L4_M0_SH_PARAM_2, mask);
                                Reg::Add(fD, fD, fC, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 20u * channelNum * len);
                                Reg::Mul(prod, prod, fD, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Mul(tb, vx, fC2, mask);
                                Reg::Mul(tc, vy, fS2, mask);
                                Reg::Sub(fD, tb, tc, mask);
                                Reg::Muls(fD, fD, L4_M4_SH_PARAM, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 24u * channelNum * len);
                                Reg::Mul(prod, prod, fD, mask); Reg::Add(acc, acc, prod, mask);
                                Reg::Mul(tb, vx, fS2, mask);
                                Reg::Mul(tc, vy, fC2, mask);
                                Reg::Add(fD, tb, tc, mask);
                                Reg::Muls(fD, fD, L4_M4_SH_PARAM, mask);
                                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(prod, cb + 16u * channelNum * len);
                                Reg::Mul(prod, prod, fD, mask); Reg::Add(acc, acc, prod, mask);
                            }
                        }
                    }
                    Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM_B32>(out + (uint32_t)c * len + off, acc, mask);
                }
            }
        }
    }

    __aicore__ inline void ComputingOutput(const LocalTensor<float> &dirsTensor,
                                           const LocalTensor<float> &coeffsTensor,
                                           const LocalTensor<float> &outTensor, uint32_t len)
    {
        if (degreeNum == L1_SH_COEFFS_NUM) {
            ComputeDegree14Vf<1>(dirsTensor, coeffsTensor, outTensor, len);
        } else if (degreeNum == L2_SH_COEFFS_NUM) {
            ComputeDegree14Vf<2>(dirsTensor, coeffsTensor, outTensor, len);
        } else if (degreeNum == L3_SH_COEFFS_NUM) {
            ComputeDegree14Vf<3>(dirsTensor, coeffsTensor, outTensor, len);
        } else {
            ComputeDegree14Vf<4>(dirsTensor, coeffsTensor, outTensor, len);
        }
    }

    __aicore__ inline void CopyOut(uint32_t degree14SubLen, uint32_t batchIdx, uint32_t degree14SubIdx)
    {
        uint32_t len = taskNumPerCore - degree14SubIdx * degree14SubLen;
        if (len > degree14SubLen) {
            len = degree14SubLen;
        }
        uint64_t taskCopyoutIndex = taskStartIndex + static_cast<uint64_t>(degree14SubIdx) * degree14SubLen;
        uint64_t batchOffset = static_cast<uint64_t>(batchIdx) * channelNum * gaussianNum;
        uint32_t numSub = (taskNumPerCore + degree14SubLen - 1) / degree14SubLen;
        bool isTailBlock = degree14SubIdx == numSub - 1 && tailNum != 0 && \
                           this->blockIndex == blockDim - 1;
        LocalTensor<DTYPE_OUTPUT> outLocal = OutputQue14.DeQue<DTYPE_OUTPUT>();
        if (isTailBlock) {
            // copy-out only the valid elements per channel (last sub of the last
            // core), one descriptor per channel as in the original tail path.
            uint32_t validLen = len - tailNum;
            DataCopyExtParams copyOutParams{
                1,
                static_cast<uint32_t>(validLen * FLOAT_SIZE),
                0,
                0,
                0
            };
            for (uint32_t ch = 0; ch < channelNum; ++ch) {
                DataCopyPad(outputGM[batchOffset + ch * gaussianNum + taskCopyoutIndex], \
                            outLocal[ch * len], copyOutParams);
            }
        } else {
            int64_t channelGap = static_cast<int64_t>(gaussianNum) - static_cast<int64_t>(len);
            if (channelGap < 0) {
                channelGap = 0;
            }
            DataCopyExtParams copyOutParams{
                static_cast<uint16_t>(channelNum),
                static_cast<uint32_t>(len * FLOAT_SIZE),
                0,
                static_cast<uint32_t>(channelGap * FLOAT_SIZE),
                0
            };
            DataCopyPad(outputGM[batchOffset + taskCopyoutIndex], outLocal, copyOutParams);
        }
        OutputQue14.FreeTensor(outLocal);
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> DirsTensorBuffer, CoeffsTensorBuffer, OutputTensorBuffer;
    TBuf <TPosition::VECCALC> ResultTensorBuffer, ComputingTensorBuffer;
    TQue <QuePosition::VECIN, DEGREE0_PIPELINE_DUB_DEPTH> CoeffsQue;
    TQue <QuePosition::VECOUT, DEGREE0_PIPELINE_DUB_DEPTH> OutputQue;
    LocalTensor<float> DirsTensor, CoeffsTensor, OutputTensor, ResultTensor, ComputingTensor;
    TQue <QuePosition::VECIN, DEGREE0_PIPELINE_DUB_DEPTH> DirsQue14;
    TQue <QuePosition::VECIN, DEGREE0_PIPELINE_DUB_DEPTH> CoeffsQue14;
    TQue <QuePosition::VECOUT, DEGREE0_PIPELINE_DUB_DEPTH> OutputQue14;

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