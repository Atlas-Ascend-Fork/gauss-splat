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

constexpr int32_t FLOAT_ALIGN_NUM = 8;
constexpr int32_t FLOAT_SIZE = 4;

constexpr uint32_t QUAT_TAIL_DIM = 4;
constexpr uint32_t SCALES_TAIL_DIM = 3;
constexpr uint32_t COVARS_TAIL_DIM = 9;
constexpr uint32_t COMPUTE_BUFFER_NUM = 9;

constexpr uint32_t QUAT_OFFSET_2 = 2;
constexpr uint32_t QUAT_OFFSET_3 = 3;
constexpr uint32_t SCALES_OFFSET_2 = 2;

constexpr uint32_t COVARS_OFFSET_2 = 2;
constexpr uint32_t COVARS_OFFSET_3 = 3;
constexpr uint32_t COVARS_OFFSET_4 = 4;
constexpr uint32_t COVARS_OFFSET_5 = 5;
constexpr uint32_t COVARS_OFFSET_6 = 6;
constexpr uint32_t COVARS_OFFSET_7 = 7;
constexpr uint32_t COVARS_OFFSET_8 = 8;

constexpr uint32_t COMPUTE_OFFSET_2 = 2;
constexpr uint32_t COMPUTE_OFFSET_3 = 3;
constexpr uint32_t COMPUTE_OFFSET_4 = 4;
constexpr uint32_t COMPUTE_OFFSET_5 = 5;
constexpr uint32_t COMPUTE_OFFSET_6 = 6;
constexpr uint32_t COMPUTE_OFFSET_7 = 7;
constexpr uint32_t COMPUTE_OFFSET_8 = 8;

constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float TWO_FLOAT_VALUE = 2.0f;
constexpr float ROTMAT_COEFF = -2.0f;

class QuatScalesToCovars {
public:
    __aicore__ inline QuatScalesToCovars()
    {}

    __aicore__ inline void GetTilingData(const QuatScalesToCovarsTilingData *tiling_data)
    {
        batchSizeNum = tiling_data->batchSizeNum;
        gaussianNum = tiling_data->gaussianNum;
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

    __aicore__ inline void PreInit(const QuatScalesToCovarsTilingData *tiling_data)
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

    __aicore__ inline void Init(GM_ADDR quat, GM_ADDR scales, GM_ADDR covars,
                                const QuatScalesToCovarsTilingData *tiling_data, TPipe* pipe)
    {
        PreInit(tiling_data);
        bnSize = batchSizeNum * gaussianNum;
        uint64_t BaseBufferSize = taskNumPerLoop * FLOAT_SIZE;
        this->_pipe = pipe;

        quatGM.SetGlobalBuffer((__gm__ DTYPE_QUAT *)quat, bnSize * QUAT_TAIL_DIM);
        scalesGM.SetGlobalBuffer((__gm__ DTYPE_SCALES *)scales, bnSize * SCALES_TAIL_DIM);
        covarsGM.SetGlobalBuffer((__gm__ DTYPE_COVARS *)covars, bnSize * COVARS_TAIL_DIM);

        this->_pipe->InitBuffer(QuatTensorBuffer, BaseBufferSize * QUAT_TAIL_DIM);
        this->_pipe->InitBuffer(ScalesTensorBuffer, BaseBufferSize * SCALES_TAIL_DIM);
        this->_pipe->InitBuffer(CovarsTensorBuffer, BaseBufferSize * COVARS_TAIL_DIM);
        this->_pipe->InitBuffer(ComputeTensorBuffer, BaseBufferSize * COMPUTE_BUFFER_NUM);

        QuatTensor = QuatTensorBuffer.Get<DTYPE_QUAT>();
        ScalesTensor = ScalesTensorBuffer.Get<DTYPE_SCALES>();
        CovarsTensor = CovarsTensorBuffer.Get<DTYPE_COVARS>();
        ComputeTensor = ComputeTensorBuffer.Get<float>();
    }

    __aicore__ inline void Process()
    {
        for (int32_t batchIndex = 0; batchIndex < batchSizeNum; batchIndex++) {
            for (int32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                ComputingTaskNum(taskLoopIndex);
                CopyIn(batchIndex, taskLoopIndex);
                NormalizeQuats();
                Quat2Rotmat();
                ComputingCovars();
                if (taskLoopIndex == taskLoop - 1 && tailNum != 0 && this->blockIndex == blockDim - 1) {
                    CopyOutTailCore(batchIndex, taskLoopIndex); // 尾核&非对齐&最后一次循环时搬出
                } else {
                    CopyOut(batchIndex, taskLoopIndex); // 非尾核或对齐场景搬出
                }
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

    __aicore__ inline void CopyIn(int32_t batchIndex, int32_t taskLoopIndex)
    {
        uint64_t batchCopyinIndex = batchIndex * gaussianNum;
        uint64_t taskCopyinIndex = taskStartIndex + taskLoopIndex * taskNumPerLoop;
        uint64_t baseCopyinLength = static_cast<uint64_t>(taskNumPerCurLoop);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        for (int32_t i = 0; i < QUAT_TAIL_DIM; i++) {
            DataCopy(QuatTensor[baseCopyinLength * i], \
                     quatGM[batchCopyinIndex * QUAT_TAIL_DIM + i * gaussianNum + taskCopyinIndex], \
                     baseCopyinLength);
        }
        for (int32_t i = 0; i < SCALES_TAIL_DIM; i++) {
            DataCopy(ScalesTensor[baseCopyinLength * i], \
                     scalesGM[batchCopyinIndex * SCALES_TAIL_DIM + i * gaussianNum + taskCopyinIndex], \
                     baseCopyinLength);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void NormalizeQuats()
    {
        Mul(ComputeTensor, QuatTensor, QuatTensor, taskNumPerCurLoop * QUAT_TAIL_DIM);
        Add(ComputeTensor, ComputeTensor, ComputeTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Add(ComputeTensor, ComputeTensor, ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Sqrt(ComputeTensor[taskNumPerCurLoop], ComputeTensor, taskNumPerCurLoop);

        Div(QuatTensor, QuatTensor, ComputeTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Div(QuatTensor[taskNumPerCurLoop], QuatTensor[taskNumPerCurLoop], \
            ComputeTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Div(QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], \
            ComputeTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Div(QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], \
            ComputeTensor[taskNumPerCurLoop], taskNumPerCurLoop);
    }

    __aicore__ inline void Quat2Rotmat()
    {
        // 0: x^2, 1: y^2, 2: z^2
        Mul(CovarsTensor, QuatTensor[taskNumPerCurLoop], QuatTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop], QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], taskNumPerCurLoop);
        // 3: x * y, 4: x * z
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], QuatTensor[taskNumPerCurLoop], \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], QuatTensor[taskNumPerCurLoop], \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], taskNumPerCurLoop);
        // 5: w * y, 6: w * z
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], QuatTensor, \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], QuatTensor, \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], taskNumPerCurLoop);
        // 7: y * z, 8: x * w
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_2], \
            QuatTensor[taskNumPerCurLoop * QUAT_OFFSET_3], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], QuatTensor[taskNumPerCurLoop], \
            QuatTensor, taskNumPerCurLoop);

        Add(ComputeTensor, CovarsTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            taskNumPerCurLoop);
        Muls(ComputeTensor, ComputeTensor, ROTMAT_COEFF, taskNumPerCurLoop);
        Adds(ComputeTensor, ComputeTensor, ONE_FLOAT_VALUE, taskNumPerCurLoop);

        Sub(ComputeTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop], ComputeTensor[taskNumPerCurLoop], \
            TWO_FLOAT_VALUE, taskNumPerCurLoop);

        Add(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            TWO_FLOAT_VALUE, taskNumPerCurLoop);

        Add(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            TWO_FLOAT_VALUE, taskNumPerCurLoop);

        Add(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], CovarsTensor, \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ROTMAT_COEFF, taskNumPerCurLoop);
        Adds(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ONE_FLOAT_VALUE, taskNumPerCurLoop);

        Sub(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            TWO_FLOAT_VALUE, taskNumPerCurLoop);

        Sub(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            TWO_FLOAT_VALUE, taskNumPerCurLoop);

        Add(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            TWO_FLOAT_VALUE, taskNumPerCurLoop);

        Add(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor, \
            CovarsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Muls(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ROTMAT_COEFF, taskNumPerCurLoop);
        Adds(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ONE_FLOAT_VALUE, taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingCovars()
    {
        Mul(ComputeTensor, ComputeTensor, ScalesTensor, taskNumPerCurLoop * SCALES_TAIL_DIM);
        Mul(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ScalesTensor, taskNumPerCurLoop * SCALES_TAIL_DIM);
        Mul(ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ScalesTensor, taskNumPerCurLoop * SCALES_TAIL_DIM);

        // bmm
        Mul(CovarsTensor, ComputeTensor, ComputeTensor, taskNumPerCurLoop * COMPUTE_BUFFER_NUM);
        for (int32_t i = 0; i < SCALES_TAIL_DIM; i++) {
            Add(ScalesTensor, CovarsTensor[i * SCALES_TAIL_DIM * taskNumPerCurLoop], \
                CovarsTensor[(i * SCALES_TAIL_DIM + 1) * taskNumPerCurLoop], taskNumPerCurLoop);
            Add(CovarsTensor[i * QUAT_TAIL_DIM * taskNumPerCurLoop], ScalesTensor, \
                CovarsTensor[(i * SCALES_TAIL_DIM + SCALES_OFFSET_2) * taskNumPerCurLoop], taskNumPerCurLoop);
        }

        Mul(ScalesTensor, ComputeTensor, ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            taskNumPerCurLoop * SCALES_TAIL_DIM);
        Add(CovarsTensor[taskNumPerCurLoop], ScalesTensor, ScalesTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop], \
            ScalesTensor[taskNumPerCurLoop * SCALES_OFFSET_2], taskNumPerCurLoop);
        Adds(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], CovarsTensor[taskNumPerCurLoop], \
            ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        Mul(ScalesTensor, ComputeTensor, ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            taskNumPerCurLoop * SCALES_TAIL_DIM);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], ScalesTensor, \
            ScalesTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            ScalesTensor[taskNumPerCurLoop * SCALES_OFFSET_2], taskNumPerCurLoop);
        Adds(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            ZERO_FLOAT_VALUE, taskNumPerCurLoop);

        Mul(ScalesTensor, ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputeTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop * SCALES_TAIL_DIM);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], ScalesTensor, \
            ScalesTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            ScalesTensor[taskNumPerCurLoop * SCALES_OFFSET_2], taskNumPerCurLoop);
        Adds(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            ZERO_FLOAT_VALUE, taskNumPerCurLoop);
    }

    __aicore__ inline void CopyOut(int32_t batchIndex, int32_t taskLoopIndex)
    {
        uint64_t taskCopyoutIndex = (taskStartIndex + taskLoopIndex * taskNumPerLoop);
        uint64_t batchCopyoutIndex = batchIndex * gaussianNum;
        uint32_t copyBlockNum = taskNumPerCurLoop / FLOAT_ALIGN_NUM;
        AscendC::SliceInfo sliceSrcInfo[] = {{0, taskNumPerCurLoop, 0, copyBlockNum, taskNumPerCurLoop}};
        AscendC::SliceInfo sliceDstInfo[] = {{0, taskNumPerCurLoop, \
                                              gaussianNum - taskNumPerCurLoop, copyBlockNum, taskNumPerCurLoop}};

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        for (int32_t i = 0; i < COVARS_TAIL_DIM; i++) {
            DataCopy(covarsGM[(batchCopyoutIndex * COVARS_TAIL_DIM + taskCopyoutIndex) + gaussianNum * i], \
                     CovarsTensor[taskNumPerCurLoop * i], sliceDstInfo, sliceSrcInfo, 1);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void CopyOutTailCore(int32_t batchIndex, int32_t taskLoopIndex)
    {
        int32_t taskNumPerTailCore = taskNumPerCurLoop - tailNum;
        uint64_t taskCopyoutIndex = (taskStartIndex + taskLoopIndex * taskNumPerLoop);
        uint64_t batchCopyoutIndex = batchIndex * gaussianNum;

        Duplicate(ComputeTensor, ONE_FLOAT_VALUE, taskNumPerCurLoop);
        for (int32_t i = 0; i < tailNum; i++) {
            ComputeTensor.SetValue(taskNumPerTailCore + i, ZERO_FLOAT_VALUE);
        }
        for (int32_t i = 0; i < COVARS_TAIL_DIM; i++) {
            Mul(CovarsTensor[taskNumPerCurLoop * i], CovarsTensor[taskNumPerCurLoop * i], \
                ComputeTensor, taskNumPerCurLoop);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        SetAtomicAdd<float>();
        for (int32_t i = 0; i < COVARS_TAIL_DIM; i++) {
            DataCopy(covarsGM[(batchCopyoutIndex * COVARS_TAIL_DIM + taskCopyoutIndex) + gaussianNum * i], \
                     CovarsTensor[taskNumPerCurLoop * i], taskNumPerCurLoop);
        }
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        SetAtomicNone();
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> QuatTensorBuffer, ScalesTensorBuffer, CovarsTensorBuffer, ComputeTensorBuffer;
    LocalTensor<float> QuatTensor, ScalesTensor, CovarsTensor, ComputeTensor;

    GlobalTensor<DTYPE_QUAT> quatGM;
    GlobalTensor<DTYPE_SCALES> scalesGM;
    GlobalTensor<DTYPE_COVARS> covarsGM;

    uint32_t batchSizeNum, gaussianNum, totalTaskNum, tailNum, taskNumPerScore, taskNumPerLcore;
    uint32_t numScore, numLcore, blockDim;
    uint64_t blockIndex, ubTotalSize, bnSize, bcSize, bcnSize;
    uint32_t taskNumPerLoop, taskNumPerCurLoop, taskNumPerCore, taskStartIndex, taskLoop;
};

extern "C" __global__ __aicore__ void quat_scales_to_covars(GM_ADDR quat, GM_ADDR scales, \
                                                            GM_ADDR covars, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        QuatScalesToCovars op;
        op.Init(quat, scales, covars, &tiling_data, &pipe);
        op.Process();
    }
}