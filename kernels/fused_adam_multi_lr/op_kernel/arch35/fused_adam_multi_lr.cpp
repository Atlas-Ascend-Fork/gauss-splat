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
#include "basic_api/kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace std;

constexpr int32_t FLOAT_ALIGN_NUM = 8;
constexpr int32_t FLOAT_SIZE = 4;
constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr uint32_t MAX_TENSOR_NUM = 64;

class FusedAdamMultiLr {
public:
    __aicore__ inline FusedAdamMultiLr()
    {}

    __aicore__ inline void GetTilingData(const FusedAdamMultiLrTilingData *tilingData)
    {
        // Keep the tiling pointer and read per-tensor arrays directly from it:
        // the tiling struct (generated kernel_tiling.h, mirrored from the host
        // fused_adam_multi_lr_tiling.h) is the single source of truth, so the class
        // does not re-declare or copy the per-tensor arrays.
        this->tilingData = tilingData;

        beta1 = tilingData->beta1;
        beta2 = tilingData->beta2;
        eps = tilingData->eps;
        weightDecay = tilingData->weightDecay;
        oneMinusBeta1 = tilingData->oneMinusBeta1;
        oneMinusBeta2 = tilingData->oneMinusBeta2;
        invBiasCorrection2 = tilingData->invBiasCorrection2;
        tensorNum = tilingData->tensorNum;
    }

    __aicore__ inline void Init(GM_ADDR param, GM_ADDR grad, GM_ADDR exp_avg, GM_ADDR exp_avg_sq,
                                GM_ADDR param_out, GM_ADDR exp_avg_out, GM_ADDR exp_avg_sq_out,
                                const FusedAdamMultiLrTilingData *tiling_data, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex = GetBlockIdx();
        this->_pipe = pipe;

        GetTilingData(tiling_data);

        // Parse TensorList inputs: each GM_ADDR is a ListTensorDesc
        paramListDesc.Init(param);
        gradListDesc.Init(grad);
        expAvgListDesc.Init(exp_avg);
        expAvgSqListDesc.Init(exp_avg_sq);
        paramOutListDesc.Init(param_out);
        expAvgOutListDesc.Init(exp_avg_out);
        expAvgSqOutListDesc.Init(exp_avg_sq_out);

        // Init UB buffers: 6 buffers for SIMD compute
        uint32_t maxTaskNumPerLoop = 0;
        for (uint32_t i = 0; i < tensorNum; i++) {
            if (static_cast<uint32_t>(tilingData->taskNumPerLoops[i]) > maxTaskNumPerLoop) {
                maxTaskNumPerLoop = static_cast<uint32_t>(tilingData->taskNumPerLoops[i]);
            }
        }
        uint64_t baseBufferSize = static_cast<uint64_t>(maxTaskNumPerLoop) * FLOAT_SIZE;

        this->_pipe->InitBuffer(ParamBuffer, baseBufferSize);
        this->_pipe->InitBuffer(GradBuffer, baseBufferSize);
        this->_pipe->InitBuffer(ExpAvgBuffer, baseBufferSize);
        this->_pipe->InitBuffer(ExpAvgSqBuffer, baseBufferSize);
        this->_pipe->InitBuffer(ComputeBuffer, baseBufferSize);
        this->_pipe->InitBuffer(Compute2Buffer, baseBufferSize);

        ParamTensor = ParamBuffer.Get<DTYPE_PARAM>();
        GradTensor = GradBuffer.Get<DTYPE_GRAD>();
        ExpAvgTensor = ExpAvgBuffer.Get<DTYPE_EXP_AVG>();
        ExpAvgSqTensor = ExpAvgSqBuffer.Get<DTYPE_EXP_AVG_SQ>();
        ComputeTensor = ComputeBuffer.Get<float>();
        Compute2Tensor = Compute2Buffer.Get<float>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tensorIdx = 0; tensorIdx < tensorNum; tensorIdx++) {
            ProcessSingleTensor(tensorIdx);
        }
    }

    __aicore__ inline void ComputeMicro(uint32_t len, float curStepSize)
    {
        // Original SIMD compute with 6 UB buffers
        if (weightDecay != ZERO_FLOAT_VALUE) {
            Muls(ComputeTensor, ParamTensor, weightDecay, len);
            Add(GradTensor, GradTensor, ComputeTensor, len);
        }
        Muls(ComputeTensor, ExpAvgTensor, beta1, len);
        Muls(Compute2Tensor, GradTensor, oneMinusBeta1, len);
        Add(ExpAvgTensor, ComputeTensor, Compute2Tensor, len);
        Mul(ComputeTensor, GradTensor, GradTensor, len);
        Muls(ExpAvgSqTensor, ExpAvgSqTensor, beta2, len);
        Muls(Compute2Tensor, ComputeTensor, oneMinusBeta2, len);
        Add(ExpAvgSqTensor, ExpAvgSqTensor, Compute2Tensor, len);
        Sqrt(ComputeTensor, ExpAvgSqTensor, len);
        Muls(ComputeTensor, ComputeTensor, invBiasCorrection2, len);
        Adds(ComputeTensor, ComputeTensor, eps, len);
        Div(Compute2Tensor, ExpAvgTensor, ComputeTensor, len);
        Muls(Compute2Tensor, Compute2Tensor, curStepSize, len);
        Sub(ParamTensor, ParamTensor, Compute2Tensor, len);
    }

    __aicore__ inline void GetCoreTaskPartition(uint32_t tensorIdx, int32_t& taskNumPerCore,
                                                uint64_t& taskStartIndex)
    {
        uint32_t curNumLcore = tilingData->numLcores[tensorIdx];
        if (static_cast<uint32_t>(this->blockIndex) < curNumLcore) {
            taskNumPerCore = static_cast<int32_t>(tilingData->taskNumPerLcores[tensorIdx]);
            taskStartIndex = this->blockIndex * taskNumPerCore;
        } else {
            taskNumPerCore = static_cast<int32_t>(tilingData->taskNumPerScores[tensorIdx]);
            taskStartIndex = static_cast<uint64_t>(curNumLcore) * tilingData->taskNumPerLcores[tensorIdx] +
                (this->blockIndex - curNumLcore) * taskNumPerCore;
        }
    }

    __aicore__ inline void SetTensorGlobalBuffers(uint32_t tensorIdx, uint32_t paramNum)
    {
        // Get GM pointers for current tensor in each TensorList
        paramGM.SetGlobalBuffer(paramListDesc.GetDataPtr<DTYPE_PARAM>(tensorIdx), paramNum);
        gradGM.SetGlobalBuffer(gradListDesc.GetDataPtr<DTYPE_GRAD>(tensorIdx), paramNum);
        expAvgGM.SetGlobalBuffer(expAvgListDesc.GetDataPtr<DTYPE_EXP_AVG>(tensorIdx), paramNum);
        expAvgSqGM.SetGlobalBuffer(expAvgSqListDesc.GetDataPtr<DTYPE_EXP_AVG_SQ>(tensorIdx), paramNum);
        paramOutGM.SetGlobalBuffer(paramOutListDesc.GetDataPtr<DTYPE_PARAM_OUT>(tensorIdx), paramNum);
        expAvgOutGM.SetGlobalBuffer(expAvgOutListDesc.GetDataPtr<DTYPE_EXP_AVG_OUT>(tensorIdx), paramNum);
        expAvgSqOutGM.SetGlobalBuffer(expAvgSqOutListDesc.GetDataPtr<DTYPE_EXP_AVG_SQ_OUT>(tensorIdx), paramNum);
    }

    __aicore__ inline void CopyInChunk(uint64_t gmOffset, uint32_t copyLen, uint32_t validLen, bool isTailChunk)
    {
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        if (isTailChunk) {
            // Tail chunk: validLen is not 8-aligned; DataCopyPad with the exact
            // valid byte length avoids reading past the tensor end.
            DataCopyExtParams tailCopyInParams{1, validLen * FLOAT_SIZE, 0, 0, 0};
            DataCopyPadExtParams<float> tailNoPadParams{false, 0, 0, 0.0f};
            DataCopyPad(ParamTensor, paramGM[gmOffset], tailCopyInParams, tailNoPadParams);
            DataCopyPad(GradTensor, gradGM[gmOffset], tailCopyInParams, tailNoPadParams);
            DataCopyPad(ExpAvgTensor, expAvgGM[gmOffset], tailCopyInParams, tailNoPadParams);
            DataCopyPad(ExpAvgSqTensor, expAvgSqGM[gmOffset], tailCopyInParams, tailNoPadParams);
        } else {
            DataCopy(ParamTensor, paramGM[gmOffset], copyLen);
            DataCopy(GradTensor, gradGM[gmOffset], copyLen);
            DataCopy(ExpAvgTensor, expAvgGM[gmOffset], copyLen);
            DataCopy(ExpAvgSqTensor, expAvgSqGM[gmOffset], copyLen);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void CopyOutChunk(uint64_t gmOffset, uint32_t copyLen, uint32_t validLen, bool isTailChunk)
    {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        if (isTailChunk) {
            // Tail chunk: validLen is not 8-aligned; DataCopyPad with the exact
            // valid byte length avoids writing past the tensor end.
            DataCopyExtParams tailCopyOutParams{1, validLen * FLOAT_SIZE, 0, 0, 0};
            DataCopyPad(paramOutGM[gmOffset], ParamTensor, tailCopyOutParams);
            DataCopyPad(expAvgOutGM[gmOffset], ExpAvgTensor, tailCopyOutParams);
            DataCopyPad(expAvgSqOutGM[gmOffset], ExpAvgSqTensor, tailCopyOutParams);
        } else {
            DataCopy(paramOutGM[gmOffset], ParamTensor, copyLen);
            DataCopy(expAvgOutGM[gmOffset], ExpAvgTensor, copyLen);
            DataCopy(expAvgSqOutGM[gmOffset], ExpAvgSqTensor, copyLen);
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void ProcessSingleTensor(uint32_t tensorIdx)
    {
        uint32_t curParamNum = tilingData->paramNums[tensorIdx];
        uint32_t curTailNum = tilingData->tailNums[tensorIdx];
        uint32_t curBlockDim = tilingData->blockDims[tensorIdx];
        int32_t curTaskNumPerLoop = tilingData->taskNumPerLoops[tensorIdx];
        float curStepSize = tilingData->stepSizes[tensorIdx];

        if (curBlockDim == 0 || curTaskNumPerLoop == 0) {
            return;
        }

        // Skip if this core is not needed for this tensor
        if (static_cast<uint32_t>(this->blockIndex) >= curBlockDim) {
            return;
        }

        int32_t curTaskNumPerCore;
        uint64_t curTaskStartIndex;
        GetCoreTaskPartition(tensorIdx, curTaskNumPerCore, curTaskStartIndex);
        int32_t curTaskLoop = (curTaskNumPerCore + curTaskNumPerLoop - 1) / curTaskNumPerLoop;

        SetTensorGlobalBuffers(tensorIdx, curParamNum);

        for (int32_t loopIdx = 0; loopIdx < curTaskLoop; loopIdx++) {
            int32_t curTaskNumPerCurLoop;
            if (loopIdx == curTaskLoop - 1) {
                curTaskNumPerCurLoop = curTaskNumPerCore - loopIdx * curTaskNumPerLoop;
            } else {
                curTaskNumPerCurLoop = curTaskNumPerLoop;
            }

            uint64_t taskCopyinIndex = curTaskStartIndex + loopIdx * curTaskNumPerLoop;
            uint32_t copyLen = static_cast<uint32_t>(curTaskNumPerCurLoop);

            // The chunk holding the tensor's real end (only the last active
            // core's last loop when tailNum != 0): validLen is not 8-aligned
            // and copyLen would read past the tensor.
            bool isTailChunk = (loopIdx == curTaskLoop - 1 && curTailNum != 0 &&
                                static_cast<uint32_t>(this->blockIndex) == curBlockDim - 1);
            uint32_t validLen = isTailChunk ? copyLen - curTailNum : copyLen;

            CopyInChunk(taskCopyinIndex, copyLen, validLen, isTailChunk);

            // Compute: MicroAPI register-based fused computation
            ComputeMicro(copyLen, curStepSize);

            CopyOutChunk(taskCopyinIndex, copyLen, validLen, isTailChunk);
        }
    }

private:
    TPipe *_pipe;
    TBuf<TPosition::VECCALC> ParamBuffer, GradBuffer, ExpAvgBuffer, ExpAvgSqBuffer, ComputeBuffer, Compute2Buffer;
    LocalTensor<float> ParamTensor, GradTensor, ExpAvgTensor, ExpAvgSqTensor, ComputeTensor, Compute2Tensor;

    GlobalTensor<DTYPE_PARAM> paramGM;
    GlobalTensor<DTYPE_GRAD> gradGM;
    GlobalTensor<DTYPE_EXP_AVG> expAvgGM;
    GlobalTensor<DTYPE_EXP_AVG_SQ> expAvgSqGM;
    GlobalTensor<DTYPE_PARAM_OUT> paramOutGM;
    GlobalTensor<DTYPE_EXP_AVG_OUT> expAvgOutGM;
    GlobalTensor<DTYPE_EXP_AVG_SQ_OUT> expAvgSqOutGM;

    ListTensorDesc paramListDesc;
    ListTensorDesc gradListDesc;
    ListTensorDesc expAvgListDesc;
    ListTensorDesc expAvgSqListDesc;
    ListTensorDesc paramOutListDesc;
    ListTensorDesc expAvgOutListDesc;
    ListTensorDesc expAvgSqOutListDesc;

    const FusedAdamMultiLrTilingData *tilingData;
    uint32_t tensorNum;
    uint64_t blockIndex;
    float beta1, beta2, eps, weightDecay, oneMinusBeta1, oneMinusBeta2, invBiasCorrection2;
};

extern "C" __global__ __aicore__ void fused_adam_multi_lr(GM_ADDR param, GM_ADDR grad, GM_ADDR exp_avg, GM_ADDR exp_avg_sq,
                                            GM_ADDR param_out, GM_ADDR exp_avg_out, GM_ADDR exp_avg_sq_out,
                                            GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        FusedAdamMultiLr op;
        op.Init(param, grad, exp_avg, exp_avg_sq, param_out, exp_avg_out, exp_avg_sq_out, &tiling_data, &pipe);
        op.Process();
    }
}