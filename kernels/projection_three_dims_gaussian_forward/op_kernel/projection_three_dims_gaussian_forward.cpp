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

constexpr int32_t FLOAT_SIZE = 4;
constexpr int32_t BROADCAST_DIM = 2;
constexpr uint32_t VIEWMATS_PARAMS_NUM = 12;
constexpr uint32_t KS_PARAMS_NUM = 6;

constexpr uint32_t MEANS_TAIL_DIM = 3;
constexpr uint32_t COVARS_TAIL_DIM = 9;
constexpr uint32_t VIEWMATS_TAIL_DIM = 16;
constexpr uint32_t KS_TAIL_DIM = 9;
constexpr uint32_t MEANS2D_TAIL_DIM = 2;
constexpr uint32_t CONICS_TAIL_DIM = 3;
constexpr uint32_t RADIUS_TAIL_DIM = 2;
constexpr uint32_t COVARS2D_TAIL_DIM = 3;
constexpr uint32_t COMPUTE_BUFFER_NUM = 12;
constexpr uint32_t CAMREA_PARAM_BUFFER_NUM = 18;

constexpr uint32_t MEANS_OFFSET_2 = 2;
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
constexpr uint32_t COMPUTE_OFFSET_9 = 9;
constexpr uint32_t COMPUTE_OFFSET_10 = 10;
constexpr uint32_t COMPUTE_OFFSET_11 = 11;

constexpr uint32_t VIEWMATS_02 = 2;
constexpr uint32_t VIEWMATS_03 = 3;
constexpr uint32_t VIEWMATS_10 = 4;
constexpr uint32_t VIEWMATS_11 = 5;
constexpr uint32_t VIEWMATS_12 = 6;
constexpr uint32_t VIEWMATS_13 = 7;
constexpr uint32_t VIEWMATS_20 = 8;
constexpr uint32_t VIEWMATS_21 = 9;
constexpr uint32_t VIEWMATS_22 = 10;
constexpr uint32_t VIEWMATS_23 = 11;

constexpr uint32_t KS_00 = 12;
constexpr uint32_t KS_01 = 13;
constexpr uint32_t KS_02 = 14;
constexpr uint32_t KS_10 = 15;
constexpr uint32_t KS_11 = 16;
constexpr uint32_t KS_12 = 17;

constexpr uint64_t CAMREA_PARAM_COPY_LENGTH = 16;
constexpr float ZERO_FLOAT_VALUE = 0.0f;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float BOUND_COEFF_PARAM = 0.15f;
constexpr float DET_MINIMUM_VALUE = 1e-10f;
constexpr float RADIUS_COEFF = 3.33f;
constexpr float EPSILON = 0.01f;

class ProjectionThreeDimsGaussianForward {
public:
    __aicore__ inline ProjectionThreeDimsGaussianForward()
    {}

    __aicore__ inline void GetTilingData(const ProjectionThreeDimsGaussianForwardTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "Block Dim can not be Zero!");
        this->blockIndex = GetBlockIdx();

        imageWidth = tiling_data->imageWidth;
        imageHeight = tiling_data->imageHeight;
        cameraModel = tiling_data->cameraModel;
        batchSizeNum = tiling_data->batchSizeNum;
        cameraNum = tiling_data->cameraNum;
        gaussianNum = tiling_data->gaussianNum;
        totalTaskNum = tiling_data->totalTaskNum;
        tailNum = tiling_data->tailNum;
        taskNumPerScore = tiling_data->taskNumPerScore;
        taskNumPerLcore = tiling_data->taskNumPerLcore;
        numScore = tiling_data->numScore;
        numLcore = tiling_data->numLcore;
        blockDim = tiling_data->blockDim;
        taskNumPerLoop = tiling_data->taskNumPerLoop;
        eps2d = tiling_data->eps2d;
        ubTotalSize = tiling_data->ubTotalSize;
        calcCompensations = tiling_data->calcCompensations;
    }

    __aicore__ inline void GMInit(GM_ADDR means, GM_ADDR covars, GM_ADDR viewmats, GM_ADDR ks, \
                                  GM_ADDR means2d, GM_ADDR depths, GM_ADDR conics, GM_ADDR compensations, \
                                  GM_ADDR det, GM_ADDR radius, GM_ADDR covars2d)
    {
        bnSize = batchSizeNum * gaussianNum;
        bcSize = batchSizeNum * cameraNum;
        bcnSize = batchSizeNum * cameraNum * gaussianNum;

        meansGM.SetGlobalBuffer((__gm__ DTYPE_MEANS *)means, bnSize * MEANS_TAIL_DIM);
        covarsGM.SetGlobalBuffer((__gm__ DTYPE_COVARS *)covars, bnSize * COVARS_TAIL_DIM);
        viewmatsGM.SetGlobalBuffer((__gm__ DTYPE_VIEWMATS *)viewmats, bcSize * VIEWMATS_TAIL_DIM);
        ksGM.SetGlobalBuffer((__gm__ DTYPE_KS *)ks, bcSize * KS_TAIL_DIM);
        means2dGM.SetGlobalBuffer((__gm__ DTYPE_MEANS2D *)means2d, bcnSize * MEANS2D_TAIL_DIM);
        depthsGM.SetGlobalBuffer((__gm__ DTYPE_DEPTHS *)depths, bcnSize);
        conicsGM.SetGlobalBuffer((__gm__ DTYPE_CONICS *)conics, bcnSize * CONICS_TAIL_DIM);
        compensationsGM.SetGlobalBuffer((__gm__ DTYPE_COMPENSATIONS *)compensations, bcnSize);
        detGM.SetGlobalBuffer((__gm__ DTYPE_DET *)det, bcnSize);
        radiusGM.SetGlobalBuffer((__gm__ DTYPE_RADIUS *)radius, bcnSize * RADIUS_TAIL_DIM);
        covars2dGM.SetGlobalBuffer((__gm__ DTYPE_COVARS2D *)covars2d, bcnSize * COVARS_TAIL_DIM);
    }

    __aicore__ inline void Init(TPipe* pipe)
    {
        if (this->blockIndex < numLcore) {
            taskNumPerCore = taskNumPerLcore;
            taskStartIndex = this->blockIndex * taskNumPerCore;
        } else {
            taskNumPerCore = taskNumPerScore;
            taskStartIndex = numLcore * taskNumPerLcore + (this->blockIndex - numLcore) * taskNumPerCore;
        }
        taskLoop = static_cast<int32_t>((taskNumPerCore + taskNumPerLoop - 1) / taskNumPerLoop);

        uint64_t BaseBufferSize = taskNumPerLoop * FLOAT_SIZE;
        this->_pipe = pipe;

        this->_pipe->InitBuffer(MeansTensorBuffer, BaseBufferSize * MEANS_TAIL_DIM);
        this->_pipe->InitBuffer(CovarsTensorBuffer, BaseBufferSize * COVARS_TAIL_DIM);
        this->_pipe->InitBuffer(ComputingTensorBuffer, BaseBufferSize * COMPUTE_BUFFER_NUM);
        this->_pipe->InitBuffer(CompareTensorBuffer, taskNumPerLoop);
        this->_pipe->InitBuffer(CameraParamTensorBuffer, BaseBufferSize * CAMREA_PARAM_BUFFER_NUM);

        MeansTensor = MeansTensorBuffer.Get<float>();
        CovarsTensor = CovarsTensorBuffer.Get<float>();
        ComputingTensor = ComputingTensorBuffer.Get<float>();
        CompareTensor = CompareTensorBuffer.Get<uint8_t>();
        CameraParamTensor = CameraParamTensorBuffer.Get<float>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIndex = 0; batchIndex < batchSizeNum; batchIndex++) {
            uint64_t batchCopyinIndex = batchIndex * gaussianNum;
            uint64_t batchCopyoutIndex = batchIndex * cameraNum * gaussianNum;

            for (uint32_t cameraIndex = 0; cameraIndex < cameraNum; cameraIndex++) {
                uint64_t cameraCopyinIndex = batchIndex * cameraNum + cameraIndex;
                uint64_t cameraCopyoutIndex = cameraIndex * gaussianNum;
                CopyInCameraParams(cameraCopyinIndex);

                for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                    ProcessSingleLoop(batchCopyinIndex, batchCopyoutIndex, cameraCopyoutIndex, taskLoopIndex);
                }
            }
        }
    }

    __aicore__ inline void ProcessSingleLoop(uint64_t batchCopyinIndex, uint64_t batchCopyoutIndex, \
                                             uint64_t cameraCopyoutIndex, uint32_t taskLoopIndex)
    {
        uint64_t taskCopyIndex = taskStartIndex + taskLoopIndex * taskNumPerLoop;
        ComputingTaskNum(taskLoopIndex);
        CopyIn(batchCopyinIndex, taskCopyIndex); // 搬入means等变量
        ComputingMeansCamera(); // 世界坐标系->相机坐标系
        ComputingCovarsCamera();
        if (cameraModel == 0) {
            PerspProjection();
        }
        ComputingDet();
        ComputingConics();
        ComputingRadius();

        if (taskLoopIndex == taskLoop - 1 && tailNum != 0 && this->blockIndex == blockDim - 1) {
            ProcessDirtyData();
            SetAtomicAdd<float>();
            CopyOut(batchCopyoutIndex, cameraCopyoutIndex, taskCopyIndex); // 尾核&非对齐&最后一次循环时搬出
            SetAtomicNone();
        } else {
            CopyOut(batchCopyoutIndex, cameraCopyoutIndex, taskCopyIndex); // 非尾核或对齐场景搬出
        }
    }

    __aicore__ inline void CopyInCameraParams(uint64_t cameraCopyinIndex)
    {
        uint32_t viewmatsSrcShape[2] = {VIEWMATS_PARAMS_NUM, 1};
        uint32_t viewmatsDstShape[2] = {VIEWMATS_PARAMS_NUM, taskNumPerLoop};
        uint32_t ksSrcShape[2] = {KS_PARAMS_NUM, 1};
        uint32_t ksDstShape[2] = {KS_PARAMS_NUM, taskNumPerLoop};

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(ComputingTensor, viewmatsGM[cameraCopyinIndex * VIEWMATS_TAIL_DIM], \
                 CAMREA_PARAM_COPY_LENGTH);
        DataCopy(ComputingTensor[CAMREA_PARAM_COPY_LENGTH], ksGM[cameraCopyinIndex * KS_TAIL_DIM], \
                 CAMREA_PARAM_COPY_LENGTH);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        Broadcast<float, BROADCAST_DIM, 1, false>(CameraParamTensor, ComputingTensor, viewmatsDstShape, \
                                                  viewmatsSrcShape);
        Broadcast<float, BROADCAST_DIM, 1, false>(CameraParamTensor[VIEWMATS_PARAMS_NUM * taskNumPerLoop], \
                                                  ComputingTensor[CAMREA_PARAM_COPY_LENGTH], ksDstShape, ksSrcShape);
    }

    __aicore__ inline void ComputingTaskNum(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == taskLoop - 1) {
            taskNumPerCurLoop = taskNumPerCore - taskLoopIndex * taskNumPerLoop;
        } else {
            taskNumPerCurLoop = taskNumPerLoop;
        }
    }

    __aicore__ inline void CopyIn(uint64_t batchCopyinIndex, uint64_t taskCopyIndex)
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        for (int32_t i = 0; i < MEANS_TAIL_DIM; i++) {
            DataCopy(MeansTensor[taskNumPerCurLoop * i], \
                     meansGM[batchCopyinIndex * MEANS_TAIL_DIM + i * gaussianNum + taskCopyIndex], \
                     taskNumPerCurLoop);
        }
        for (int32_t i = 0; i < COVARS_TAIL_DIM; i++) {
            DataCopy(CovarsTensor[taskNumPerCurLoop * i], \
                     covarsGM[batchCopyinIndex * COVARS_TAIL_DIM + i * gaussianNum + taskCopyIndex], \
                     taskNumPerCurLoop);
        }

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }

    __aicore__ inline void ComputingMeansCamera()
    {
        Mul(ComputingTensor, MeansTensor, CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop], MeansTensor[taskNumPerCurLoop], \
            CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor, CameraParamTensor[taskNumPerLoop * VIEWMATS_03], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop], MeansTensor, \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            MeansTensor[taskNumPerCurLoop], CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], CameraParamTensor[taskNumPerLoop * VIEWMATS_12], \
            taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_13], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], MeansTensor, \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MeansTensor[taskNumPerCurLoop], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_23], taskNumPerCurLoop);
        Muls(MeansTensor, ComputingTensor, ONE_FLOAT_VALUE, taskNumPerCurLoop * MEANS_TAIL_DIM);
    }

    __aicore__ inline void ComputingCovarsCamera()
    {
        // 计算tmp_covars
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor, CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            taskNumPerCurLoop); // tmp_covars_c0

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor[taskNumPerCurLoop], CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c1

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c2

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor, CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_12], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c3

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor[taskNumPerCurLoop], CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_12], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c4

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_12], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c5

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor, CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c6

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor[taskNumPerCurLoop], CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c7

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // tmp_covars_c8

        // 计算covars_c
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor, CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop], CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(CovarsTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor, CovarsTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            taskNumPerCurLoop); // covars_c0

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor, CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop], CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_12], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c1

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor, CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop], CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c2

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c3

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_12], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c4

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c5

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], CameraParamTensor, taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            CameraParamTensor[taskNumPerLoop], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_02], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c6

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_12], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c7

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_20], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_21], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            CameraParamTensor[taskNumPerLoop * VIEWMATS_22], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop); // covars_c8
    }

    __aicore__ inline void PerspProjection()
    {
        ComputingMeans2D();
        ComputingCovParams();
        ComputingCovars2D();
    }

    __aicore__ inline void ComputingMeans2D()
    {
        // 计算means2d
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], MeansTensor, \
            CameraParamTensor[taskNumPerLoop * KS_00], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], MeansTensor[taskNumPerCurLoop], \
            CameraParamTensor[taskNumPerLoop * KS_01], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * KS_02], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Add(ComputingTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        Div(ComputingTensor, ComputingTensor, MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], MeansTensor, \
            CameraParamTensor[taskNumPerLoop * KS_10], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], MeansTensor[taskNumPerCurLoop], \
            CameraParamTensor[taskNumPerLoop * KS_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * KS_12], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop);
        Div(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingCovParams()
    {
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], CameraParamTensor[taskNumPerLoop * KS_02], \
            CameraParamTensor[taskNumPerLoop * KS_00], taskNumPerCurLoop);
        Duplicate(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ONE_FLOAT_VALUE, taskNumPerCurLoop);
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop * KS_00], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], imageWidth, taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], BOUND_COEFF_PARAM, taskNumPerCurLoop);

        // lim_x_neg
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], (-1.0f), taskNumPerCurLoop);

        // lim_x_pos
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);

        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            CameraParamTensor[taskNumPerLoop * KS_12], CameraParamTensor[taskNumPerLoop * KS_11], taskNumPerCurLoop);
        Duplicate(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], ONE_FLOAT_VALUE, taskNumPerCurLoop);
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            CameraParamTensor[taskNumPerLoop * KS_11], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], imageHeight, taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], BOUND_COEFF_PARAM, taskNumPerCurLoop);

        // lim_y_neg
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], (-1.0f), taskNumPerCurLoop);
        // lim_y_pos
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);

        // 计算tx, ty, tz2
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], MeansTensor, \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop);
        Compare(CompareTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CMPMODE::GT, taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], CompareTensor, \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);
        Compare(CompareTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], CMPMODE::LT, taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], CompareTensor, \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);
        Mul(MeansTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop); // tx

        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], MeansTensor[taskNumPerCurLoop], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop);
        Compare(CompareTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], CMPMODE::GT, taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], CompareTensor, \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);
        Compare(CompareTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], CMPMODE::LT, taskNumPerCurLoop);
        Select(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], CompareTensor, \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);
        Mul(MeansTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop); // ty

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop); // tz2

        // 计算a1, a2, b1, b2
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], CameraParamTensor[taskNumPerLoop * KS_00], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop); // a1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], MeansTensor, \
            CameraParamTensor[taskNumPerLoop * KS_00], taskNumPerCurLoop);
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], (-1.0f), taskNumPerCurLoop); // a2
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            CameraParamTensor[taskNumPerLoop * KS_11], \
            MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop); // b1
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], MeansTensor[taskNumPerCurLoop], \
            CameraParamTensor[taskNumPerLoop * KS_11], taskNumPerCurLoop);
        Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], (-1.0f), taskNumPerCurLoop); // b2
    }

    __aicore__ inline void ComputingCovars2D()
    {
        // 计算cov
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor, \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Add(CovarsTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);

        // 计算cov2d
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            CovarsTensor, taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], taskNumPerCurLoop);

        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            CovarsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], taskNumPerCurLoop);

        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], taskNumPerCurLoop);

        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], taskNumPerCurLoop);
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
            CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_7], taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingDet()
    {
        // det_orig
        if (calcCompensations) {
            Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop);
            Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
            Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        }

        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], eps2d, taskNumPerCurLoop);
        Adds(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], eps2d, taskNumPerCurLoop);

        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], DET_MINIMUM_VALUE, taskNumPerCurLoop);

        if (calcCompensations) {
            Div(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_2], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
            ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], ZERO_FLOAT_VALUE, taskNumPerCurLoop);
            Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        }
    }

    __aicore__ inline void ComputingConics()
    {
        Div(CovarsTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Add(CovarsTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_10], taskNumPerCurLoop);
        Div(CovarsTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        Muls(CovarsTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop], float(-0.5f), taskNumPerCurLoop);
        Div(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
    }

    __aicore__ inline void ComputingRadius()
    {
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop);
        Muls(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
             float(0.5f), taskNumPerCurLoop); // b
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Sub(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);
        ClampMin(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], EPSILON, taskNumPerCurLoop);
        Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop); // tmp
        Add(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_5], taskNumPerCurLoop);
        Sqrt(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_6], taskNumPerCurLoop); // r1(无系数)

        Sqrt(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);
        Sqrt(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
             ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], taskNumPerCurLoop);
        Compare(CompareTensor, CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], CMPMODE::LT, taskNumPerCurLoop);
        Select(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], CompareTensor, \
               CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);
        Compare(CompareTensor, CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], CMPMODE::LT, taskNumPerCurLoop);
        Select(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], CompareTensor, \
               CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
               ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], \
               SELMODE::VSEL_TENSOR_TENSOR_MODE, taskNumPerCurLoop);

        Muls(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
             RADIUS_COEFF, taskNumPerCurLoop);
        Muls(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
             RADIUS_COEFF, taskNumPerCurLoop);

        Ceil(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_5], \
             taskNumPerCurLoop);
        Ceil(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_6], \
             taskNumPerCurLoop);
    }

    __aicore__ inline void ProcessDirtyData()
    {
        int32_t taskNumPerTailCore = taskNumPerCurLoop - tailNum;
        Duplicate(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], ONE_FLOAT_VALUE, taskNumPerCurLoop);
        for (int32_t i = 0; i < tailNum; i++) {
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4].SetValue(taskNumPerTailCore + i, ZERO_FLOAT_VALUE);
        }

        // 处理means2d
        Mul(ComputingTensor, ComputingTensor, \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop], ComputingTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);

        // 处理depth和det
        Mul(MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);

        // 处理covars2d
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);

        // 处理conics
        Mul(CovarsTensor, CovarsTensor, ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop], CovarsTensor[taskNumPerCurLoop], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);

        // 处理radius
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        Mul(CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], \
            ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);

        if (calcCompensations) {
            Mul(ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], \
                ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_4], taskNumPerCurLoop);
        }
    }

    __aicore__ inline void CopyOut(uint64_t batchCopyoutIndex, uint64_t cameraCopyoutIndex, uint64_t taskCopyIndex)
    {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        DataCopy(depthsGM[batchCopyoutIndex + cameraCopyoutIndex + taskCopyIndex], \
                 MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], taskNumPerCurLoop);
        DataCopy(detGM[batchCopyoutIndex + cameraCopyoutIndex + taskCopyIndex], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], taskNumPerCurLoop);

        DataCopy(radiusGM[(batchCopyoutIndex + cameraCopyoutIndex) * RADIUS_TAIL_DIM + taskCopyIndex], \
                 CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], taskNumPerCurLoop);
        DataCopy(radiusGM[(batchCopyoutIndex + cameraCopyoutIndex) * RADIUS_TAIL_DIM + taskCopyIndex + gaussianNum], \
                 CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], taskNumPerCurLoop);

        DataCopy(means2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * MEANS2D_TAIL_DIM + taskCopyIndex], \
                 ComputingTensor, taskNumPerCurLoop);
        DataCopy(means2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * MEANS2D_TAIL_DIM + taskCopyIndex + gaussianNum], \
                 ComputingTensor[taskNumPerCurLoop], taskNumPerCurLoop);

        DataCopy(conicsGM[(batchCopyoutIndex + cameraCopyoutIndex) * CONICS_TAIL_DIM + taskCopyIndex], \
                 CovarsTensor, taskNumPerCurLoop);
        DataCopy(conicsGM[(batchCopyoutIndex + cameraCopyoutIndex) * CONICS_TAIL_DIM + taskCopyIndex + gaussianNum], \
                 CovarsTensor[taskNumPerCurLoop], taskNumPerCurLoop);
        DataCopy(conicsGM[(batchCopyoutIndex + cameraCopyoutIndex) * CONICS_TAIL_DIM + taskCopyIndex + gaussianNum \
                 * COVARS_OFFSET_2], CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_2], taskNumPerCurLoop);

        DataCopy(covars2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * COVARS2D_TAIL_DIM + taskCopyIndex], \
                 ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], taskNumPerCurLoop);
        DataCopy(covars2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * COVARS2D_TAIL_DIM + taskCopyIndex + \
                 gaussianNum], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], taskNumPerCurLoop);
        DataCopy(covars2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * COVARS2D_TAIL_DIM + taskCopyIndex + \
                 gaussianNum * COVARS_OFFSET_2], ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], \
                 taskNumPerCurLoop);

        if (calcCompensations) {
            DataCopy(compensationsGM[batchCopyoutIndex + cameraCopyoutIndex + taskCopyIndex], \
                     ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], taskNumPerCurLoop);
        }

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> MeansTensorBuffer, CovarsTensorBuffer;
    TBuf <TPosition::VECCALC> ComputingTensorBuffer, CameraParamTensorBuffer;
    TBuf <TPosition::VECCALC> CompareTensorBuffer;

    LocalTensor<float> MeansTensor, CovarsTensor;
    LocalTensor<float> ComputingTensor, CameraParamTensor;
    LocalTensor<uint8_t> CompareTensor;

    GlobalTensor<DTYPE_MEANS> meansGM;
    GlobalTensor<DTYPE_COVARS> covarsGM;
    GlobalTensor<DTYPE_VIEWMATS> viewmatsGM;
    GlobalTensor<DTYPE_KS> ksGM;
    GlobalTensor<DTYPE_MEANS2D> means2dGM;
    GlobalTensor<DTYPE_DEPTHS> depthsGM;
    GlobalTensor<DTYPE_CONICS> conicsGM;
    GlobalTensor<DTYPE_COMPENSATIONS> compensationsGM;
    GlobalTensor<DTYPE_DET> detGM;
    GlobalTensor<DTYPE_RADIUS> radiusGM;
    GlobalTensor<DTYPE_COVARS2D> covars2dGM;

    uint32_t cameraModel;
    uint32_t batchSizeNum, cameraNum, gaussianNum, totalTaskNum, tailNum, taskNumPerScore, taskNumPerLcore;
    uint32_t numScore, numLcore, blockDim;
    uint64_t blockIndex, ubTotalSize, bnSize, bcSize, bcnSize;
    uint32_t taskNumPerLoop, taskNumPerCurLoop, taskNumPerCore, taskStartIndex, taskLoop;
    float eps2d, imageWidth, imageHeight;
    bool calcCompensations;
};

extern "C" __global__ __aicore__ void projection_three_dims_gaussian_forward(GM_ADDR means, GM_ADDR covars, \
                                                                             GM_ADDR viewmats, GM_ADDR ks, \
                                                                             GM_ADDR means2d, GM_ADDR depths, \
                                                                             GM_ADDR conics, GM_ADDR compensations, \
                                                                             GM_ADDR det, GM_ADDR radius, \
                                                                             GM_ADDR covars2d, GM_ADDR workspace, \
                                                                             GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        ProjectionThreeDimsGaussianForward op;
        op.GetTilingData(&tiling_data);
        op.GMInit(means, covars, viewmats, ks, means2d, depths, conics, compensations, det, radius, covars2d);
        op.Init(&pipe);
        op.Process();
    }
}