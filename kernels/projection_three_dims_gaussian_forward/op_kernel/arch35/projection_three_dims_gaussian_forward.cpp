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
constexpr uint32_t CAMERA_PARAM_SLOT_NUM = 64;   // camera-param UB slot: 32 used (16 viewmats + 16 ks), 64 for alignment headroom

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

// quat/scales fused-path constants
constexpr uint32_t QUAT_TAIL_DIM = 4;
constexpr uint32_t SCALES_TAIL_DIM = 3;
constexpr uint32_t QUAT_OFFSET_2 = 2;
constexpr uint32_t QUAT_OFFSET_3 = 3;
constexpr uint32_t SCALES_OFFSET_2 = 2;
constexpr float TWO_FLOAT_VALUE = 2.0f;
constexpr float ROTMAT_COEFF = -2.0f;         // rotation-matrix term 1 - 2(x^2+y^2)
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

constexpr uint32_t KS_00 = 16;  // ks region starts at UB[16], after the 16-float viewmats block
constexpr uint32_t KS_01 = 17;
constexpr uint32_t KS_02 = 18;
constexpr uint32_t KS_10 = 19;
constexpr uint32_t KS_11 = 20;
constexpr uint32_t KS_12 = 21;

constexpr uint64_t CAMREA_PARAM_COPY_LENGTH = 16;
constexpr uint64_t KS_COPY_LENGTH = 8;  // ks copy: 6 used scalars, 8-float/32B aligned, within the KS_TAIL_DIM(9) stride
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
        inputIsQuatScales = tiling_data->inputIsQuatScales;
    }

    __aicore__ inline void GMInit(GM_ADDR means, GM_ADDR covars, GM_ADDR viewmats, GM_ADDR ks, \
                                  GM_ADDR quat, GM_ADDR scales, \
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
        // Optional quat/scales GM, null on the covars-direct path. CopyIn
        // dispatches on inputIsQuatScales so the null is never dereferenced.
        // Tail dims: quat=4, scales=3.
        quatGM.SetGlobalBuffer((__gm__ DTYPE_QUAT *)quat, bnSize * 4);
        scalesGM.SetGlobalBuffer((__gm__ DTYPE_SCALES *)scales, bnSize * 3);
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
        this->_pipe->InitBuffer(CameraParamTensorBuffer, CAMERA_PARAM_SLOT_NUM * sizeof(float));
        this->_pipe->InitBuffer(MeansWorldTensorBuffer, BaseBufferSize * MEANS_TAIL_DIM);
        this->_pipe->InitBuffer(CovarsWorldTensorBuffer, BaseBufferSize * COVARS_TAIL_DIM);
        // Alternate output slots for the three largest outputs (conics,
        // covars2d, means2d), selected per camera by dbParity so one camera's
        // CopyOut can overlap the next camera's compute.
        this->_pipe->InitBuffer(ConicsDbBuffer,   BaseBufferSize * CONICS_TAIL_DIM);
        this->_pipe->InitBuffer(Covars2dDbBuffer, BaseBufferSize * COVARS2D_TAIL_DIM);
        this->_pipe->InitBuffer(Means2dDbBuffer,  BaseBufferSize * MEANS2D_TAIL_DIM);

        MeansTensor = MeansTensorBuffer.Get<float>();
        CovarsTensor = CovarsTensorBuffer.Get<float>();
        ComputingTensor = ComputingTensorBuffer.Get<float>();
        CameraParamTensor = CameraParamTensorBuffer.Get<float>();
        MeansWorldTensor = MeansWorldTensorBuffer.Get<float>();
        CovarsWorldTensor = CovarsWorldTensorBuffer.Get<float>();
        ConicsDbTensor   = ConicsDbBuffer.Get<float>();
        Covars2dDbTensor = Covars2dDbBuffer.Get<float>();
        Means2dDbTensor  = Means2dDbBuffer.Get<float>();

        // Time-multiplex Quat/ScalesTensor onto CovarsTensor's UB region rather
        // than allocating dedicated buffers: CovarsTensor (9 floats/task) is the
        // per-camera working buffer, unused during CopyIn and the quat/scales
        // conversion, and quat(4)+scales(3) fit within its 9. This keeps the
        // fused-path tile the same size as the covars-direct path.
        if (inputIsQuatScales == 1) {
            QuatTensor   = CovarsTensor;
            ScalesTensor = CovarsTensor[QUAT_TAIL_DIM * taskNumPerLoop];
        }
    }

    __aicore__ inline void Process()
    {
        // Loop order: taskLoop outer, camera inner. World means/covars depend
        // only on (batch, taskLoop), so hoisting CopyIn above the camera loop
        // loads them once per taskLoop instead of once per camera, removing the
        // redundant per-camera world-input load.
        for (uint32_t batchIndex = 0; batchIndex < batchSizeNum; batchIndex++) {
            uint64_t batchCopyinIndex = batchIndex * gaussianNum;
            uint64_t batchCopyoutIndex = batchIndex * cameraNum * gaussianNum;

            for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                uint64_t taskCopyIndex = taskStartIndex + taskLoopIndex * taskNumPerLoop;
                ComputingTaskNum(taskLoopIndex);
                CopyIn(batchCopyinIndex, taskCopyIndex); // load world-frame means/covars, once per taskLoop

                for (uint32_t cameraIndex = 0; cameraIndex < cameraNum; cameraIndex++) {
                    dbParity = cameraIndex & 1u;  // alternate output buffer each camera
                    uint64_t cameraCopyinIndex = batchIndex * cameraNum + cameraIndex;
                    uint64_t cameraCopyoutIndex = cameraIndex * gaussianNum;
                    ProcessSingleCamera(cameraCopyinIndex, batchCopyoutIndex, cameraCopyoutIndex, taskCopyIndex);
                }
            }
        }
    }

    __aicore__ inline void ProcessSingleCamera(uint64_t cameraCopyinIndex, uint64_t batchCopyoutIndex, \
                                               uint64_t cameraCopyoutIndex, uint64_t taskCopyIndex)
    {
        // Per-camera projection. World means/covars are already resident in the
        // MeansWorld/CovarsWorld buffers (loaded once per taskLoop by Process).
        CopyInCameraParams(cameraCopyinIndex);
        ComputingMeansCamera(); // world -> camera frame
        ComputingCovarsCamera();
        if (cameraModel == 0) {
            PerspProjection();
        }
        ComputingDet();
        ComputingConics();
        ComputingRadius();

        // CopyOut is bounded to actualValidCount, so writes never spill into
        // the next GM tensor.
        CopyOut(batchCopyoutIndex, cameraCopyoutIndex, taskCopyIndex);
    }

    // Single-camera path: camera-major loop order. With cameraNum == 1 the
    // taskLoop-major interchange has no cross-camera redundancy to remove and
    // only adds scheduling overhead, so this path keeps
    // `for camera { CopyInCameraParams; for taskLoop { ... } }`.
    __aicore__ inline void ProcessCameraMajor()
    {
        for (uint32_t batchIndex = 0; batchIndex < batchSizeNum; batchIndex++) {
            uint64_t batchCopyinIndex = batchIndex * gaussianNum;
            uint64_t batchCopyoutIndex = batchIndex * cameraNum * gaussianNum;

            for (uint32_t cameraIndex = 0; cameraIndex < cameraNum; cameraIndex++) {
                uint64_t cameraCopyinIndex = batchIndex * cameraNum + cameraIndex;
                uint64_t cameraCopyoutIndex = cameraIndex * gaussianNum;
                CopyInCameraParams(cameraCopyinIndex);

                for (uint32_t taskLoopIndex = 0; taskLoopIndex < taskLoop; taskLoopIndex++) {
                    uint64_t taskCopyIndex = taskStartIndex + taskLoopIndex * taskNumPerLoop;
                    ComputingTaskNum(taskLoopIndex);
                    CopyIn(batchCopyinIndex, taskCopyIndex); // load world-frame means/covars
                    ComputingMeansCamera(); // world -> camera frame
                    ComputingCovarsCamera();
                    if (cameraModel == 0) {
                        PerspProjection();
                    }
                    ComputingDet();
                    ComputingConics();
                    ComputingRadius();
                    CopyOut(batchCopyoutIndex, cameraCopyoutIndex, taskCopyIndex);
                }
            }
        }
    }

    __aicore__ inline void CopyInCameraParams(uint64_t cameraCopyinIndex)
    {
        // Load the scalar camera params directly to CameraParamTensor.
        // UB layout: [0..15] viewmats (12 used + 4 pad), [16..31] ks (6 used at KS_*).
        // Readers broadcast each scalar to 64 lanes via LoadDist::DIST_BRC_B32,
        // so the per-task camera buffer collapses to 32 scalars.
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);

        DataCopy(CameraParamTensor, viewmatsGM[cameraCopyinIndex * VIEWMATS_TAIL_DIM], \
                 CAMREA_PARAM_COPY_LENGTH);
        // ks GM row stride is KS_TAIL_DIM(9); copying 16 floats would overrun
        // into the next camera's ks and past the tensor tail. KS_COPY_LENGTH(8)
        // covers the 6 used ks scalars, stays within the 9-float row, and keeps
        // the copy 32-byte aligned.
        DataCopy(CameraParamTensor[CAMREA_PARAM_COPY_LENGTH], ksGM[cameraCopyinIndex * KS_TAIL_DIM], \
                 KS_COPY_LENGTH);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void ComputingTaskNum(uint32_t taskLoopIndex)
    {
        if (taskLoopIndex == taskLoop - 1) {
            taskNumPerCurLoop = taskNumPerCore - taskLoopIndex * taskNumPerLoop;
        } else {
            taskNumPerCurLoop = taskNumPerLoop;
        }
        // actualValidCount = real gaussians in this tile. Only the last loop of
        // the last core can have padding (tailNum > 0); elsewhere it equals
        // taskNumPerCurLoop.
        if (taskLoopIndex == taskLoop - 1 && tailNum != 0 && this->blockIndex == blockDim - 1) {
            actualValidCount = taskNumPerCurLoop - tailNum;
        } else {
            actualValidCount = taskNumPerCurLoop;
        }
    }

    // Inlined quat_scales_to_covars (vector MicroAPI, register-resident).
    // Reads QuatTensor (4 floats/task) + ScalesTensor (3 floats/task) from UB
    // and computes the world-frame covariance Sigma = R*diag(s^2)*R^T via
    // M = R*diag(s), Sigma = M @ M^T, writing 9 floats/task to CovarsWorldTensor
    // in row-major 3x3 layout -- identical to the covars-direct CopyIn path, so
    // downstream scopes consume it the same way.
    __aicore__ inline void ComputeCovarsFromQuatScales()
    {
        __ubuf__ float* p_quat   = (__ubuf__ float*) QuatTensor.GetPhyAddr();
        __ubuf__ float* p_scales = (__ubuf__ float*) ScalesTensor.GetPhyAddr();
        __ubuf__ float* p_covars = (__ubuf__ float*) CovarsWorldTensor.GetPhyAddr();

        constexpr uint16_t LANE = 64;
        uint16_t lanesPerChunk = static_cast<uint16_t>(
            (taskNumPerCurLoop + LANE - 1) / LANE);

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            // Quat / scales(7) + q-products(9) + rotmat/M(9) + scratch(3) = 28 RegTensors
            // declared once per scope; reused across lanes.
            AscendC::MicroAPI::RegTensor<float> qw, qx, qy, qz;
            AscendC::MicroAPI::RegTensor<float> sx, sy, sz;
            AscendC::MicroAPI::RegTensor<float> qx2, qy2, qz2, qxy, qxz, qwy, qwz, qyz, qxw;
            AscendC::MicroAPI::RegTensor<float> r00, r01, r02, r10, r11, r12, r20, r21, r22;
            AscendC::MicroAPI::RegTensor<float> tmp, acc, qn;

            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off = static_cast<uint32_t>(lane) * LANE;

                // === Load quat(4) + scales(3) into regs ===
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    qw, p_quat + 0 * taskNumPerCurLoop + off, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    qx, p_quat + 1 * taskNumPerCurLoop + off, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    qy, p_quat + QUAT_OFFSET_2 * taskNumPerCurLoop + off, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    qz, p_quat + QUAT_OFFSET_3 * taskNumPerCurLoop + off, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    sx, p_scales + 0 * taskNumPerCurLoop + off, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    sy, p_scales + 1 * taskNumPerCurLoop + off, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    sz, p_scales + SCALES_OFFSET_2 * taskNumPerCurLoop + off, 1, mask);

                // === NormalizeQuats: |q|=sqrt(qw^2+qx^2+qy^2+qz^2); q /= |q| ===
                AscendC::MicroAPI::Mul(qn, qw, qw, mask);
                AscendC::MicroAPI::Mul(tmp, qx, qx, mask);
                AscendC::MicroAPI::Add(qn, qn, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, qy, qy, mask);
                AscendC::MicroAPI::Add(qn, qn, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, qz, qz, mask);
                AscendC::MicroAPI::Add(qn, qn, tmp, mask);
                AscendC::MicroAPI::Sqrt(qn, qn, mask);
                AscendC::MicroAPI::Div(qw, qw, qn, mask);
                AscendC::MicroAPI::Div(qx, qx, qn, mask);
                AscendC::MicroAPI::Div(qy, qy, qn, mask);
                AscendC::MicroAPI::Div(qz, qz, qn, mask);

                // === Q-products: x2,y2,z2,xy,xz,wy,wz,yz,xw -> 9 regs ===
                AscendC::MicroAPI::Mul(qx2, qx, qx, mask);
                AscendC::MicroAPI::Mul(qy2, qy, qy, mask);
                AscendC::MicroAPI::Mul(qz2, qz, qz, mask);
                AscendC::MicroAPI::Mul(qxy, qx, qy, mask);
                AscendC::MicroAPI::Mul(qxz, qx, qz, mask);
                AscendC::MicroAPI::Mul(qwy, qw, qy, mask);
                AscendC::MicroAPI::Mul(qwz, qw, qz, mask);
                AscendC::MicroAPI::Mul(qyz, qy, qz, mask);
                AscendC::MicroAPI::Mul(qxw, qx, qw, mask);

                // === Build R into r-regs (row-major 3x3) ===
                // R[0,0] = 1 - 2(qy^2 + qz^2)
                AscendC::MicroAPI::Add(tmp, qy2, qz2, mask);
                AscendC::MicroAPI::Muls(tmp, tmp, ROTMAT_COEFF, mask);
                AscendC::MicroAPI::Adds(r00, tmp, ONE_FLOAT_VALUE, mask);
                // R[0,1] = 2(qxy - qwz)
                AscendC::MicroAPI::Sub(tmp, qxy, qwz, mask);
                AscendC::MicroAPI::Muls(r01, tmp, TWO_FLOAT_VALUE, mask);
                // R[0,2] = 2(qxz + qwy)
                AscendC::MicroAPI::Add(tmp, qxz, qwy, mask);
                AscendC::MicroAPI::Muls(r02, tmp, TWO_FLOAT_VALUE, mask);
                // R[1,0] = 2(qxy + qwz)
                AscendC::MicroAPI::Add(tmp, qxy, qwz, mask);
                AscendC::MicroAPI::Muls(r10, tmp, TWO_FLOAT_VALUE, mask);
                // R[1,1] = 1 - 2(qx^2 + qz^2)
                AscendC::MicroAPI::Add(tmp, qx2, qz2, mask);
                AscendC::MicroAPI::Muls(tmp, tmp, ROTMAT_COEFF, mask);
                AscendC::MicroAPI::Adds(r11, tmp, ONE_FLOAT_VALUE, mask);
                // R[1,2] = 2(qyz - qxw)
                AscendC::MicroAPI::Sub(tmp, qyz, qxw, mask);
                AscendC::MicroAPI::Muls(r12, tmp, TWO_FLOAT_VALUE, mask);
                // R[2,0] = 2(qxz - qwy)
                AscendC::MicroAPI::Sub(tmp, qxz, qwy, mask);
                AscendC::MicroAPI::Muls(r20, tmp, TWO_FLOAT_VALUE, mask);
                // R[2,1] = 2(qyz + qxw)
                AscendC::MicroAPI::Add(tmp, qyz, qxw, mask);
                AscendC::MicroAPI::Muls(r21, tmp, TWO_FLOAT_VALUE, mask);
                // R[2,2] = 1 - 2(qx^2 + qy^2)
                AscendC::MicroAPI::Add(tmp, qx2, qy2, mask);
                AscendC::MicroAPI::Muls(tmp, tmp, ROTMAT_COEFF, mask);
                AscendC::MicroAPI::Adds(r22, tmp, ONE_FLOAT_VALUE, mask);

                // === M = R * diag(s): r_{i,j} *= s_j (in-place, 9 muls) ===
                AscendC::MicroAPI::Mul(r00, r00, sx, mask);
                AscendC::MicroAPI::Mul(r01, r01, sy, mask);
                AscendC::MicroAPI::Mul(r02, r02, sz, mask);
                AscendC::MicroAPI::Mul(r10, r10, sx, mask);
                AscendC::MicroAPI::Mul(r11, r11, sy, mask);
                AscendC::MicroAPI::Mul(r12, r12, sz, mask);
                AscendC::MicroAPI::Mul(r20, r20, sx, mask);
                AscendC::MicroAPI::Mul(r21, r21, sy, mask);
                AscendC::MicroAPI::Mul(r22, r22, sz, mask);
                // r-regs now hold M; sx/sy/sz dead from here on.

                // === Sigma = M @ M^T (6 unique entries) ===
                // Sigma[0,0] = m00^2 + m01^2 + m02^2  -> covars[0]
                AscendC::MicroAPI::Mul(acc, r00, r00, mask);
                AscendC::MicroAPI::Mul(tmp, r01, r01, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, r02, r02, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + 0 * taskNumPerCurLoop + off, acc, 1, mask);

                // Sigma[0,1] = m00*m10 + m01*m11 + m02*m12  -> covars[1] & covars[3]
                AscendC::MicroAPI::Mul(acc, r00, r10, mask);
                AscendC::MicroAPI::Mul(tmp, r01, r11, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, r02, r12, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + 1 * taskNumPerCurLoop + off, acc, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_3 * taskNumPerCurLoop + off, acc, 1, mask);

                // Sigma[0,2] = m00*m20 + m01*m21 + m02*m22  -> covars[2] & covars[6]
                AscendC::MicroAPI::Mul(acc, r00, r20, mask);
                AscendC::MicroAPI::Mul(tmp, r01, r21, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, r02, r22, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_2 * taskNumPerCurLoop + off, acc, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_6 * taskNumPerCurLoop + off, acc, 1, mask);

                // Sigma[1,1] = m10^2 + m11^2 + m12^2  -> covars[4]
                AscendC::MicroAPI::Mul(acc, r10, r10, mask);
                AscendC::MicroAPI::Mul(tmp, r11, r11, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, r12, r12, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_4 * taskNumPerCurLoop + off, acc, 1, mask);

                // Sigma[1,2] = m10*m20 + m11*m21 + m12*m22  -> covars[5] & covars[7]
                AscendC::MicroAPI::Mul(acc, r10, r20, mask);
                AscendC::MicroAPI::Mul(tmp, r11, r21, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, r12, r22, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_5 * taskNumPerCurLoop + off, acc, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_7 * taskNumPerCurLoop + off, acc, 1, mask);

                // Sigma[2,2] = m20^2 + m21^2 + m22^2  -> covars[8]
                AscendC::MicroAPI::Mul(acc, r20, r20, mask);
                AscendC::MicroAPI::Mul(tmp, r21, r21, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::Mul(tmp, r22, r22, mask);
                AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_8 * taskNumPerCurLoop + off, acc, 1, mask);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t batchCopyinIndex, uint64_t taskCopyIndex)
    {
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);

        // Bounded reads with DataCopyPad: each GM read is clamped to
        // actualValidCount floats per row (no OOB read past bcnSize). The UB
        // dirty tail is left untouched; CopyOut truncation stops dirty-lane
        // results from reaching GM.
        DataCopyExtParams ci_params{1, static_cast<uint32_t>(actualValidCount * sizeof(float)), 0, 0, 0};
        // isPad=false matches the production templates (fully_fused_projection_bwd);
        // the hardware MTE pad mechanism rejects rightPadding>0 here (subErrType 0x4).
        // The UB dirty tail is previous-iteration garbage; CopyOut truncates the GM
        // write to actualValidCount, so dirty-lane ALU results never reach GM.
        DataCopyPadExtParams<float> ci_pad{false, 0, 0, 0.0f};

        for (int32_t i = 0; i < MEANS_TAIL_DIM; i++) {
            DataCopyPad(MeansWorldTensor[taskNumPerCurLoop * i],
                        meansGM[batchCopyinIndex * MEANS_TAIL_DIM + i * gaussianNum + taskCopyIndex],
                        ci_params, ci_pad);
        }
        // Dispatch: covars-direct vs inlined quat_scales_to_covars.
        if (inputIsQuatScales == 1) {
            // Load quat (4f) + scales (3f), then inline produce world-frame covars.
            for (int32_t i = 0; i < QUAT_TAIL_DIM; i++) {
                DataCopyPad(QuatTensor[taskNumPerCurLoop * i],
                            quatGM[batchCopyinIndex * QUAT_TAIL_DIM + i * gaussianNum + taskCopyIndex],
                            ci_params, ci_pad);
            }
            for (int32_t i = 0; i < SCALES_TAIL_DIM; i++) {
                DataCopyPad(ScalesTensor[taskNumPerCurLoop * i],
                            scalesGM[batchCopyinIndex * SCALES_TAIL_DIM + i * gaussianNum + taskCopyIndex],
                            ci_params, ci_pad);
            }
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            ComputeCovarsFromQuatScales();
        } else {
            for (int32_t i = 0; i < COVARS_TAIL_DIM; i++) {
                DataCopyPad(CovarsWorldTensor[taskNumPerCurLoop * i],
                            covarsGM[batchCopyinIndex * COVARS_TAIL_DIM + i * gaussianNum + taskCopyIndex],
                            ci_params, ci_pad);
            }
        }

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    // A5 Reg-API helpers (pure refactor; no algorithmic change).
    //
    // Both helpers implement the verified pattern from step-1.2 / 2 / 3 / 4 / 5 / 6:
    //   __VEC_SCOPE__ {
    //     for each lane in [0, lanesPerChunk):
    //       load src1 (1 lane, 64 floats) into RegTensor a
    //       load src2 (1 lane, 64 floats) into RegTensor b
    //       Mul / Add a, b -> RegTensor c
    //       store c into dst (1 lane, 64 floats)
    //   }
    //
    // Each chunk lives at offset (chunk_idx * chunk_stride) within its buffer.
    // Chunk strides are buffer-specific (taskNumPerCurLoop for Means/Computing,
    // taskNumPerLoop for CameraParam) -- caller supplies the right one.
    //
    // All buffers must be lane-strided 64-aligned; lanesPerChunk = chunk_size / 64.
    __aicore__ inline void VecMul3Buf(
        __ubuf__ float* p_dst,  uint32_t dst_chunk,  uint32_t dst_stride,
        __ubuf__ float* p_src1, uint32_t src1_chunk, uint32_t src1_stride,
        __ubuf__ float* p_src2, uint32_t src2_chunk, uint32_t src2_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> b;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst  = dst_chunk  * dst_stride  + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src1 = src1_chunk * src1_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src2 = src2_chunk * src2_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src1 + off_src1, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    b, p_src2 + off_src2, 1, mask);
                AscendC::MicroAPI::Mul(c, a, b, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    __aicore__ inline void VecAdd3Buf(
        __ubuf__ float* p_dst,  uint32_t dst_chunk,  uint32_t dst_stride,
        __ubuf__ float* p_src1, uint32_t src1_chunk, uint32_t src1_stride,
        __ubuf__ float* p_src2, uint32_t src2_chunk, uint32_t src2_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> b;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst  = dst_chunk  * dst_stride  + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src1 = src1_chunk * src1_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src2 = src2_chunk * src2_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src1 + off_src1, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    b, p_src2 + off_src2, 1, mask);
                AscendC::MicroAPI::Add(c, a, b, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecDiv3Buf helper: same shape as VecMul3Buf / VecAdd3Buf, op = MicroAPI::Div.
    __aicore__ inline void VecDiv3Buf(
        __ubuf__ float* p_dst,  uint32_t dst_chunk,  uint32_t dst_stride,
        __ubuf__ float* p_src1, uint32_t src1_chunk, uint32_t src1_stride,
        __ubuf__ float* p_src2, uint32_t src2_chunk, uint32_t src2_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> b;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst  = dst_chunk  * dst_stride  + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src1 = src1_chunk * src1_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src2 = src2_chunk * src2_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src1 + off_src1, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    b, p_src2 + off_src2, 1, mask);
                AscendC::MicroAPI::Div(c, a, b, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecSub3Buf: same shape as VecAdd3Buf, op = MicroAPI::Sub.
    __aicore__ inline void VecSub3Buf(
        __ubuf__ float* p_dst,  uint32_t dst_chunk,  uint32_t dst_stride,
        __ubuf__ float* p_src1, uint32_t src1_chunk, uint32_t src1_stride,
        __ubuf__ float* p_src2, uint32_t src2_chunk, uint32_t src2_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> b;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst  = dst_chunk  * dst_stride  + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src1 = src1_chunk * src1_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src2 = src2_chunk * src2_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src1 + off_src1, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    b, p_src2 + off_src2, 1, mask);
                AscendC::MicroAPI::Sub(c, a, b, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecMulsScalar: reg = reg * scalar.
    __aicore__ inline void VecMulsScalar(
        __ubuf__ float* p_dst, uint32_t dst_chunk, uint32_t dst_stride,
        __ubuf__ float* p_src, uint32_t src_chunk, uint32_t src_stride,
        float scalar,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst = dst_chunk * dst_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src = src_chunk * src_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src + off_src, 1, mask);
                AscendC::MicroAPI::Muls(c, a, scalar, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecDupScalar: reg = scalar (broadcast).
    __aicore__ inline void VecDupScalar(
        __ubuf__ float* p_dst, uint32_t dst_chunk, uint32_t dst_stride,
        float value,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst = dst_chunk * dst_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::Duplicate(c, value, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecMax3Buf / VecMin3Buf: same shape as VecAdd3Buf, ops = MicroAPI::Max / Min.
    __aicore__ inline void VecMax3Buf(
        __ubuf__ float* p_dst,  uint32_t dst_chunk,  uint32_t dst_stride,
        __ubuf__ float* p_src1, uint32_t src1_chunk, uint32_t src1_stride,
        __ubuf__ float* p_src2, uint32_t src2_chunk, uint32_t src2_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> b;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst  = dst_chunk  * dst_stride  + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src1 = src1_chunk * src1_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src2 = src2_chunk * src2_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src1 + off_src1, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    b, p_src2 + off_src2, 1, mask);
                AscendC::MicroAPI::Max(c, a, b, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    __aicore__ inline void VecMin3Buf(
        __ubuf__ float* p_dst,  uint32_t dst_chunk,  uint32_t dst_stride,
        __ubuf__ float* p_src1, uint32_t src1_chunk, uint32_t src1_stride,
        __ubuf__ float* p_src2, uint32_t src2_chunk, uint32_t src2_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> b;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst  = dst_chunk  * dst_stride  + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src1 = src1_chunk * src1_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src2 = src2_chunk * src2_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src1 + off_src1, 1, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    b, p_src2 + off_src2, 1, mask);
                AscendC::MicroAPI::Min(c, a, b, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecAddsScalar / VecMaxsScalar / VecSqrt2Buf helpers.
    __aicore__ inline void VecAddsScalar(
        __ubuf__ float* p_dst, uint32_t dst_chunk, uint32_t dst_stride,
        __ubuf__ float* p_src, uint32_t src_chunk, uint32_t src_stride,
        float scalar,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst = dst_chunk * dst_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src = src_chunk * src_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src + off_src, 1, mask);
                AscendC::MicroAPI::Adds(c, a, scalar, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    __aicore__ inline void VecMaxsScalar(
        __ubuf__ float* p_dst, uint32_t dst_chunk, uint32_t dst_stride,
        __ubuf__ float* p_src, uint32_t src_chunk, uint32_t src_stride,
        float scalar,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst = dst_chunk * dst_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src = src_chunk * src_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src + off_src, 1, mask);
                AscendC::MicroAPI::Maxs(c, a, scalar, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    __aicore__ inline void VecSqrt2Buf(
        __ubuf__ float* p_dst, uint32_t dst_chunk, uint32_t dst_stride,
        __ubuf__ float* p_src, uint32_t src_chunk, uint32_t src_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a;
            AscendC::MicroAPI::RegTensor<float> c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst = dst_chunk * dst_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src = src_chunk * src_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src + off_src, 1, mask);
                AscendC::MicroAPI::Sqrt(c, a, mask);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask);
            }
        }
    }

    // VecCeil2Buf via indicator-add pattern.
    //
    // History of attempts:
    //   1. MicroAPI::Cast<int32_t, float, CAST_CEIL> -- ccec rejected
    //      ("no matching function"). The trait-based Cast for f32 -> int32
    //      seems to support only CAST_RINT / CAST_TRUNC, not CAST_CEIL.
    //      (int16 path supports CAST_CEIL per ascend_quant; we could fall
    //       back to that if needed, radii fit in int16 range.)
    //   2. CAST_RINT + indicator trick (copied from archive/v1-easyasc):
    //      compiled but UT precision wrong. RINT on cann-8.5.0 is
    //      genuine IEEE round-to-nearest, NOT truncation -- so for
    //      x = 5.7 -> rint = 6, frac = -0.3, indicator goes large-negative,
    //      result explodes.
    //   3. CAST_TRUNC + indicator trick (current): TRUNC is round-toward-
    //      zero, which equals floor for x >= 0. Indicator trick then works:
    //        x=5.0 -> trunc=5, frac=0, indicator=0, result=5
    //        x=5.3 -> trunc=5, frac=0.3, indicator=1, result=6
    //        x=5.7 -> trunc=5, frac=0.7, indicator=1, result=6
    //
    // Algorithm (radius inputs are always >= 0 in this kernel):
    //   floor_x   = (float)(int32_t)x        // CAST_TRUNC round-trip
    //   frac      = x - floor_x              // in [0, 1)
    //   indicator = min(frac * 1e10f, 1.0f)  // 0 if frac == 0 else 1
    //   result    = floor_x + indicator      // ceil(x)
    static constexpr AscendC::MicroAPI::CastTrait VEC_CEIL_TRAIT = {
        AscendC::MicroAPI::RegLayout::UNKNOWN,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC, // round toward zero == floor for x>=0; indicator trick needs deterministic floor
    };

    __aicore__ inline void VecCeil2Buf(
        __ubuf__ float* p_dst, uint32_t dst_chunk, uint32_t dst_stride,
        __ubuf__ float* p_src, uint32_t src_chunk, uint32_t src_stride,
        uint16_t lanesPerChunk)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask_f =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg mask_i =
                AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float>   a;
            AscendC::MicroAPI::RegTensor<int32_t> b;
            AscendC::MicroAPI::RegTensor<float>   floor_x;
            AscendC::MicroAPI::RegTensor<float>   frac;
            AscendC::MicroAPI::RegTensor<float>   indicator;
            AscendC::MicroAPI::RegTensor<float>   c;
            constexpr uint16_t LANE = 64;
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_dst = dst_chunk * dst_stride + static_cast<uint32_t>(lane) * LANE;
                const uint32_t off_src = src_chunk * src_stride + static_cast<uint32_t>(lane) * LANE;
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a, p_src + off_src, 1, mask_f);
                AscendC::MicroAPI::Cast<int32_t, float, VEC_CEIL_TRAIT>(b, a, mask_i);
                AscendC::MicroAPI::Cast<float, int32_t, VEC_CEIL_TRAIT>(floor_x, b, mask_f);
                AscendC::MicroAPI::Sub(frac, a, floor_x, mask_f);
                AscendC::MicroAPI::Muls(frac, frac, 1.0e10f, mask_f);
                AscendC::MicroAPI::Mins(indicator, frac, 1.0f, mask_f);
                AscendC::MicroAPI::Add(c, floor_x, indicator, mask_f);
                AscendC::MicroAPI::DataCopy<float,
                    AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_dst + off_dst, c, 1, mask_f);
            }
        }
    }

    __aicore__ inline void ComputingMeansCamera()
    {
        // World->camera means: mxc/myc/mzc = R . means + t.
        // mx/my/mz are loaded once per lane and kept register-resident across
        // the three legs; each result is written to both ComputingTensor[i] and
        // MeansTensor[i] in a single scope.
        {
            __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
            __ubuf__ float* p_means   = (__ubuf__ float*) MeansTensor.GetPhyAddr();
            __ubuf__ float* p_means_world = (__ubuf__ float*) MeansWorldTensor.GetPhyAddr();  // world-frame input
            __ubuf__ float* p_camera  = (__ubuf__ float*) CameraParamTensor.GetPhyAddr();
            const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg mask =
                    AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::RegTensor<float> mx, my, mz;       // input means; register-resident across the 3 legs
                AscendC::MicroAPI::RegTensor<float> ra, rb, rc, tr;   // per-leg R column + t
                AscendC::MicroAPI::RegTensor<float> tmp, acc;
                constexpr uint16_t LANE = 64;

                for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                    const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                    // Load means once; held in mx/my/mz registers across all 3 legs.
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        mx, p_means_world + 0 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        my, p_means_world + 1 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        mz, p_means_world + MEANS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);

                    // === mxc leg: acc = R00*mx + R01*my + R02*mz + t0 ===
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + 0);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + 1);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_02);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        tr, p_camera + VIEWMATS_03);
                    AscendC::MicroAPI::Mul(acc, ra, mx, mask);
                    AscendC::MicroAPI::Mul(tmp, rb, my, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Mul(tmp, rc, mz, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Add(acc, acc, tr, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + 0 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_means + 0 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // === myc leg: acc = R10*mx + R11*my + R12*mz + t1 ===
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + VIEWMATS_10);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + VIEWMATS_11);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_12);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        tr, p_camera + VIEWMATS_13);
                    AscendC::MicroAPI::Mul(acc, ra, mx, mask);
                    AscendC::MicroAPI::Mul(tmp, rb, my, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Mul(tmp, rc, mz, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Add(acc, acc, tr, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + 1 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_means + 1 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // === mzc leg: acc = R20*mx + R21*my + R22*mz + t2 ===
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + VIEWMATS_20);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + VIEWMATS_21);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_22);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        tr, p_camera + VIEWMATS_23);
                    AscendC::MicroAPI::Mul(acc, ra, mx, mask);
                    AscendC::MicroAPI::Mul(tmp, rb, my, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Mul(tmp, rc, mz, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Add(acc, acc, tr, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + COMPUTE_OFFSET_2 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_means + MEANS_OFFSET_2 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                }
            }
        }
    }
    __aicore__ inline void ComputingCovarsCamera()
    {
        // World->camera covariance: covars_c = R . cov . R^T, as two 3x3
        // matrix mults in series (Part A: tmp = R . cov, Part B: tmp . R^T).
        // In each part the 9 input elements are loaded once per lane and held
        // register-resident across the 9 outputs (each a 3-term dot product).
        // The two parts exchange data through UB, since registers do not
        // survive across a __VEC_SCOPE__ boundary.

        __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
        __ubuf__ float* p_covars  = (__ubuf__ float*) CovarsTensor.GetPhyAddr();
        __ubuf__ float* p_covars_world = (__ubuf__ float*) CovarsWorldTensor.GetPhyAddr();  // world-frame input
        __ubuf__ float* p_camera  = (__ubuf__ float*) CameraParamTensor.GetPhyAddr();
        const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

        // ============================================================
        // Part A: tmp = R . cov  (tmp written to ComputingTensor[0..8])
        // ============================================================
        // cov layout (input, in CovarsTensor):
        //   row 0: cov[0],          cov[OFFSET_3], cov[OFFSET_6]
        //   row 1: cov[1],          cov[OFFSET_4], cov[OFFSET_7]
        //   row 2: cov[OFFSET_2],   cov[OFFSET_5], cov[OFFSET_8]
        //
        // Output chunk index in ComputingTensor:
        //   tmp[output] = sum_k cov[col_of_output * 3 + k] * R[row_of_output][k]
        //
        // tmp[0..2] use R row 0; tmp[3..5] use R row 1; tmp[6..8] use R row 2.
        // tmp[0/3/6] dot col 0 of cov; tmp[1/4/7] dot col 1; tmp[2/5/8] dot col 2.
        {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg mask =
                    AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::RegTensor<float> c0, c1, c2, c3, c4, c5, c6, c7, c8;  // all 9 cov elements (register-resident)
                AscendC::MicroAPI::RegTensor<float> ra, rb, rc;                          // current R row
                AscendC::MicroAPI::RegTensor<float> tmp, acc;
                constexpr uint16_t LANE = 64;

                for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                    const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                    // Load all 9 cov entries; held alive across the 9 output computations.
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c0, p_covars_world + 0              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c1, p_covars_world + 1              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c2, p_covars_world + COVARS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c3, p_covars_world + COVARS_OFFSET_3 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c4, p_covars_world + COVARS_OFFSET_4 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c5, p_covars_world + COVARS_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c6, p_covars_world + COVARS_OFFSET_6 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c7, p_covars_world + COVARS_OFFSET_7 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c8, p_covars_world + COVARS_OFFSET_8 * taskNumPerCurLoop + off_lane, 1, mask);

                    #define DOT3_STORE(out_off, x0, x1, x2)                                            \
                        AscendC::MicroAPI::Mul(acc, ra, x0, mask);                                     \
                        AscendC::MicroAPI::Mul(tmp, rb, x1, mask);                                     \
                        AscendC::MicroAPI::Add(acc, acc, tmp, mask);                                   \
                        AscendC::MicroAPI::Mul(tmp, rc, x2, mask);                                     \
                        AscendC::MicroAPI::Add(acc, acc, tmp, mask);                                   \
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>( \
                            p_compute + (out_off) * taskNumPerCurLoop + off_lane, acc, 1, mask)

                    // --- R row 0 -> tmp[0], tmp[1], tmp[2] ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + 0);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + 1);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_02);
                    DOT3_STORE(0,                c0, c3, c6);  // tmp[0] = R00*c0 + R01*c3 + R02*c6
                    DOT3_STORE(1,                c1, c4, c7);  // tmp[1] = R00*c1 + R01*c4 + R02*c7
                    DOT3_STORE(COMPUTE_OFFSET_2, c2, c5, c8);  // tmp[2] = R00*c2 + R01*c5 + R02*c8

                    // --- R row 1 -> tmp[3], tmp[4], tmp[5] ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + VIEWMATS_10);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + VIEWMATS_11);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_12);
                    DOT3_STORE(COMPUTE_OFFSET_3, c0, c3, c6);
                    DOT3_STORE(COMPUTE_OFFSET_4, c1, c4, c7);
                    DOT3_STORE(COMPUTE_OFFSET_5, c2, c5, c8);

                    // --- R row 2 -> tmp[6], tmp[7], tmp[8] ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + VIEWMATS_20);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + VIEWMATS_21);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_22);
                    DOT3_STORE(COMPUTE_OFFSET_6, c0, c3, c6);
                    DOT3_STORE(COMPUTE_OFFSET_7, c1, c4, c7);
                    DOT3_STORE(COMPUTE_OFFSET_8, c2, c5, c8);

                    #undef DOT3_STORE
                }
            }
        }

        // ============================================================
        // Part B: covars_c = tmp . R^T  (covars_c written to CovarsTensor[0..8])
        // ============================================================
        // tmp layout (input, in ComputingTensor [0..8]):
        //   row 0: tmp[0], tmp[1], tmp[OFFSET_2]
        //   row 1: tmp[OFFSET_3], tmp[OFFSET_4], tmp[OFFSET_5]
        //   row 2: tmp[OFFSET_6], tmp[OFFSET_7], tmp[OFFSET_8]
        //
        // covars[output] = sum_k tmp[row][k] * R[R_row][k]      (R^T -> R indexed by R_row, k)
        //   covars[0/1/2]:           tmp row 0 . R row 0/1/2
        //   covars[3/4/5]:           tmp row 1 . R row 0/1/2
        //   covars[6/7/8]:           tmp row 2 . R row 0/1/2
        {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg mask =
                    AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::RegTensor<float> t0, t1, t2, t3, t4, t5, t6, t7, t8;  // all 9 tmp (register-resident)
                AscendC::MicroAPI::RegTensor<float> ra, rb, rc;
                AscendC::MicroAPI::RegTensor<float> tmp, acc;
                constexpr uint16_t LANE = 64;

                for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                    const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t0, p_compute + 0               * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t1, p_compute + 1               * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t2, p_compute + COMPUTE_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t3, p_compute + COMPUTE_OFFSET_3 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t4, p_compute + COMPUTE_OFFSET_4 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t5, p_compute + COMPUTE_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t6, p_compute + COMPUTE_OFFSET_6 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t7, p_compute + COMPUTE_OFFSET_7 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        t8, p_compute + COMPUTE_OFFSET_8 * taskNumPerCurLoop + off_lane, 1, mask);

                    #define DOT3_STORE_COVARS(out_off, x0, x1, x2)                                     \
                        AscendC::MicroAPI::Mul(acc, ra, x0, mask);                                     \
                        AscendC::MicroAPI::Mul(tmp, rb, x1, mask);                                     \
                        AscendC::MicroAPI::Add(acc, acc, tmp, mask);                                   \
                        AscendC::MicroAPI::Mul(tmp, rc, x2, mask);                                     \
                        AscendC::MicroAPI::Add(acc, acc, tmp, mask);                                   \
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>( \
                            p_covars + (out_off) * taskNumPerCurLoop + off_lane, acc, 1, mask)

                    // --- R row 0 -> covars[0], covars[OFFSET_3], covars[OFFSET_6] ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + 0);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + 1);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_02);
                    DOT3_STORE_COVARS(0,               t0, t1, t2);  // covars[0] = R0 . tmp_row0
                    // covars[3]=c10=c01 (symmetric) -> skip; downstream reads OFFSET_1 instead.
                    // covars[6]=c20=c02 (symmetric) -> skip; downstream reads OFFSET_2 instead.

                    // --- R row 1 -> covars[1], covars[OFFSET_4], covars[OFFSET_7] ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + VIEWMATS_10);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + VIEWMATS_11);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_12);
                    DOT3_STORE_COVARS(1,               t0, t1, t2);
                    DOT3_STORE_COVARS(COVARS_OFFSET_4, t3, t4, t5);
                    // covars[7]=c21=c12 (symmetric) -> skip; downstream reads OFFSET_5 instead.

                    // --- R row 2 -> covars[OFFSET_2], covars[OFFSET_5], covars[OFFSET_8] ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ra, p_camera + VIEWMATS_20);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rb, p_camera + VIEWMATS_21);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        rc, p_camera + VIEWMATS_22);
                    DOT3_STORE_COVARS(COVARS_OFFSET_2, t0, t1, t2);
                    DOT3_STORE_COVARS(COVARS_OFFSET_5, t3, t4, t5);
                    DOT3_STORE_COVARS(COVARS_OFFSET_8, t6, t7, t8);

                    #undef DOT3_STORE_COVARS
                }
            }
        }
    }

    __aicore__ inline void PerspProjection()
    {
        ComputingMeans2D();
        ComputingCovParams();
        ComputingCovars2D();
    }

    __aicore__ inline void ComputingMeans2D()
    {
        // Project camera-space means to 2D:
        //   means2d_x = (K00*mxc + K01*myc + K02*mzc) / mzc
        //   means2d_y = (K10*mxc + K11*myc + K12*mzc) / mzc
        // mxc/myc/mzc are loaded once per lane and kept register-resident
        // across both legs (mzc is reused as the divisor in each).
        {
            __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
            __ubuf__ float* p_m2d = dbParity ? (__ubuf__ float*) Means2dDbTensor.GetPhyAddr() : p_compute;
            __ubuf__ float* p_means   = (__ubuf__ float*) MeansTensor.GetPhyAddr();
            __ubuf__ float* p_camera  = (__ubuf__ float*) CameraParamTensor.GetPhyAddr();
            const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg mask =
                    AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::RegTensor<float> mxc, myc, mzc;  // register-resident across both legs
                AscendC::MicroAPI::RegTensor<float> ka, kb, kc;     // current K row
                AscendC::MicroAPI::RegTensor<float> tmp, acc;
                constexpr uint16_t LANE = 64;

                for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                    const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        mxc, p_means + 0              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        myc, p_means + 1              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        mzc, p_means + MEANS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);

                    // --- x-leg: out[0] = (K00*mxc + K01*myc + K02*mzc) / mzc ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ka, p_camera + KS_00);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        kb, p_camera + KS_01);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        kc, p_camera + KS_02);
                    AscendC::MicroAPI::Mul(acc, ka, mxc, mask);
                    AscendC::MicroAPI::Mul(tmp, kb, myc, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Mul(tmp, kc, mzc, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Div(acc, acc, mzc, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_m2d + 0 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // --- y-leg: out[1] = (K10*mxc + K11*myc + K12*mzc) / mzc ---
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        ka, p_camera + KS_10);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        kb, p_camera + KS_11);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                        kc, p_camera + KS_12);
                    AscendC::MicroAPI::Mul(acc, ka, mxc, mask);
                    AscendC::MicroAPI::Mul(tmp, kb, myc, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Mul(tmp, kc, mzc, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::Div(acc, acc, mzc, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_m2d + 1 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                }
            }
        }
    }
    __aicore__ inline void ComputingCovParams()
    {
        // Perspective-projection Jacobian helpers, in one scope across three
        // segments that share registers (mxc/myc/mzc and fx/fy loaded once; the
        // clamp limits and tz2/tx/ty pass between segments):
        //   Seg1: lim_*_neg/pos = view-frustum clamp limits from c*/f* and W,H
        //   Seg2: tx = clamp(mxc/mzc, lim_x) * mzc;  ty likewise;  tz2 = mzc*mzc
        //   Seg3: a1 = fx/mzc, a2 = -(tx*fx)/tz2, b1 = fy/mzc, b2 = -(ty*fy)/tz2
        // Outputs are written back to UB for the downstream Covars2D scope.

        __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
        __ubuf__ float* p_means   = (__ubuf__ float*) MeansTensor.GetPhyAddr();
        __ubuf__ float* p_camera  = (__ubuf__ float*) CameraParamTensor.GetPhyAddr();
        const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> mxc, myc, mzc;          // shared Seg2+Seg3
            AscendC::MicroAPI::RegTensor<float> fx, fy, cx, cy;         // shared Seg1 (+Seg3 for fx/fy)
            AscendC::MicroAPI::RegTensor<float> lim_xn, lim_xp, lim_yn, lim_yp;  // Seg1 -> Seg2
            AscendC::MicroAPI::RegTensor<float> tx, ty, tz2;            // Seg2 -> Seg3
            AscendC::MicroAPI::RegTensor<float> ones, Wofx, Wofx_b, Hofy, Hofy_b, cxofx, cyofy;
            AscendC::MicroAPI::RegTensor<float> tmp, acc;
            constexpr uint16_t LANE = 64;

            // ============ PER-CAMERA SETUP (hoisted out of lane loop) ============
            //   Camera params + Seg 1 lim_* are invariant across lane iterations.
            //   Original code recomputed 32 times per ComputingCovParams call.
            //   The 4 UB writes to p_compute OFFSET_8/9/10/11 were dead (Seg 2 uses
            //   RegTensors directly; ComputingCovars2D Part B overwrites those slots
            //   before any downstream read).
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                fx, p_camera + KS_00);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                fy, p_camera + KS_11);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                cx, p_camera + KS_02);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                cy, p_camera + KS_12);

            AscendC::MicroAPI::Duplicate(ones, ONE_FLOAT_VALUE, mask);

            // x: cxofx = cx/fx; 1/fx -> Wofx -> W/fx; Wofx_b = W/fx*BOUND
            AscendC::MicroAPI::Div(cxofx, cx, fx, mask);
            AscendC::MicroAPI::Div(Wofx, ones, fx, mask);
            AscendC::MicroAPI::Muls(Wofx, Wofx, imageWidth, mask);
            AscendC::MicroAPI::Muls(Wofx_b, Wofx, BOUND_COEFF_PARAM, mask);
            AscendC::MicroAPI::Add(acc, cxofx, Wofx_b, mask);       // cx/fx + W/fx*BOUND
            AscendC::MicroAPI::Muls(lim_xn, acc, -1.0f, mask);      // lim_x_neg
            AscendC::MicroAPI::Add(lim_xp, Wofx, Wofx_b, mask);     // W/fx + W/fx*BOUND
            AscendC::MicroAPI::Sub(lim_xp, lim_xp, cxofx, mask);    // -= cx/fx => lim_x_pos

            // y: same shape with cy/fy/imageHeight
            AscendC::MicroAPI::Div(cyofy, cy, fy, mask);
            AscendC::MicroAPI::Div(Hofy, ones, fy, mask);
            AscendC::MicroAPI::Muls(Hofy, Hofy, imageHeight, mask);
            AscendC::MicroAPI::Muls(Hofy_b, Hofy, BOUND_COEFF_PARAM, mask);
            AscendC::MicroAPI::Add(acc, cyofy, Hofy_b, mask);
            AscendC::MicroAPI::Muls(lim_yn, acc, -1.0f, mask);
            AscendC::MicroAPI::Add(lim_yp, Hofy, Hofy_b, mask);
            AscendC::MicroAPI::Sub(lim_yp, lim_yp, cyofy, mask);

            // ============ PER-LANE LOOP (only Seg 2 + Seg 3 remain inside) ============
            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    mxc, p_means + 0              * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    myc, p_means + 1              * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    mzc, p_means + MEANS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);

                // ============ Segment 2: tx / ty / tz2 ============
                // tx = clamp(mxc/mzc, lim_x_neg, lim_x_pos) * mzc
                AscendC::MicroAPI::Div(acc, mxc, mzc, mask);
                AscendC::MicroAPI::Max(acc, acc, lim_xn, mask);
                AscendC::MicroAPI::Min(acc, acc, lim_xp, mask);
                AscendC::MicroAPI::Mul(tx, acc, mzc, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_means + 0 * taskNumPerCurLoop + off_lane, tx, 1, mask);

                // ty = clamp(myc/mzc, lim_y_neg, lim_y_pos) * mzc
                AscendC::MicroAPI::Div(acc, myc, mzc, mask);
                AscendC::MicroAPI::Max(acc, acc, lim_yn, mask);
                AscendC::MicroAPI::Min(acc, acc, lim_yp, mask);
                AscendC::MicroAPI::Mul(ty, acc, mzc, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_means + 1 * taskNumPerCurLoop + off_lane, ty, 1, mask);

                // tz2 = mzc * mzc
                AscendC::MicroAPI::Mul(tz2, mzc, mzc, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_2 * taskNumPerCurLoop + off_lane, tz2, 1, mask);

                // ============ Segment 3: a1 / a2 / b1 / b2 ============
                // a1 = fx / mzc
                AscendC::MicroAPI::Div(acc, fx, mzc, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_3 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                // a2 = -(tx * fx) / tz2   (tx is read from MeansTensor[0], not the original mxc)
                AscendC::MicroAPI::Mul(acc, tx, fx, mask);
                AscendC::MicroAPI::Div(acc, acc, tz2, mask);
                AscendC::MicroAPI::Muls(acc, acc, -1.0f, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_4 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                // b1 = fy / mzc
                AscendC::MicroAPI::Div(acc, fy, mzc, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_5 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                // b2 = -(ty * fy) / tz2
                AscendC::MicroAPI::Mul(acc, ty, fy, mask);
                AscendC::MicroAPI::Div(acc, acc, tz2, mask);
                AscendC::MicroAPI::Muls(acc, acc, -1.0f, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_6 * taskNumPerCurLoop + off_lane, acc, 1, mask);
            }
        }
    }

    __aicore__ inline void ComputingCovars2D()
    {
        // 2D covariance: cov2d = J . cov_3x3 . J^T, in two scopes --
        // Part A (cov_2x3 = J . cov_3x3) and Part B (cov2d = cov_2x3 . J^T).
        // Each part keeps its inputs register-resident across the outputs.
        // J row 0 = (a1, 0, a2), row 1 = (0, b1, b2); the zeros are dropped at
        // the algebra level so each entry is a 2-term dot product.

        __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
        __ubuf__ float* p_covars  = (__ubuf__ float*) CovarsTensor.GetPhyAddr();
        const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

        // ============================================================
        // Part A: cov_2x3 = J . cov_3x3
        //   cov_2x3[0,k] = a1*cov[0,k] + a2*cov[2,k]   for k = 0,1,2
        //   cov_2x3[1,k] = b1*cov[1,k] + b2*cov[2,k]   for k = 0,1,2
        // Output: CovarsTensor[0..OFFSET_5] (overwrites cov[0,*] / cov[1,*])
        // ============================================================
        {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg mask =
                    AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::RegTensor<float> a1, a2, b1, b2;
                AscendC::MicroAPI::RegTensor<float> c00, c01, c02;   // cov row 0
                AscendC::MicroAPI::RegTensor<float> c10, c11, c12;   // cov row 1
                AscendC::MicroAPI::RegTensor<float> c20, c21, c22;   // cov row 2 (used in all 6 entries)
                AscendC::MicroAPI::RegTensor<float> tmp, acc;
                constexpr uint16_t LANE = 64;

                for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                    const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                    // Load 4 Jacobian helpers (kept alive across Part A entries).
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        a1, p_compute + COMPUTE_OFFSET_3 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        a2, p_compute + COMPUTE_OFFSET_4 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        b1, p_compute + COMPUTE_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        b2, p_compute + COMPUTE_OFFSET_6 * taskNumPerCurLoop + off_lane, 1, mask);

                    // Load 9 cov_3x3 elements (kept alive within Part A).
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c00, p_covars + 0              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c01, p_covars + 1              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c02, p_covars + COVARS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c10, p_covars + 1               * taskNumPerCurLoop + off_lane, 1, mask);  // c10 = c01 (cov_c symmetric)
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c11, p_covars + COVARS_OFFSET_4 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c12, p_covars + COVARS_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c20, p_covars + COVARS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);  // c20 = c02 (cov_c symmetric)
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c21, p_covars + COVARS_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);  // c21 = c12 (cov_c symmetric)
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        c22, p_covars + COVARS_OFFSET_8 * taskNumPerCurLoop + off_lane, 1, mask);

                    #define A_ROW0(dst_off, ck)                                                                       \
                        AscendC::MicroAPI::Mul(acc, a1, ck##0, mask);                                                 \
                        AscendC::MicroAPI::Mul(tmp, a2, ck##1, mask);                                                 \
                        AscendC::MicroAPI::Add(acc, acc, tmp, mask);                                                  \
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(         \
                            p_covars + (dst_off) * taskNumPerCurLoop + off_lane, acc, 1, mask)

                    #define A_ROW1(dst_off, ck)                                                                       \
                        AscendC::MicroAPI::Mul(acc, b1, ck##0, mask);                                                 \
                        AscendC::MicroAPI::Mul(tmp, b2, ck##1, mask);                                                 \
                        AscendC::MicroAPI::Add(acc, acc, tmp, mask);                                                  \
                        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(         \
                            p_covars + (dst_off) * taskNumPerCurLoop + off_lane, acc, 1, mask)

                    // cov_2x3 row 0: a1 * cov[0,k] + a2 * cov[2,k]
                    // Pair (c0k, c2k) with macro: first src = c[0,k], second = c[2,k]
                    // ck##0 = c00 etc means we name args c0,c2 -> c0c2 -- doesn't quite work with token paste.
                    // Just emit inline manually:
                    #undef A_ROW0
                    #undef A_ROW1

                    // cov_2x3[0,0] = a1*c00 + a2*c20
                    AscendC::MicroAPI::Mul(acc, a1, c00, mask);
                    AscendC::MicroAPI::Mul(tmp, a2, c20, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_covars + 0 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov_2x3[0,1] = a1*c01 + a2*c21
                    AscendC::MicroAPI::Mul(acc, a1, c01, mask);
                    AscendC::MicroAPI::Mul(tmp, a2, c21, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_covars + 1 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov_2x3[0,2] = a1*c02 + a2*c22
                    AscendC::MicroAPI::Mul(acc, a1, c02, mask);
                    AscendC::MicroAPI::Mul(tmp, a2, c22, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_covars + COVARS_OFFSET_2 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov_2x3[1,0] = b1*c10 + b2*c20
                    AscendC::MicroAPI::Mul(acc, b1, c10, mask);
                    AscendC::MicroAPI::Mul(tmp, b2, c20, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_covars + COVARS_OFFSET_3 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov_2x3[1,1] = b1*c11 + b2*c21
                    AscendC::MicroAPI::Mul(acc, b1, c11, mask);
                    AscendC::MicroAPI::Mul(tmp, b2, c21, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_covars + COVARS_OFFSET_4 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov_2x3[1,2] = b1*c12 + b2*c22
                    AscendC::MicroAPI::Mul(acc, b1, c12, mask);
                    AscendC::MicroAPI::Mul(tmp, b2, c22, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_covars + COVARS_OFFSET_5 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                }
            }
        }

        // ============================================================
        // Part B: cov2d_2x2 = cov_2x3 . J^T
        //   cov2d[0,0] = a1*cov_2x3[0,0] + a2*cov_2x3[0,2]
        //   cov2d[0,1] = b1*cov_2x3[0,1] + b2*cov_2x3[0,2]
        //   cov2d[1,0] = a1*cov_2x3[1,0] + a2*cov_2x3[1,2]
        //   cov2d[1,1] = b1*cov_2x3[1,1] + b2*cov_2x3[1,2]
        // Output: ComputingTensor[OFFSET_8..OFFSET_11]
        // ============================================================
        {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::MaskReg mask =
                    AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
                AscendC::MicroAPI::RegTensor<float> a1, a2, b1, b2;
                AscendC::MicroAPI::RegTensor<float> r00, r01, r02;     // cov_2x3 row 0
                AscendC::MicroAPI::RegTensor<float> r10, r11, r12;     // cov_2x3 row 1
                AscendC::MicroAPI::RegTensor<float> tmp, acc;
                constexpr uint16_t LANE = 64;

                for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                    const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        a1, p_compute + COMPUTE_OFFSET_3 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        a2, p_compute + COMPUTE_OFFSET_4 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        b1, p_compute + COMPUTE_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        b2, p_compute + COMPUTE_OFFSET_6 * taskNumPerCurLoop + off_lane, 1, mask);

                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        r00, p_covars + 0              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        r01, p_covars + 1              * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        r02, p_covars + COVARS_OFFSET_2 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        r10, p_covars + COVARS_OFFSET_3 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        r11, p_covars + COVARS_OFFSET_4 * taskNumPerCurLoop + off_lane, 1, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        r12, p_covars + COVARS_OFFSET_5 * taskNumPerCurLoop + off_lane, 1, mask);

                    // cov2d[0,0] = a1*r00 + a2*r02
                    AscendC::MicroAPI::Mul(acc, a1, r00, mask);
                    AscendC::MicroAPI::Mul(tmp, a2, r02, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + COMPUTE_OFFSET_8 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov2d[0,1] = b1*r01 + b2*r02
                    AscendC::MicroAPI::Mul(acc, b1, r01, mask);
                    AscendC::MicroAPI::Mul(tmp, b2, r02, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + COMPUTE_OFFSET_9 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                    // cov2d[1,0] omitted: equals cov2d[0,1] by symmetry, and
                    // downstream Det/Conics consume a01 only.

                    // cov2d[1,1] = b1*r11 + b2*r12
                    AscendC::MicroAPI::Mul(acc, b1, r11, mask);
                    AscendC::MicroAPI::Mul(tmp, b2, r12, mask);
                    AscendC::MicroAPI::Add(acc, acc, tmp, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + COMPUTE_OFFSET_11 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                }
            }
        }
    }
    __aicore__ inline void ComputingDet()
    {
        // 2D-covariance determinant (and optional compensation). a00/a01/a11
        // are loaded once per lane and held register-resident across the pre-eps
        // and post-eps passes:
        //   if calcCompensations: det_orig = a00*a11 - a01*a10   (pre-eps)
        //   a00 += eps2d; a11 += eps2d                            (regularize)
        //   det = max(a00*a11 - a01*a10, DET_MIN)                 (post-eps)
        //   if calcCompensations: compensation = sqrt(max(det_orig/det, 0))
        // The regularized a00/a11, det and compensation are written back to UB
        // for ComputingConics.

        __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
        const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a00, a01, a11;
            AscendC::MicroAPI::RegTensor<float> det_orig, det;
            AscendC::MicroAPI::RegTensor<float> tmp, acc;
            constexpr uint16_t LANE = 64;

            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a00, p_compute + COMPUTE_OFFSET_8  * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a01, p_compute + COMPUTE_OFFSET_9  * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a11, p_compute + COMPUTE_OFFSET_11 * taskNumPerCurLoop + off_lane, 1, mask);

                // Pre-eps det_orig (only if compensation requested)
                if (calcCompensations) {
                    AscendC::MicroAPI::Mul(det_orig, a00, a11, mask);
                    // a01*a10 = a01^2 by cov2d symmetry.
                    AscendC::MicroAPI::Mul(tmp,      a01, a01, mask);
                    AscendC::MicroAPI::Sub(det_orig, det_orig, tmp, mask);
                }

                // In-place +eps2d, store back to UB so ComputingConics reads the
                // regularized values.
                AscendC::MicroAPI::Adds(a00, a00, eps2d, mask);
                AscendC::MicroAPI::Adds(a11, a11, eps2d, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_8  * taskNumPerCurLoop + off_lane, a00, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_11 * taskNumPerCurLoop + off_lane, a11, 1, mask);

                // det = max(a00*a11 - a01^2, DET_MINIMUM_VALUE); cov2d is
                //        symmetric so a10 == a01 -> a01*a10 = a01^2.
                AscendC::MicroAPI::Mul(acc, a00, a11, mask);
                AscendC::MicroAPI::Mul(tmp, a01, a01, mask);
                AscendC::MicroAPI::Sub(acc, acc, tmp, mask);
                AscendC::MicroAPI::Maxs(det, acc, DET_MINIMUM_VALUE, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_compute + COMPUTE_OFFSET_3 * taskNumPerCurLoop + off_lane, det, 1, mask);

                if (calcCompensations) {
                    // compensation = sqrt(max(det_orig / det, 0))
                    AscendC::MicroAPI::Div(acc, det_orig, det, mask);
                    AscendC::MicroAPI::Maxs(acc, acc, ZERO_FLOAT_VALUE, mask);
                    AscendC::MicroAPI::Sqrt(acc, acc, mask);
                    AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                        p_compute + COMPUTE_OFFSET_7 * taskNumPerCurLoop + off_lane, acc, 1, mask);
                }
            }
        }
    }
    __aicore__ inline void ComputingConics()
    {
        // Conic (inverse 2D covariance): load cov2d + det once per lane and
        // compute the three outputs in registers:
        //   c00 = a11 / det
        //   c01 = -0.5 * (a01 + a10) / det
        //   c11 = a00 / det

        __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
        __ubuf__ float* p_covars  = (__ubuf__ float*) CovarsTensor.GetPhyAddr();
        __ubuf__ float* p_con = dbParity ? (__ubuf__ float*) ConicsDbTensor.GetPhyAddr() : p_covars;
        const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a00, a01, a11, det;
            AscendC::MicroAPI::RegTensor<float> tmp, acc;
            constexpr uint16_t LANE = 64;

            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a00, p_compute + COMPUTE_OFFSET_8  * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a01, p_compute + COMPUTE_OFFSET_9  * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a11, p_compute + COMPUTE_OFFSET_11 * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    det, p_compute + COMPUTE_OFFSET_3  * taskNumPerCurLoop + off_lane, 1, mask);

                // c00 = a11 / det
                AscendC::MicroAPI::Div(acc, a11, det, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_con + 0 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                // c01 = -a01 / det  (cov2d symmetric -> (a01+a10)/2 = a01)
                AscendC::MicroAPI::Div(tmp, a01, det, mask);
                AscendC::MicroAPI::Muls(acc, tmp, -1.0f, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_con + 1 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                // c11 = a00 / det
                AscendC::MicroAPI::Div(acc, a00, det, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_con + COVARS_OFFSET_2 * taskNumPerCurLoop + off_lane, acc, 1, mask);
            }
        }
    }

    __aicore__ inline void ComputingRadius()
    {
        // Screen-space radius per axis. a00/a11/det are loaded once and the
        // intermediates stay register-resident; the ceil is inlined per axis:
        //   b = 0.5 * (a00 + a11)
        //   tmp_inner = sqrt(max(b*b - det, EPSILON))
        //   radius_x = ceil(RADIUS_COEFF * min(sqrt(a00), sqrt(b + tmp_inner)))
        //   radius_y = ceil(RADIUS_COEFF * min(sqrt(a11), sqrt(b + tmp_inner)))

        __ubuf__ float* p_compute = (__ubuf__ float*) ComputingTensor.GetPhyAddr();
        __ubuf__ float* p_covars  = (__ubuf__ float*) CovarsTensor.GetPhyAddr();
        const uint16_t lanesPerChunk = static_cast<uint16_t>(taskNumPerCurLoop / 64);

        __VEC_SCOPE__
        {
            AscendC::MicroAPI::MaskReg mask =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg mask_i =
                AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> a00, a11, det;
            AscendC::MicroAPI::RegTensor<float> b, tmp_inner, r1;
            AscendC::MicroAPI::RegTensor<float> sqrt_a00, sqrt_a11;
            AscendC::MicroAPI::RegTensor<float> rx, ry;       // intermediate radii before ceil
            AscendC::MicroAPI::RegTensor<int32_t> i32_tmp;
            AscendC::MicroAPI::RegTensor<float> floor_x, frac, indicator;
            AscendC::MicroAPI::RegTensor<float> tmp, acc;
            constexpr uint16_t LANE = 64;

            for (uint16_t lane = 0; lane < lanesPerChunk; ++lane) {
                const uint32_t off_lane = static_cast<uint32_t>(lane) * LANE;

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a00, p_compute + COMPUTE_OFFSET_8  * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    a11, p_compute + COMPUTE_OFFSET_11 * taskNumPerCurLoop + off_lane, 1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    det, p_compute + COMPUTE_OFFSET_3  * taskNumPerCurLoop + off_lane, 1, mask);

                // b = 0.5 * (a00 + a11)
                AscendC::MicroAPI::Add(b, a00, a11, mask);
                AscendC::MicroAPI::Muls(b, b, 0.5f, mask);

                // tmp_inner = sqrt(max(b*b - det, EPSILON))
                AscendC::MicroAPI::Mul(acc, b, b, mask);
                AscendC::MicroAPI::Sub(acc, acc, det, mask);
                AscendC::MicroAPI::Maxs(acc, acc, EPSILON, mask);
                AscendC::MicroAPI::Sqrt(tmp_inner, acc, mask);

                // min(sqrt(a), sqrt(b)) = sqrt(min(a, b)) for a,b >= 0 (sqrt is
                // monotone), so take the min on the radicands and sqrt once per
                // axis. acc = b + tmp_inner is the shared radicand.
                AscendC::MicroAPI::Add(acc, b, tmp_inner, mask);

                // rx = sqrt(min(a00, bt)) * RADIUS_COEFF
                AscendC::MicroAPI::Min(rx, a00, acc, mask);
                AscendC::MicroAPI::Sqrt(rx, rx, mask);
                AscendC::MicroAPI::Muls(rx, rx, RADIUS_COEFF, mask);

                // ry = sqrt(min(a11, bt)) * RADIUS_COEFF
                AscendC::MicroAPI::Min(ry, a11, acc, mask);
                AscendC::MicroAPI::Sqrt(ry, ry, mask);
                AscendC::MicroAPI::Muls(ry, ry, RADIUS_COEFF, mask);

                // ceil(rx) via truncating cast + fractional-indicator add
                AscendC::MicroAPI::Cast<int32_t, float, VEC_CEIL_TRAIT>(i32_tmp, rx, mask_i);
                AscendC::MicroAPI::Cast<float, int32_t, VEC_CEIL_TRAIT>(floor_x, i32_tmp, mask);
                AscendC::MicroAPI::Sub(frac, rx, floor_x, mask);
                AscendC::MicroAPI::Muls(frac, frac, 1.0e10f, mask);
                AscendC::MicroAPI::Mins(indicator, frac, 1.0f, mask);
                AscendC::MicroAPI::Add(acc, floor_x, indicator, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_3 * taskNumPerCurLoop + off_lane, acc, 1, mask);

                // ceil(ry)
                AscendC::MicroAPI::Cast<int32_t, float, VEC_CEIL_TRAIT>(i32_tmp, ry, mask_i);
                AscendC::MicroAPI::Cast<float, int32_t, VEC_CEIL_TRAIT>(floor_x, i32_tmp, mask);
                AscendC::MicroAPI::Sub(frac, ry, floor_x, mask);
                AscendC::MicroAPI::Muls(frac, frac, 1.0e10f, mask);
                AscendC::MicroAPI::Mins(indicator, frac, 1.0f, mask);
                AscendC::MicroAPI::Add(acc, floor_x, indicator, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
                    p_covars + COVARS_OFFSET_4 * taskNumPerCurLoop + off_lane, acc, 1, mask);
            }
        }
    }

    __aicore__ inline void CopyOut(uint64_t batchCopyoutIndex, uint64_t cameraCopyoutIndex, uint64_t taskCopyIndex)
    {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        // All writes go through DataCopyPad with blockLen =
        // actualValidCount * sizeof(float), capping GM writes to the real valid
        // gaussian count so they never run out of bounds.
        LocalTensor<float> m2dSrc = dbParity ? Means2dDbTensor : ComputingTensor;
        LocalTensor<float> conSrc = dbParity ? ConicsDbTensor  : CovarsTensor;
        DataCopyExtParams co_params{
            /*blockCount=*/1,
            /*blockLen=*/static_cast<uint32_t>(actualValidCount * sizeof(float)),
            /*srcStride=*/0,
            /*dstStride=*/0,
            /*rsv=*/0
        };

        DataCopyPad(depthsGM[batchCopyoutIndex + cameraCopyoutIndex + taskCopyIndex],
                    MeansTensor[taskNumPerCurLoop * MEANS_OFFSET_2], co_params);
        DataCopyPad(detGM[batchCopyoutIndex + cameraCopyoutIndex + taskCopyIndex],
                    ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_3], co_params);

        DataCopyPad(radiusGM[(batchCopyoutIndex + cameraCopyoutIndex) * RADIUS_TAIL_DIM + taskCopyIndex],
                    CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_3], co_params);
        DataCopyPad(radiusGM[(batchCopyoutIndex + cameraCopyoutIndex) * RADIUS_TAIL_DIM + taskCopyIndex + gaussianNum],
                    CovarsTensor[taskNumPerCurLoop * COVARS_OFFSET_4], co_params);

        DataCopyPad(means2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * MEANS2D_TAIL_DIM + taskCopyIndex],
                    m2dSrc, co_params);
        DataCopyPad(means2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * MEANS2D_TAIL_DIM + taskCopyIndex + gaussianNum],
                    m2dSrc[taskNumPerCurLoop], co_params);

        DataCopyPad(conicsGM[(batchCopyoutIndex + cameraCopyoutIndex) * CONICS_TAIL_DIM + taskCopyIndex],
                    conSrc, co_params);
        DataCopyPad(conicsGM[(batchCopyoutIndex + cameraCopyoutIndex) * CONICS_TAIL_DIM + taskCopyIndex + gaussianNum],
                    conSrc[taskNumPerCurLoop], co_params);
        DataCopyPad(conicsGM[(batchCopyoutIndex + cameraCopyoutIndex) * CONICS_TAIL_DIM + taskCopyIndex
                             + gaussianNum * COVARS_OFFSET_2],
                    conSrc[taskNumPerCurLoop * COVARS_OFFSET_2], co_params);

        DataCopyPad(covars2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * COVARS2D_TAIL_DIM + taskCopyIndex],
                    ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_8], co_params);
        DataCopyPad(covars2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * COVARS2D_TAIL_DIM + taskCopyIndex + gaussianNum],
                    ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_9], co_params);
        DataCopyPad(covars2dGM[(batchCopyoutIndex + cameraCopyoutIndex) * COVARS2D_TAIL_DIM + taskCopyIndex
                               + gaussianNum * COVARS_OFFSET_2],
                    ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_11], co_params);

        if (calcCompensations) {
            DataCopyPad(compensationsGM[batchCopyoutIndex + cameraCopyoutIndex + taskCopyIndex],
                        ComputingTensor[taskNumPerCurLoop * COMPUTE_OFFSET_7], co_params);
        }

        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        }

private:
    TPipe *_pipe;
    TBuf <TPosition::VECCALC> MeansTensorBuffer, CovarsTensorBuffer;
    TBuf <TPosition::VECCALC> ComputingTensorBuffer, CameraParamTensorBuffer;
    TBuf <TPosition::VECCALC> MeansWorldTensorBuffer, CovarsWorldTensorBuffer;  // persistent world-frame inputs
    TBuf <TPosition::VECCALC> QuatTensorBuffer, ScalesTensorBuffer;  // allocated only when inputIsQuatScales=1
    TBuf <TPosition::VECCALC> ConicsDbBuffer, Covars2dDbBuffer, Means2dDbBuffer;  // alternate output slots (double-buffer)

    LocalTensor<float> MeansTensor, CovarsTensor;
    LocalTensor<float> MeansWorldTensor, CovarsWorldTensor;  // world means/covars, separate from working buffers
    LocalTensor<float> ComputingTensor, CameraParamTensor;
    LocalTensor<float> QuatTensor, ScalesTensor;  // used iff inputIsQuatScales=1
    LocalTensor<float> ConicsDbTensor, Covars2dDbTensor, Means2dDbTensor;  // alternate conics/covars2d/means2d

    GlobalTensor<DTYPE_MEANS> meansGM;
    GlobalTensor<DTYPE_COVARS> covarsGM;
    GlobalTensor<DTYPE_VIEWMATS> viewmatsGM;
    GlobalTensor<DTYPE_KS> ksGM;
    // optional quat/scales inputs (DTYPE_* generated from IR).
    GlobalTensor<DTYPE_QUAT> quatGM;
    GlobalTensor<DTYPE_SCALES> scalesGM;
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
    // actual valid gaussian count in the current loop (<= taskNumPerCurLoop);
    // used by CopyOut as blockLen so writes stay within GM bounds.
    uint32_t actualValidCount;
    float eps2d, imageWidth, imageHeight;
    bool calcCompensations;
    uint32_t inputIsQuatScales;  // 0=covars-direct, 1=fused (read quat/scales)
    uint32_t dbParity = 0;  // selects working (0) vs alternate (1) output buffers
};

extern "C" __global__ __aicore__ void projection_three_dims_gaussian_forward(GM_ADDR means, GM_ADDR covars, \
                                                                             GM_ADDR viewmats, GM_ADDR ks, \
                                                                             GM_ADDR quat, GM_ADDR scales, \
                                                                             GM_ADDR means2d, GM_ADDR depths, \
                                                                             GM_ADDR conics, GM_ADDR compensations, \
                                                                             GM_ADDR det, GM_ADDR radius, \
                                                                             GM_ADDR covars2d, GM_ADDR workspace, \
                                                                             GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    // key 1 = multi-camera (taskLoop-major), key 2 = single-camera
    // (camera-major). Host picks the key by cameraNum.
    if (TILING_KEY_IS(1)) {
        ProjectionThreeDimsGaussianForward op;
        op.GetTilingData(&tiling_data);
        op.GMInit(means, covars, viewmats, ks, quat, scales, means2d, depths, conics, compensations, det, radius, covars2d);
        op.Init(&pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        ProjectionThreeDimsGaussianForward op;
        op.GetTilingData(&tiling_data);
        op.GMInit(means, covars, viewmats, ks, quat, scales, means2d, depths, conics, compensations, det, radius, covars2d);
        op.Init(&pipe);
        op.ProcessCameraMajor();
    }
}