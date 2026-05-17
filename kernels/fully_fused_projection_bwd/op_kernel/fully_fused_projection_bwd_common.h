/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FULLY_FUSED_PROJECTION_BWD_COMMON_H
#define FULLY_FUSED_PROJECTION_BWD_COMMON_H

#include "kernel_operator.h"

namespace FullyFusedProjectionBwdNs {
using namespace AscendC;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT32_ONE_BLOCK_NUM = 8;
constexpr int64_t FLOAT_ONE_BLOCK_NUM = 8;
constexpr int64_t INT64_ONE_BLOCK_NUM = 4;
constexpr int64_t CONCAT_AGLIN_VALUE = 16;
constexpr int64_t INT32_TO_INT16 = 2;
constexpr int64_t FLOAT_SIZE = 4;
constexpr int32_t INT32_SIZE = 4;
constexpr int64_t INT16_SIZE = 2;
constexpr int64_t UINT8_BITS = 8;
constexpr int64_t MASK_BATCH_SIZE = 256;
constexpr int64_t ROTMAT_DIM = 3;
constexpr int64_t VR_DIM = 3;
constexpr int64_t M_DIM = 3;
constexpr int64_t VPW_ELEMENT = 3;
constexpr int64_t CONICS_ELEMENT = 3;
constexpr int64_t VCOLORSCULLING_ELEMENT = 3;
constexpr int64_t VOPACITIESCULLING_ELEMENT = 1;
constexpr int64_t COMPENSATIONS_ELEMENT = 1;
constexpr int64_t MEANS_ELEMENT = 3;
constexpr int64_t SCALES_ELEMENT = 3;
constexpr int64_t QUAT_ELEMENT = 4;
constexpr int64_t VIEWMATS_ELEMENT = 16;
constexpr int64_t VR_ELEMENT = 9;
constexpr int64_t KS_ELEMENT = 9;
constexpr int64_t FX_INDEX = 0;
constexpr int64_t FY_INDEX = 4;
constexpr int64_t CX_INDEX = 2;
constexpr int64_t CY_INDEX = 5;
constexpr float ONE_FLOAT_VALUE = 1.0f;
constexpr float MINUS_ONE_FLOAT_VALUE = -1.0f;
constexpr float TWO_FLOAT_VALUE = 2.0f;
constexpr int64_t COPYINQUE_LEN = 4;
constexpr int64_t INPUTBUF_LEN = 19;
constexpr int64_t INTERMEDIATEBUF_LEN = 49;
constexpr int64_t CALBUF_LEN = 16;
constexpr int64_t MASKBUF_LEN = 8;
constexpr int64_t WORKSPACE_CNTPERCORE_OFFSET = 96;
constexpr int64_t WORKSPACE_SORTEDIDX_OFFSET = 2;
constexpr int64_t RTBUF_LEN = 24;
constexpr int64_t FILTERBUF_CONSTOFFSET = 2;
constexpr int32_t SORT_ALIGNCONST = 32;
constexpr int64_t FILTERNOTINDEX_OFFSET = 2;
constexpr int64_t FILTERINDEX2_OFFSET = 4;
constexpr int64_t FILTERINDEX1_OFFSET = 5;
constexpr int64_t STARTVALUE_ZERO = 0;
constexpr int64_t PADVALUE_ZERO = 0;
constexpr float PADVALUE_ZERO_FLOAT = 0;
constexpr int64_t DIFFSTEP_ONE = 1;
constexpr int64_t SRC0BLOCKSTRIDE = 1;
constexpr int64_t REPEATTIMES = 1;
constexpr int64_t SRC0REPEATSTRIDE = 8;
constexpr int64_t SRC1REPEATSTRIDE = 0;
constexpr int64_t FILTER_ALIGNCONST = 8;
constexpr int64_t SORTEDLOCAL_OFFSET = 2;
constexpr int64_t SORTEDTEMP_OFFSET = 6;
constexpr int64_t DSTVALUELOCAL_OFFSET = 4;
constexpr int64_t NEWINDEXFP32_OFFSET = 0;
constexpr int64_t COPYINFILTER_OFFSET = 39 * 4;
constexpr int64_t MAXCORENUM = 48;
constexpr int64_t VIEWMAT_INDEX11 = 0;
constexpr int64_t VIEWMAT_INDEX12 = 1;
constexpr int64_t VIEWMAT_INDEX13 = 2;
constexpr int64_t VIEWMAT_INDEX14 = 3;
constexpr int64_t VIEWMAT_INDEX21 = 4;
constexpr int64_t VIEWMAT_INDEX22 = 5;
constexpr int64_t VIEWMAT_INDEX23 = 6;
constexpr int64_t VIEWMAT_INDEX24 = 7;
constexpr int64_t VIEWMAT_INDEX31 = 8;
constexpr int64_t VIEWMAT_INDEX32 = 9;
constexpr int64_t VIEWMAT_INDEX33 = 10;
constexpr int64_t VIEWMAT_INDEX34 = 11;
constexpr int64_t RMAT_INDEX11 = 0;
constexpr int64_t RMAT_INDEX12 = 1;
constexpr int64_t RMAT_INDEX13 = 2;
constexpr int64_t RMAT_INDEX21 = 3;
constexpr int64_t RMAT_INDEX22 = 4;
constexpr int64_t RMAT_INDEX23 = 5;
constexpr int64_t RMAT_INDEX31 = 6;
constexpr int64_t RMAT_INDEX32 = 7;
constexpr int64_t RMAT_INDEX33 = 8;
constexpr int64_t TMAT_INDEX11 = 0;
constexpr int64_t TMAT_INDEX12 = 1;
constexpr int64_t TMAT_INDEX13 = 2;
constexpr int64_t RVEC_OFFSET = 0;
constexpr int64_t XVEC_OFFSET = 1;
constexpr int64_t YVEC_OFFSET = 2;
constexpr int64_t ZVEC_OFFSET = 3;
constexpr int64_t QUANTNORM_TMPBUF_OFFSET = 21;
constexpr int64_t QUANTNORM_TMPR2_OFFSET = 0;
constexpr int64_t QUANTNORM_TMPX2_OFFSET = 1;
constexpr int64_t QUANTNORM_TMPY2_OFFSET = 2;
constexpr int64_t QUANTNORM_TMPZ2_OFFSET = 3;
constexpr int64_t QUANTNORM_TMPINVNORM_OFFSET = 4;
constexpr int64_t QUANTNORM_RNORM_OFFSET = 5;
constexpr int64_t QUANTNORM_XNORM_OFFSET = 6;
constexpr int64_t QUANTNORM_YNORM_OFFSET = 7;
constexpr int64_t QUANTNORM_ZNORM_OFFSET = 8;
constexpr float AVOID_ZERO = 1e-12f;
constexpr int64_t QUANTOROT_RNORM_OFFSET = 26;
constexpr int64_t QUANTOROT_XNORM_OFFSET = 27;
constexpr int64_t QUANTOROT_YNORM_OFFSET = 28;
constexpr int64_t QUANTOROT_ZNORM_OFFSET = 29;
constexpr int64_t QUANTOROT_TMPBUF_OFFSET = 30;
constexpr int64_t QUANTOROT_TMPR2_OFFSET = 0;
constexpr int64_t QUANTOROT_TMPX2_OFFSET = 1;
constexpr int64_t QUANTOROT_TMPY2_OFFSET = 2;
constexpr int64_t QUANTOROT_TMPZ2_OFFSET = 3;
constexpr int64_t QUANTOROT_TMPRX_OFFSET = 4;
constexpr int64_t QUANTOROT_TMPRY_OFFSET = 5;
constexpr int64_t QUANTOROT_TMPRZ_OFFSET = 6;
constexpr int64_t QUANTOROT_TMPXY_OFFSET = 7;
constexpr int64_t QUANTOROT_TMPXZ_OFFSET = 8;
constexpr int64_t QUANTOROT_TMPYZ_OFFSET = 9;
constexpr int64_t QUANTOROT_R00_OFFSET = 10;
constexpr int64_t QUANTOROT_R01_OFFSET = 1;
constexpr int64_t QUANTOROT_R02_OFFSET = 2;
constexpr int64_t QUANTOROT_R10_OFFSET = 3;
constexpr int64_t QUANTOROT_R11_OFFSET = 4;
constexpr int64_t QUANTOROT_R12_OFFSET = 5;
constexpr int64_t QUANTOROT_R20_OFFSET = 6;
constexpr int64_t QUANTOROT_R21_OFFSET = 7;
constexpr int64_t QUANTOROT_R22_OFFSET = 8;
constexpr int64_t INVJP_TMPBUF_OFFSET = 26;
constexpr float HALFVC1_VALUE = 0.5f;
constexpr int64_t INVJP_INTERCAL_OFFSET = 27;
constexpr int64_t C0_OFFSET = 0;
constexpr int64_t C1_OFFSET = 1;
constexpr int64_t C2_OFFSET = 2;
constexpr float INTERCAL_VALUE = 2.0f;
constexpr int64_t QUANTSCALE_TMP1BUF_OFFSET = 21;
constexpr int64_t QUANTSCALE_TMP2BUF_OFFSET = 30;
constexpr int64_t QUANTSCALE_TMP1BUF_LEN = 9;
constexpr int64_t QUANTSCALE_TMP2BUF_LEN = 2;
constexpr int64_t LEFTMAT_INDEX11 = 0;
constexpr int64_t LEFTMAT_INDEX12 = 1;
constexpr int64_t LEFTMAT_INDEX13 = 2;
constexpr int64_t LEFTMAT_INDEX21 = 3;
constexpr int64_t LEFTMAT_INDEX22 = 4;
constexpr int64_t LEFTMAT_INDEX23 = 5;
constexpr int64_t LEFTMAT_INDEX31 = 6;
constexpr int64_t LEFTMAT_INDEX32 = 7;
constexpr int64_t LEFTMAT_INDEX33 = 8;
constexpr int64_t TMP2SUM1_OFFSET = 0;
constexpr int64_t TMP2SUM2_OFFSET = 1;
constexpr int64_t COVMAT_INDEX11 = 0;
constexpr int64_t COVMAT_INDEX12 = 1;
constexpr int64_t COVMAT_INDEX13 = 2;
constexpr int64_t COVMAT_INDEX22 = 3;
constexpr int64_t COVMAT_INDEX23 = 4;
constexpr int64_t COVMAT_INDEX33 = 5;
constexpr int64_t MEANS1_OFFSET = 0;
constexpr int64_t MEANS2_OFFSET = 1;
constexpr int64_t MEANS3_OFFSET = 2;
constexpr int64_t REDUCESUMRESULT_OFFSET = 10;
constexpr int64_t REDUCESUMTMP_OFFSET = 9;
constexpr int64_t VPWTMP_OFFSET = 9;
constexpr int64_t VPWTMP1_OFFSET = 0;
constexpr int64_t VPWTMP2_OFFSET = 1;
constexpr int64_t VPWTMP3_OFFSET = 2;
constexpr int64_t VMEANSC1_OFFSET = 0;
constexpr int64_t VMEANSC2_OFFSET = 1;
constexpr int64_t VMEANSC3_OFFSET = 2;
constexpr int64_t COVARSW2C_BUF1_OFFSET = 27;
constexpr int64_t COVARSW2C_BUF2_OFFSET = 9;
constexpr int64_t COVARSW2C_BUFLEN = 9;
constexpr float TANFOV_COE = 0.5f;
constexpr float LIM_COE = 0.3f;
constexpr int64_t VMEANSC2D1_OFFSET = 0;
constexpr int64_t VMEANSC2D2_OFFSET = 1;
constexpr int64_t VCOV2D1_OFFSET = 0;
constexpr int64_t VCOV2D2_OFFSET = 1;
constexpr int64_t VCOV2D3_OFFSET = 2;
constexpr int64_t TXDIVTZ_OFFSET = 0;
constexpr int64_t TYDIVTZ_OFFSET = 1;
constexpr int64_t FXDIVTZ_OFFSET = 2;
constexpr int64_t FYDIVTZ_OFFSET = 3;
constexpr int64_t PERSP_TMPBUF0_OFFSET = 4;
constexpr int64_t PERSP_TMPBUF1_OFFSET = 5;
constexpr int64_t PERSP_TMPBUF2_OFFSET = 6;
constexpr int64_t PERSP_TMPBUF3_OFFSET = 7;
constexpr int64_t PERSP_TMPBUF4_OFFSET = 12;
constexpr int64_t XCLIPMASK0_OFFSET = 0;
constexpr int64_t XCLIPMASK1_OFFSET = 1;
constexpr int64_t YCLIPMASK0_OFFSET = 2;
constexpr int64_t YCLIPMASK1_OFFSET = 3;
constexpr int64_t VCOVARC00_OFFSET = 0;
constexpr int64_t VCOVARC01_OFFSET = 1;
constexpr int64_t VCOVARC02_OFFSET = 2;
constexpr int64_t VCOVARC11_OFFSET = 3;
constexpr int64_t VCOVARC12_OFFSET = 4;
constexpr int64_t VCOVARC22_OFFSET = 5;
constexpr int64_t COVARC00_OFFSET = 0;
constexpr int64_t COVARC01_OFFSET = 1;
constexpr int64_t COVARC02_OFFSET = 2;
constexpr int64_t COVARC11_OFFSET = 3;
constexpr int64_t COVARC12_OFFSET = 4;
constexpr int64_t COVARC22_OFFSET = 5;
constexpr int64_t A10_OFFSET = 9;
constexpr int64_t A11_OFFSET = 10;
constexpr int64_t A12_OFFSET = 11;
constexpr int64_t VJ00_OFFSET = 0;
constexpr int64_t VJ01_OFFSET = 1;
constexpr int64_t VJ02_OFFSET = 2;
constexpr int64_t FXDIVTZ2_OFFSET = 9;
constexpr int64_t FYDIVTZ2_OFFSET = 10;
constexpr float VMEANC2_VALUE = -2.0f;
constexpr int64_t M_FISRTLINE_OFFSET = 0;
constexpr int64_t M_SECONDLINE_OFFSET = 3;
constexpr int64_t M_THIRDLINE_OFFSET = 6;
constexpr int64_t VM_OFFSET = 15;
constexpr int64_t QUATSCALETOCO_TEMP_OFFSET = 24;
constexpr int64_t VM_LEN = 9;
constexpr int64_t VRMAT_FIRSTLINE_OFFSET = 0;
constexpr int64_t VRMAT_SECONDLINE_OFFSET = 3;
constexpr int64_t VRMAT_THIRDLINE_OFFSET = 6;
constexpr int64_t VMMAT_FIRSTLINE_OFFSET = 0;
constexpr int64_t VMMAT_SECONDLINE_OFFSET = 3;
constexpr int64_t VMMAT_THIRDLINE_OFFSET = 6;
constexpr int64_t LOCALINDEX_OFFSET = 5;
constexpr int64_t QUATSN_OFFSET = 15;
constexpr int64_t NORM_OFFSET = 19;
constexpr int64_t W_OFFSET = 0;
constexpr int64_t X_OFFSET = 1;
constexpr int64_t Y_OFFSET = 2;
constexpr int64_t Z_OFFSET = 3;
constexpr int64_t VR00_OFFSET = 0;
constexpr int64_t VR01_OFFSET = 1;
constexpr int64_t VR02_OFFSET = 2;
constexpr int64_t VR10_OFFSET = 3;
constexpr int64_t VR11_OFFSET = 4;
constexpr int64_t VR12_OFFSET = 5;
constexpr int64_t VR20_OFFSET = 6;
constexpr int64_t VR21_OFFSET = 7;
constexpr int64_t VR22_OFFSET = 8;
constexpr int64_t VQUATN_OFFSET = 23;
constexpr int64_t QUATSCALCO_TMP_OFFSET = 27;
constexpr int64_t VCOVAR_LEN = 6;
constexpr int64_t QUATS_OFFSET = 3;
constexpr int64_t ROT_OFFSET = 6;
constexpr int64_t SCALES_OFFSET = 7;
constexpr int64_t COVARS_OFFSET = 15;
constexpr int64_t COVAR_LEN = 9;
constexpr int64_t VCOLORSCCULLINGSUM_OFFSET = 45;
constexpr int64_t VOPACITIESCCULLINGSUM_OFFSET = 48;
constexpr int64_t VCOLORSCCULLINGSUM_LEN = 4;
constexpr int64_t CONICS_OFFSET = 10;
constexpr int64_t VCONICS_OFFSET = 14;
constexpr int64_t VCOVARS2D_OFFSET = 30;
constexpr int64_t VIEWMAT_OFFSET = 17;
constexpr int64_t RMAT_LEN = 16;
constexpr int64_t MEANSC_OFFSET = 36;
constexpr int64_t COVARSC_OFFSET = 39;
constexpr int64_t KS_OFFSET = 18;
constexpr int64_t VMEANS2D_OFFSET = 10;
constexpr int64_t VDEPTHS_OFFSET = 14;
constexpr int64_t VMEANSC_OFFSET = 27;
constexpr int64_t VCOVARC_OFFSET = 21;
constexpr int64_t VMEANS_CONSTDIM = 2;
constexpr int64_t VDEPTHS_CONSTDIM = 1;
constexpr int64_t VR_OFFSET = 14;
constexpr int64_t VCOLORSCULLING_OFFSET = 30;
constexpr int64_t VOPACITIESCULLING_OFFSET = 33;
constexpr int64_t COMPENSATION_OFFSET = 34;
constexpr int64_t VQUATS_OFFSET = 10;
constexpr int64_t VSCALES_OFFSET = 14;

constexpr int64_t MAT_LINE_LEN = 3;
constexpr int64_t MAT_IDX11 = 0;
constexpr int64_t MAT_IDX12 = 1;
constexpr int64_t MAT_IDX13 = 2;
constexpr int64_t MAT_IDX21 = 3;
constexpr int64_t MAT_IDX22 = 4;
constexpr int64_t MAT_IDX23 = 5;
constexpr int64_t MAT_IDX31 = 6;
constexpr int64_t MAT_IDX32 = 7;
constexpr int64_t MAT_IDX33 = 8;

__aicore__ inline int64_t Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
}

__aicore__ inline int64_t Align256(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + MASK_BATCH_SIZE - 1) / MASK_BATCH_SIZE * MASK_BATCH_SIZE / bytes;
}

__aicore__ inline int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T> __aicore__ inline T Min(T a, T b) { return a > b ? b : a; }

template <typename T> __aicore__ inline T Max(T a, T b) { return a < b ? b : a; }

__aicore__ inline int GetSymmetricIndex(int i, int j)
{
    int originIndex = i * MAT_LINE_LEN + j;
    switch (originIndex) {
        case MAT_IDX11:
            return MAT_IDX11;
        case MAT_IDX12:
            return MAT_IDX12;
        case MAT_IDX13:
            return MAT_IDX13;
        case MAT_IDX21:
            return MAT_IDX12;
        case MAT_IDX22:
            return MAT_IDX21;
        case MAT_IDX23:
            return MAT_IDX22;
        case MAT_IDX31:
            return MAT_IDX13;
        case MAT_IDX32:
            return MAT_IDX22;
        case MAT_IDX33:
            return MAT_IDX23;
        default:
            return 0;
    }
}

template <HardEvent event> __aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

} // namespace FullyFusedProjectionBwdNs
#endif // FULLY_FUSED_PROJECTION_BWD_COMMON_H