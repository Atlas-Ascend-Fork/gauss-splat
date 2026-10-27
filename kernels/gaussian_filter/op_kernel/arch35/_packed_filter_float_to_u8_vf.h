/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "tensorutils.h"

static constexpr MicroAPI::CastTrait _packed_filter_float_to_u8_vf_pack_half = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
static constexpr MicroAPI::CastTrait _packed_filter_float_to_u8_vf_pack_u8 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };

__aicore__ inline void _packed_filter_float_to_u8_vf(__ubuf__ float* src_float_ub, __ubuf__ half* tmp_half_ub, __ubuf__ uint8_t* dst_u8_ub){
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg _tmp_maskreg_2 = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg _tmp_maskreg_1 = MicroAPI::CreateMask<half, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg _tmp_maskreg_0 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<float> src_r;
        MicroAPI::RegTensor<half> half_r;
        MicroAPI::RegTensor<uint8_t> u8_r;
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(src_r, src_float_ub, 1, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, _packed_filter_float_to_u8_vf_pack_half>(half_r, src_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(tmp_half_ub, half_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(src_r, src_float_ub + 1 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, _packed_filter_float_to_u8_vf_pack_half>(half_r, src_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(tmp_half_ub + 1 * 128, half_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(src_r, src_float_ub + 2 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, _packed_filter_float_to_u8_vf_pack_half>(half_r, src_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(tmp_half_ub + 2 * 128, half_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(src_r, src_float_ub + 3 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, _packed_filter_float_to_u8_vf_pack_half>(half_r, src_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(tmp_half_ub + 3 * 128, half_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(src_r, src_float_ub + 4 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, _packed_filter_float_to_u8_vf_pack_half>(half_r, src_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(tmp_half_ub + 4 * 128, half_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(src_r, src_float_ub + 5 * 64, 1, _tmp_maskreg_0);
        MicroAPI::Cast<half, float, _packed_filter_float_to_u8_vf_pack_half>(half_r, src_r, _tmp_maskreg_1);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(tmp_half_ub + 5 * 128, half_r, _tmp_maskreg_1);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(half_r, tmp_half_ub, 1, _tmp_maskreg_1);
        MicroAPI::Cast<uint8_t, half, _packed_filter_float_to_u8_vf_pack_u8>(u8_r, half_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK_B16>(dst_u8_ub, u8_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(half_r, tmp_half_ub + 1 * 128, 1, _tmp_maskreg_1);
        MicroAPI::Cast<uint8_t, half, _packed_filter_float_to_u8_vf_pack_u8>(u8_r, half_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK_B16>(dst_u8_ub + 1 * 64, u8_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(half_r, tmp_half_ub + 2 * 128, 1, _tmp_maskreg_1);
        MicroAPI::Cast<uint8_t, half, _packed_filter_float_to_u8_vf_pack_u8>(u8_r, half_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK_B16>(dst_u8_ub + 2 * 64, u8_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(half_r, tmp_half_ub + 3 * 128, 1, _tmp_maskreg_1);
        MicroAPI::Cast<uint8_t, half, _packed_filter_float_to_u8_vf_pack_u8>(u8_r, half_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK_B16>(dst_u8_ub + 3 * 64, u8_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(half_r, tmp_half_ub + 4 * 128, 1, _tmp_maskreg_1);
        MicroAPI::Cast<uint8_t, half, _packed_filter_float_to_u8_vf_pack_u8>(u8_r, half_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK_B16>(dst_u8_ub + 4 * 64, u8_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<half, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(half_r, tmp_half_ub + 5 * 128, 1, _tmp_maskreg_1);
        MicroAPI::Cast<uint8_t, half, _packed_filter_float_to_u8_vf_pack_u8>(u8_r, half_r, _tmp_maskreg_2);
        MicroAPI::DataCopy<uint8_t, MicroAPI::StoreDist::DIST_PACK_B16>(dst_u8_ub + 5 * 64, u8_r, _tmp_maskreg_2);
    }
}

