/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/torch.h>
#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;

namespace {
void CheckTensor(const at::Tensor &tensor, const at::Tensor &gs,
                 at::ScalarType dtype, const char *name)
{
    TORCH_CHECK(tensor.device() == gs.device(), name, ": inconsistent device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, ": unexpected dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, ": expected contiguous tensor");
}
void CheckInputs(const at::Tensor &gs, const at::Tensor &coords,
                 const at::Tensor &offsets, const at::Tensor &ids)
{
    TORCH_CHECK(gs.device().type() == at::kPrivateUse1, "SIMT rendering requires NPU tensors");
    CheckTensor(gs, gs, at::kFloat, "gs");
    CheckTensor(coords, gs, at::kFloat, "tileCoords");
    CheckTensor(offsets, gs, at::kLong, "offsets");
    CheckTensor(ids, gs, at::kLong, "gsIds");
    TORCH_CHECK(gs.dim() == 2 && gs.size(1) == 10, "gs must have shape [N,10]");
    TORCH_CHECK(coords.dim() == 3 && coords.size(1) == 2, "tileCoords must have shape [T,2,P]");
    TORCH_CHECK(coords.size(2) > 0 && coords.size(2) <= 1024, "SIMT rendering requires 1 <= P <= 1024");
    TORCH_CHECK(offsets.dim() == 1 && ids.dim() == 1, "offsets and gsIds must be vectors");
}
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> calc_render_simt_fwd(
    const at::Tensor &gs, const at::Tensor &tileCoords,
    const at::Tensor &offsets, const at::Tensor &gsIds)
{
    CheckInputs(gs, tileCoords, offsets, gsIds);
    const auto t = tileCoords.size(0), p = tileCoords.size(2);
    auto color = at::zeros({3, t, p}, gs.options());
    auto depth = at::zeros({t, p}, gs.options());
    auto lastCumsum = at::zeros({t, p}, gs.options());
    auto error = at::zeros({t, p}, gs.options());
    auto lastIds = at::full({t, p}, -1, gs.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnCalcRenderSimtFwd, gs, tileCoords, offsets, gsIds,
                 color, depth, lastCumsum, error, lastIds);
    return std::make_tuple(color, depth, lastCumsum, error, lastIds);
}

at::Tensor calc_render_simt_bwd(
    const at::Tensor &vColor, const at::Tensor &vDepth,
    const at::Tensor &lastCumsum, const at::Tensor &error,
    const at::Tensor &gs, const at::Tensor &tileCoords,
    const at::Tensor &offsets, const at::Tensor &gsIds, const at::Tensor &lastIds)
{
    CheckInputs(gs, tileCoords, offsets, gsIds);
    CheckTensor(vColor, gs, at::kFloat, "vColor");
    CheckTensor(vDepth, gs, at::kFloat, "vDepth");
    CheckTensor(lastCumsum, gs, at::kFloat, "lastCumsum");
    CheckTensor(error, gs, at::kFloat, "error");
    CheckTensor(lastIds, gs, at::kInt, "lastIds");
    const auto t = tileCoords.size(0), p = tileCoords.size(2);
    TORCH_CHECK(vColor.sizes() == at::IntArrayRef({3, t, p}), "vColor must have shape [3,T,P]");
    TORCH_CHECK(vDepth.sizes() == at::IntArrayRef({1, t, p}), "vDepth must have shape [1,T,P]");
    TORCH_CHECK(lastCumsum.sizes() == at::IntArrayRef({t, p}) &&
                error.sizes() == lastCumsum.sizes() && lastIds.sizes() == lastCumsum.sizes(),
                "lastCumsum, error and lastIds must have shape [T,P]");
    auto vGs = at::zeros_like(gs);
    EXEC_NPU_CMD(aclnnCalcRenderSimtBwd, vColor, vDepth, lastCumsum, error,
                 gs, tileCoords, offsets, gsIds, lastIds, vGs);
    return vGs;
}
