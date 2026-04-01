// Copyright (c) 2025 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <string>
#include <torch/torch.h>
#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;
using namespace std;

at::Tensor calc_render_bwd_var_clip_gsids(
    const at::Tensor &vColor, const at::Tensor &vDepth, const at::Tensor &lastCumsum, const at::Tensor &error,
    const at::Tensor &gs, const at::Tensor &tileCoords,
    const at::Tensor &offsets, const at::Tensor &gsIds, const at::Tensor &gsClipIndex, const at::Tensor &alphaClipIndex)
{
    TORCH_CHECK(vColor.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(vDepth.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(lastCumsum.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(error.device() == vColor.device(), "Inconsistent device.");

    TORCH_CHECK(gs.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(tileCoords.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(offsets.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(gsIds.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(gsClipIndex.device() == vColor.device(), "Inconsistent device.");
    TORCH_CHECK(alphaClipIndex.device() == vColor.device(), "Inconsistent device.");

    auto device = vColor.device();
    auto options = at::TensorOptions().dtype(at::kFloat).layout(at::kStrided).device(device);

    int64_t nGauss = gs.sizes()[1];
    int64_t nPixel = vColor.sizes()[2];

    at::Tensor vGs = at::zeros_like(gs, options);
    at::Tensor gsClipIndex_gsIds = at::cat({gsClipIndex, gsIds}, 0); // dim = 0

    EXEC_NPU_CMD(aclnnCalcRenderBwdVarClipGsids,
        vColor, vDepth, lastCumsum, error,
        gs, tileCoords, offsets, gsClipIndex_gsIds,
        alphaClipIndex, vGs);
    return vGs;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> calc_render_fwd_double_clip_gsids(
    const at::Tensor &gs, const at::Tensor &tileCoords, const at::Tensor &offsets, const at::Tensor &gsIds)
{
    TORCH_CHECK(gs.device().type() == at::kPrivateUse1, "Invalid device.");
    TORCH_CHECK(tileCoords.device() == gs.device(), "Inconsistent device.");
    TORCH_CHECK(offsets.device() == gs.device(), "Inconsistent device.");
    TORCH_CHECK(gsIds.device() == gs.device(), "Inconsistent device.");
    
    auto device = gs.device();
    auto options = at::TensorOptions().dtype(at::kFloat).layout(at::kStrided).device(device);

    int64_t tileNum = tileCoords.sizes()[0];
    int64_t nPixel = tileCoords.sizes()[2];

    at::Tensor color = at::zeros({3, tileNum, nPixel}, options);
    at::Tensor depth = at::zeros({1, tileNum, nPixel}, options);
    at::Tensor lastCumsum = at::zeros({tileNum, nPixel}, options);
    at::Tensor error = at::zeros({tileNum, nPixel}, options);
    
    options = at::TensorOptions().dtype(torch::kInt64).layout(at::kStrided).device(device);
    at::Tensor gsClipIndex = at::zeros({tileNum}, options);
    options = at::TensorOptions().dtype(torch::kUInt8).layout(at::kStrided).device(device);
    at::Tensor alphaClipIndex = at::zeros({gsIds.sizes()[0], 2}, options);

    EXEC_NPU_CMD(aclnnCalcRenderFwdDoubleClipGsids,
        gs, tileCoords, offsets, gsIds,
        color, depth, lastCumsum, error,
        gsClipIndex, alphaClipIndex);
    return std::tie(color, depth, lastCumsum, error, gsClipIndex, alphaClipIndex);
}
