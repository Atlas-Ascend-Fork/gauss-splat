/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include "functions.h"
#include "OpApiCommon.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void fused_adam_multi_lr(std::vector<at::Tensor>& params, std::vector<at::Tensor>& grads,
           std::vector<at::Tensor>& exp_avgs, std::vector<at::Tensor>& exp_avg_sqs,
           std::vector<double>& lrs, double beta1, double beta2, double eps, double weight_decay, int64_t step)
{
    int64_t num = static_cast<int64_t>(params.size());
    constexpr int64_t kMaxTensorNum = 64; // must match MAX_TENSOR_NUM in op_host tiling
    TORCH_CHECK(num > 0, "params list must not be empty");
    TORCH_CHECK(num <= kMaxTensorNum, "params list size must not exceed ", kMaxTensorNum,
        " (kernel MAX_TENSOR_NUM); split the call into smaller batches");
    TORCH_CHECK(static_cast<int64_t>(grads.size()) == num, "grads list size must match params");
    TORCH_CHECK(static_cast<int64_t>(exp_avgs.size()) == num, "exp_avgs list size must match params");
    TORCH_CHECK(static_cast<int64_t>(exp_avg_sqs.size()) == num, "exp_avg_sqs list size must match params");
    TORCH_CHECK(static_cast<int64_t>(lrs.size()) == num, "lrs list size must match params");

    for (int64_t i = 0; i < num; i++) {
        TORCH_CHECK(torch_npu::utils::is_npu(params[i]), "param[", i, "] must be NPU tensor");
        TORCH_CHECK(params[i].scalar_type() == at::kFloat,
            "param[", i, "]: float32 tensor expected but got: ", params[i].scalar_type());
        TORCH_CHECK(params[i].sizes() == grads[i].sizes(), "param and grad must have same shape at index ", i);
        TORCH_CHECK(params[i].sizes() == exp_avgs[i].sizes(), "param and exp_avg must have same shape at index ", i);
        TORCH_CHECK(params[i].sizes() == exp_avg_sqs[i].sizes(), "param and exp_avg_sq must have same shape at index ", i);
        TORCH_CHECK(params[i].is_contiguous(), "param[", i, "] must be contiguous");
        TORCH_CHECK(grads[i].is_contiguous(), "grad[", i, "] must be contiguous");
        TORCH_CHECK(exp_avgs[i].is_contiguous(), "exp_avg[", i, "] must be contiguous");
        TORCH_CHECK(exp_avg_sqs[i].is_contiguous(), "exp_avg_sq[", i, "] must be contiguous");
    }

    // In-place: pass input tensors also as outputs, eliminating output allocation + copy-back.
    at::TensorList paramList(params);
    at::TensorList gradList(grads);
    at::TensorList expAvgList(exp_avgs);
    at::TensorList expAvgSqList(exp_avg_sqs);

    std::vector<float> lrVec(num);
    for (int64_t i = 0; i < num; i++) {
        lrVec[i] = static_cast<float>(lrs[i]);
    }
    at::ArrayRef<float> lrArray(lrVec);

    EXEC_NPU_CMD(aclnnFusedAdamMultiLr, paramList, gradList, expAvgList, expAvgSqList,
                 lrArray, beta1, beta2, eps, weight_decay, step, num,
                 paramList, expAvgList, expAvgSqList);
}