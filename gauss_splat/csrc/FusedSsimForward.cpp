/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <string>
 #include "OpApiCommon.h"
 #include "functions.h"
 
 using namespace NPU_NAME_SPACE;
 using namespace std;
 
 namespace {
 static const int64_t IMG_DIM = 4;
 static const int64_t WEIGHT_DIM = 1;
 static const int64_t WEIGHT_PADDED_LEN = 16;
 static const int64_t DIM_W = 3;
 } // namespace
 
 // Fused single-channel SSIM forward. img1/img2: [B,C,H,W] f32. weight: 1D
 // Gaussian window zero-padded to length 16. Returns 6 tensors of the same
 // [B,C,H,W] shape: (ssim_map, mu1, mu2, S11, S22, S12). The 5 trailing
 // "intermediate" tensors are the windowed sums needed by the analytical
 // backward (avoids a 5-conv recompute on every loss.backward()).
 std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
 fused_ssim_forward(at::Tensor &img1, at::Tensor &img2, at::Tensor &weight, int window_size)
 {
     TORCH_CHECK(img1.device().type() == at::kPrivateUse1, "Invalid device.");
     TORCH_CHECK(img2.device() == img1.device(), "Inconsistent device.");
     TORCH_CHECK(weight.device() == img1.device(), "Inconsistent device.");
 
     TORCH_CHECK(img1.dim() == IMG_DIM, "img1's dim should be 4 (B,C,H,W).");
     TORCH_CHECK(img2.dim() == IMG_DIM, "img2's dim should be 4 (B,C,H,W).");
     TORCH_CHECK(img1.sizes() == img2.sizes(), "img1 and img2 must have the same shape.");
     TORCH_CHECK(weight.dim() == WEIGHT_DIM, "weight's dim should be 1.");
     TORCH_CHECK(weight.sizes()[0] == WEIGHT_PADDED_LEN, "weight must be zero-padded to length 16.");
     TORCH_CHECK(window_size > 0 && window_size <= WEIGHT_PADDED_LEN && (window_size % 2 == 1),
                 "window_size must be odd and in (0, 16].");


     TORCH_CHECK(img1.scalar_type() == at::kFloat, "img1 must be float32");
     TORCH_CHECK(img2.scalar_type() == at::kFloat, "img2 must be float32");
     TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");

     at::Tensor img1c = img1.contiguous();
     at::Tensor img2c = img2.contiguous();
     at::Tensor weightc = weight.contiguous();
     auto opts = img1c.options().dtype(at::kFloat);
     at::Tensor ssimMap = at::empty(img1c.sizes(), opts).contiguous();
     at::Tensor mu1     = at::empty(img1c.sizes(), opts).contiguous();
     at::Tensor mu2     = at::empty(img1c.sizes(), opts).contiguous();
     at::Tensor s11     = at::empty(img1c.sizes(), opts).contiguous();
     at::Tensor s22     = at::empty(img1c.sizes(), opts).contiguous();
     at::Tensor s12     = at::empty(img1c.sizes(), opts).contiguous();
 
     EXEC_NPU_CMD(aclnnFusedSsimForward, img1c, img2c, weightc, window_size,
                  ssimMap, mu1, mu2, s11, s22, s12);
 
     return std::make_tuple(ssimMap, mu1, mu2, s11, s22, s12);
 }
 
