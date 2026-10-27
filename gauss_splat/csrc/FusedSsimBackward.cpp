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
 
 // Fused SSIM backward. Inputs: grad_ssim_map [B,C,H,W], img1/img2 [B,C,H,W],
 // weight [16] (zero-padded 1D Gaussian), and the 5 windowed-sum intermediates
 // saved by fused_ssim_forward (mu1, mu2, S11, S22, S12) -- all [B,C,H,W].
 // Returns (grad_img1, grad_img2). grad_img2 is only filled meaningfully when
 // need_grad_img2=true; otherwise the kernel leaves it as workspace-zero so the
 // Python wrapper can simply discard it.
 std::tuple<at::Tensor, at::Tensor>
 fused_ssim_backward(at::Tensor &grad_ssim_map, at::Tensor &img1, at::Tensor &img2, at::Tensor &weight,
                     at::Tensor &mu1, at::Tensor &mu2,
                     at::Tensor &s11, at::Tensor &s22, at::Tensor &s12,
                     int window_size, int need_grad_img2)
 {
     TORCH_CHECK(grad_ssim_map.device().type() == at::kPrivateUse1, "Invalid device.");
     TORCH_CHECK(img1.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(img2.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(weight.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(mu1.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(mu2.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(s11.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(s22.device() == grad_ssim_map.device(), "Inconsistent device.");
     TORCH_CHECK(s12.device() == grad_ssim_map.device(), "Inconsistent device.");
 
     TORCH_CHECK(grad_ssim_map.dim() == IMG_DIM, "grad_ssim_map's dim should be 4 (B,C,H,W).");
     TORCH_CHECK(img1.sizes() == grad_ssim_map.sizes(), "img1 must match grad_ssim_map shape.");
     TORCH_CHECK(img2.sizes() == grad_ssim_map.sizes(), "img2 must match grad_ssim_map shape.");
     TORCH_CHECK(mu1.sizes() == grad_ssim_map.sizes(), "mu1 must match grad_ssim_map shape.");
     TORCH_CHECK(mu2.sizes() == grad_ssim_map.sizes(), "mu2 must match grad_ssim_map shape.");
     TORCH_CHECK(s11.sizes() == grad_ssim_map.sizes(), "S11 must match grad_ssim_map shape.");
     TORCH_CHECK(s22.sizes() == grad_ssim_map.sizes(), "S22 must match grad_ssim_map shape.");
     TORCH_CHECK(s12.sizes() == grad_ssim_map.sizes(), "S12 must match grad_ssim_map shape.");
     TORCH_CHECK(weight.dim() == WEIGHT_DIM, "weight's dim should be 1.");
     TORCH_CHECK(weight.sizes()[0] == WEIGHT_PADDED_LEN, "weight must be zero-padded to length 16.");
     TORCH_CHECK(window_size > 0 && window_size <= WEIGHT_PADDED_LEN && (window_size % 2 == 1),
                 "window_size must be odd and in (0, 16].");

     TORCH_CHECK(grad_ssim_map.scalar_type() == at::kFloat, "grad_ssim_map must be float32");
     TORCH_CHECK(img1.scalar_type() == at::kFloat, "img1 must be float32");
     TORCH_CHECK(img2.scalar_type() == at::kFloat, "img2 must be float32");
     TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
     TORCH_CHECK(mu1.scalar_type() == at::kFloat, "mu1 must be float32");
     TORCH_CHECK(mu2.scalar_type() == at::kFloat, "mu2 must be float32");
     TORCH_CHECK(s11.scalar_type() == at::kFloat, "s11 must be float32");
     TORCH_CHECK(s22.scalar_type() == at::kFloat, "s22 must be float32");
     TORCH_CHECK(s12.scalar_type() == at::kFloat, "s12 must be float32");

     at::Tensor gradSsimC = grad_ssim_map.contiguous();
     at::Tensor img1c = img1.contiguous();
     at::Tensor img2c = img2.contiguous();
     at::Tensor weightc = weight.contiguous();
     at::Tensor mu1c = mu1.contiguous();
     at::Tensor mu2c = mu2.contiguous();
     at::Tensor s11c = s11.contiguous();
     at::Tensor s22c = s22.contiguous();
     at::Tensor s12c = s12.contiguous();
 
     auto opts = img1c.options().dtype(at::kFloat);
     // grad_img2 is always allocated; when need_grad_img2=false the kernel
     // dispatches to tiling key 1 and leaves grad_img2 as the workspace's
     // initial state (typically zero). Python wrapper discards it.
     at::Tensor gradImg1 = at::empty(img1c.sizes(), opts).contiguous();
     at::Tensor gradImg2 = at::empty(img1c.sizes(), opts).contiguous();
 
     EXEC_NPU_CMD(aclnnFusedSsimBackward,
                  gradSsimC, img1c, img2c, weightc, mu1c, mu2c, s11c, s22c, s12c,
                  window_size, need_grad_img2,
                  gradImg1, gradImg2);
 
     return std::make_tuple(gradImg1, gradImg2);
 }
