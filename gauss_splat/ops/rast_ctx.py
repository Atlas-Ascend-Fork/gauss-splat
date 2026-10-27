import torch
from gauss_splat import gsplat_c as _C

class RastCtx:
    def __init__(self):
        # features
        self.backend = '3dgs'
        self.if_depth_correct = False
        self.use_arch35 = getattr(_C, "use_arch35", False)
        self.proj_backward_culling = getattr(_C, "use_arch35", False)
        self.pose_refine = getattr(_C, "use_arch35", False)
        self.covars_fused = getattr(_C, "sf_covars_fused", False)

        # alpha max and grad abs
        self.alpha_max = None
        self.grad_abs = None
        self.last_N = 0
        
        # projection forward outputs
        self.proj_forward_outputs = dict(
            means2d = None,
            depths = None,
            conics = None,
            compensations = None,
            det = None,
            radius = None,
            covars2d = None
        )

        # filter outputs
        self.filter_outputs = dict(
            means_culling = None,
            colors_culling = None,
            means2d_culling = None,
            depths_culling = None,
            radius_culling = None,
            covars2d_culling = None,
            conics_culling = None,
            opacities_culling = None,
            quats_culling = None,
            scales_culling = None,
            vi_culling = None,
            compensations_culling = None,
            proj_filter = None,
            cnt_float = None,
            filter_float = None,
            dense_indices = None
        )

        # projection backward outputs
        self.proj_backward_outputs = dict(
            v_pw = None,
            v_quats = None,
            v_scales = None,
            v_r = None,
            v_colors = None,
            v_opacities = None,
            v_viewmats = None,
            v_ks = None,
            dL_dvi = None,
            dL_dsigma = None
        )

    def prealloc_if_need(self, B, C, N, device):
        """Pre-allocate all filter/culling tensors used by the forward
        (and required for the backward) of projection_three_dims_gaussian_fused."""
        config = (B, C, N, device, self.proj_backward_culling,
                  self.alpha_max is not None and self.grad_abs is not None)
        if getattr(self, "_allocation_config", None) != config:
            for outputs in (self.proj_forward_outputs, self.filter_outputs,
                            self.proj_backward_outputs):
                outputs.update({key: None for key in outputs})
            self._allocation_config = config

        # projection forward outputs
        outputs = self.proj_forward_outputs
        if outputs["means2d"] is None or self.last_N != N:
            outputs["means2d"] = torch.empty((B, C, 2, N), device=device, dtype=torch.float32)

        if outputs["depths"] is None or self.last_N != N:
            outputs["depths"] = torch.empty((B, C, 1, N), device=device, dtype=torch.float32)

        if outputs["conics"] is None or self.last_N != N:
            outputs["conics"] = torch.empty((B, C, 3, N), device=device, dtype=torch.float32)

        if outputs["compensations"] is None or self.last_N != N:
            outputs["compensations"] = torch.empty((B, C, 1, N), device=device, dtype=torch.float32)

        if outputs["det"] is None or self.last_N != N:
            outputs["det"] = torch.empty((B, C, 1, N), device=device, dtype=torch.float32)

        if outputs["radius"] is None or self.last_N != N:
            outputs["radius"] = torch.empty((B, C, 2, N), device=device, dtype=torch.float32)

        if outputs["covars2d"] is None or self.last_N != N:
            outputs["covars2d"] = torch.empty((B, C, 3, N), device=device, dtype=torch.float32)

        # filter outputs
        outputs = self.filter_outputs
        if outputs["means_culling"] is None or self.last_N != N:
            outputs["means_culling"] = torch.empty((B, C, 3, N), device=device, dtype=torch.float32)

        colors_culling_dim = 7 if self.proj_backward_culling else 3
        if outputs["colors_culling"] is None or self.last_N != N:
            outputs["colors_culling"] = torch.empty((B, C, colors_culling_dim, N), device=device, dtype=torch.float32)

        if outputs["means2d_culling"] is None or self.last_N != N:
            outputs["means2d_culling"] = torch.empty((B, C, 2, N), device=device, dtype=torch.float32)

        if outputs["depths_culling"] is None or self.last_N != N:
            outputs["depths_culling"] = torch.empty((B, C, N), device=device, dtype=torch.float32)

        if outputs["radius_culling"] is None or self.last_N != N:
            outputs["radius_culling"] = torch.empty((B, C, 2, N), device=device, dtype=torch.float32)

        if outputs["covars2d_culling"] is None or self.last_N != N:
            outputs["covars2d_culling"] = torch.empty((B, C, 3, N), device=device, dtype=torch.float32)

        if outputs["conics_culling"] is None or self.last_N != N:
            outputs["conics_culling"] = torch.empty((B, C, 3, N), device=device, dtype=torch.float32)

        if outputs["opacities_culling"] is None or self.last_N != N:
            outputs["opacities_culling"] = torch.empty((B, C, N), device=device, dtype=torch.float32)

        if outputs["quats_culling"] is None or self.last_N != N:
            outputs["quats_culling"] = torch.empty((B, C, 4, N), device=device, dtype=torch.float32)

        if outputs["scales_culling"] is None or self.last_N != N:
            outputs["scales_culling"] = torch.empty((B, C, 3, N), device=device, dtype=torch.float32)
    
        if outputs["vi_culling"] is None or self.last_N != N:
            outputs["vi_culling"] = torch.empty((B, C, N), device=device, dtype=torch.float32)

        if outputs["compensations_culling"] is None or self.last_N != N:
            outputs["compensations_culling"] = torch.empty((B, C, N), device=device, dtype=torch.float32)

        if outputs["proj_filter"] is None or self.last_N != N:
            outputs["proj_filter"] = torch.empty((B, C, (N + 7) // 8), device=device, dtype=torch.uint8)

        if outputs["cnt_float"] is None or self.last_N != N:
            outputs["cnt_float"] = torch.zeros((B, C), device=device, dtype=torch.float32)

        if outputs["filter_float"] is None or self.last_N != N:
            outputs["filter_float"] = torch.zeros((B, C, N), device=device, dtype=torch.float32)

        if outputs["dense_indices"] is None or self.last_N != N:
            outputs["dense_indices"] = torch.empty((B, C, N), device=device, dtype=torch.int32)

        # backward gradient tensors
        outputs = self.proj_backward_outputs
        if outputs["v_pw"] is None or self.last_N != N:
            outputs["v_pw"] = torch.zeros((B, 3, N), device=device, dtype=torch.float32) \
                if self.use_arch35 else torch.zeros((B, N, 3), device=device, dtype=torch.float32)

        if outputs["v_quats"] is None or self.last_N != N:
            outputs["v_quats"] = torch.zeros((B, 4, N), device=device, dtype=torch.float32) \
                if self.use_arch35 else torch.zeros((B, N, 4), device=device, dtype=torch.float32)

        if outputs["v_scales"] is None or self.last_N != N:
            outputs["v_scales"] = torch.zeros((B, 3, N), device=device, dtype=torch.float32) \
                if self.use_arch35 else torch.zeros((B, N, 3), device=device, dtype=torch.float32)

        if outputs["v_r"] is None or self.last_N != N:
            outputs["v_r"] = torch.zeros((B, C, 3, 3), device=device, dtype=torch.float32)

        ### more 3 dimension need to save alpha max and grad abs
        v_color_dim = 6 if self.alpha_max is not None and self.grad_abs is not None else 3
        if outputs["v_colors"] is None or self.last_N != N:
            outputs["v_colors"] = torch.zeros((B, v_color_dim, N), device=device, dtype=torch.float32)

        if outputs["v_opacities"] is None or self.last_N != N:
            outputs["v_opacities"] = torch.zeros((B, N), device=device, dtype=torch.float32)

        if outputs["v_viewmats"] is None:
            outputs["v_viewmats"] = torch.zeros((B, C, 16), device=device, dtype=torch.float32)

        if outputs["v_ks"] is None:
            outputs["v_ks"] = torch.zeros((B, C, 9), device=device, dtype=torch.float32)

        if outputs["dL_dvi"] is None or self.last_N != N:
            outputs["dL_dvi"] = torch.zeros((B * C, N), device=device, dtype=torch.float32)

        if outputs["dL_dsigma"] is None or self.last_N != N:
            outputs["dL_dsigma"] = torch.zeros((B * C, N), device=device, dtype=torch.float32)

        self.last_N = N

    def outputs_for_proj_forward(self):
        return list(self.proj_forward_outputs.values())

    def outputs_for_proj_backward(self):
        return list(self.proj_backward_outputs.values())

    def outputs_for_filter(self):
        return list(self.filter_outputs.values())
