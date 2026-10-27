"""Direct fused_ssim_backward tests adapted to the gauss_splat native API."""
import math

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from gauss_splat import gsplat_c


def weight(ws, sigma=0.75):
    values = torch.tensor([math.exp(-(i - ws // 2) ** 2 / (2 * sigma * sigma))
                           for i in range(ws)], dtype=torch.float32)
    out = torch.zeros(16)
    out[:ws] = values / values.sum()
    return out


def reference(grad, img1, img2, mu1, mu2, s11, s22, s12, w, ws):
    c = img1.shape[1]
    c1, c2 = .01 ** 2, .03 ** 2
    m11, m22, m12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    v1, v2, cov = s11 - m11, s22 - m22, s12 - m12
    a, b = 2 * m12 + c1, 2 * cov + c2
    p, q = m11 + m22 + c1, v1 + v2 + c2
    ssim = a * b / (p * q)
    scale = 2 * grad / (p * q)
    ka = scale * (mu2 * (b - a) - ssim * mu1 * (q - p))
    ka2 = scale * (mu1 * (b - a) - ssim * mu2 * (q - p))
    kb, kg = scale * a, -scale * ssim * p
    ww = (w[:ws, None] * w[None, :ws]).expand(c, 1, ws, ws)
    conv = lambda x: F.conv2d(x, ww, padding=ws // 2, groups=c)
    return conv(ka) + img2 * conv(kb) + img1 * conv(kg), \
        conv(ka2) + img1 * conv(kb) + img2 * conv(kg)


@pytest.mark.parametrize("shape,ws", [((1, 3, 16, 16), 7),
                                       ((1, 3, 64, 70), 11),
                                       ((2, 1, 32, 33), 15)])
def test_fused_ssim_backward(shape, ws):
    torch.manual_seed(2026)
    x = torch.rand(shape).npu()
    y = torch.rand(shape).npu()
    g = torch.randn(shape).npu()
    w = weight(ws).npu()
    _, mu1, mu2, s11, s22, s12 = gsplat_c.fused_ssim_forward(x, y, w, ws)
    g1, _ = gsplat_c.fused_ssim_backward(g, x, y, w, mu1, mu2, s11, s22, s12, ws, 0)
    g1b, g2 = gsplat_c.fused_ssim_backward(g, x, y, w, mu1, mu2, s11, s22, s12, ws, 1)
    ref1, ref2 = reference(g.cpu(), x.cpu(), y.cpu(), mu1.cpu(), mu2.cpu(),
                           s11.cpu(), s22.cpu(), s12.cpu(), w.cpu(), ws)
    torch.testing.assert_close(g1.cpu(), ref1, atol=5e-4, rtol=5e-5)
    torch.testing.assert_close(g1b.cpu(), ref1, atol=5e-4, rtol=5e-5)
    torch.testing.assert_close(g2.cpu(), ref2, atol=5e-4, rtol=5e-5)

