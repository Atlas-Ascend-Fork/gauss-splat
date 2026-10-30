"""Real NPU forward/backward tests against an independent CPU SSIM formula.

Run with: python -m pytest -v tests/test_fused_ssim.py
"""
import pytest
import torch
import torch.nn.functional as F
import torch_npu

from gauss_splat import fused_ssim


def ssim_reference(x, y, window_size, sigma):
    offsets = torch.arange(window_size, dtype=x.dtype) - window_size // 2
    weights = torch.exp(-offsets.square() / (2 * sigma**2))
    weights = weights / weights.sum()
    kernel = (weights[:, None] * weights[None, :]).expand(x.shape[1], 1, -1, -1)

    def conv(value):
        return F.conv2d(value, kernel, padding=window_size // 2, groups=x.shape[1])

    mx, my = conv(x), conv(y)
    vx, vy = conv(x.square()) - mx.square(), conv(y.square()) - my.square()
    cov = conv(x * y) - mx * my
    return ((2 * mx * my + 0.01**2) * (2 * cov + 0.03**2)
            / ((mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2)))


@pytest.mark.parametrize('shape,window,sigma,reduction,need_y,transpose', [
    ((1, 3, 17, 31), 7, 0.75, 'mean', False, False),
    ((2, 3, 16, 32), 7, 0.75, 'none', True, False),
    ((1, 1, 8, 9), 7, 0.75, 'sum', True, False),
    ((1, 1, 16, 17), 15, 1.5, 'mean', True, False),
    ((1, 3, 64, 70), 11, 1.0, 'mean', True, False),
])
def test_fused_ssim_forward_backward(shape, window, sigma, reduction, need_y, transpose):
    torch.manual_seed(42)
    x, y = torch.rand(shape), torch.rand(shape)
    if transpose:
        x, y = x.transpose(-1, -2), y.transpose(-1, -2)
    x.requires_grad_()
    y.requires_grad_(need_y)
    xn = x.detach().npu().requires_grad_()
    yn = y.detach().npu().requires_grad_(need_y)
    expected = ssim_reference(x, y, window, sigma)
    expected = expected.mean((1, 2, 3)) if reduction == 'none' else getattr(expected, reduction)()
    actual = fused_ssim(xn, yn, window_size=window, sigma=sigma, reduction=reduction)
    # A nonuniform upstream gradient also tests the per-image reduction path.
    grad = torch.linspace(0.3, 1.1, expected.numel()).reshape(expected.shape)
    expected.backward(grad)
    actual.backward(grad.npu())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(xn.grad.cpu(), x.grad, atol=2e-4, rtol=3e-3)
    if need_y:
        torch.testing.assert_close(yn.grad.cpu(), y.grad, atol=2e-4, rtol=3e-3)
    else:
        assert yn.grad is None


def test_fused_ssim_identical_images():
    """The exact identity case must produce an SSIM map of one."""
    torch.manual_seed(7)
    image = torch.rand(1, 3, 64, 70)
    result = fused_ssim(image.npu(), image.npu(), window_size=7, sigma=0.75,
                        reduction="none")
    torch.testing.assert_close(result.cpu(), torch.ones(1), atol=0.0, rtol=0.0)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
