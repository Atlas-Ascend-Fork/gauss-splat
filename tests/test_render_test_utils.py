# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import unittest
from collections import namedtuple

import torch

from render_test_utils import (
    assert_render_results, cpu_inputs, make_covariance, render_with_gsids,
    transmittance_mask,
)


GSAttributes = namedtuple('GSAttributes', ['mean_ndc', 'conics_0', 'conics_1', 'conics_2',
                                           'colors', 'opacities', 'depths'])


def attributes(n, opacity=0.95):
    return GSAttributes(
        torch.full((n, 2), 16.0, dtype=torch.float64),
        torch.full((n,), 1e-3, dtype=torch.float64),
        torch.zeros(n, dtype=torch.float64),
        torch.full((n,), 1e-3, dtype=torch.float64),
        torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float64).repeat(n, 1),
        torch.full((n, 1), opacity, dtype=torch.float64),
        torch.full((n,), 2.0, dtype=torch.float64),
    )


def render(attrs):
    n = attrs[0].shape[0]
    return render_with_gsids((32, 32), attrs, torch.tensor([n]), torch.arange(n), 32)


class TestRenderReference(unittest.TestCase):
    def test_covariance_is_positive_definite_and_reproducible(self):
        torch.manual_seed(1234)
        cov = make_covariance(1000)
        torch.manual_seed(1234)
        self.assertTrue(torch.equal(cov, make_covariance(1000)))
        self.assertTrue(torch.equal(cov, cov.transpose(-1, -2)))
        self.assertTrue(bool((torch.linalg.eigvalsh(cov) > 0).all()))

    def test_cpu_inputs_are_independent_leaves(self):
        inputs_type = namedtuple('Inputs', ['value', 'ids', 'width'])
        source = torch.ones(2, requires_grad=True) * 2
        inputs = inputs_type(source, torch.tensor([1, 2]), 32)
        copied = cpu_inputs(inputs)
        self.assertIsInstance(copied, inputs_type)
        self.assertEqual(copied.value.device.type, 'cpu')
        self.assertTrue(copied.value.is_leaf and copied.value.requires_grad)
        self.assertFalse(copied.ids.requires_grad)
        self.assertEqual(copied.ids.dtype, torch.int64)
        with torch.no_grad():
            copied.value[0] = 7
            copied.ids[0] = 9
        self.assertEqual(source[0].item(), 2)
        self.assertEqual(inputs.ids[0].item(), 1)
        self.assertEqual(copied.width, 32)

    def test_transmittance_tail_and_group_boundaries(self):
        for n in range(10):
            with self.subTest(n=n):
                transparency = torch.full((n, 1024, 1), 0.001)
                expected = torch.ones_like(transparency, dtype=torch.bool)
                if n >= 8:
                    expected[4:] = False
                self.assertTrue(torch.equal(transmittance_mask(transparency), expected))

    def test_transmittance_threshold_is_strict_and_tile_wide(self):
        transparency = torch.full((12, 1024, 1), 0.001)
        transparency[4, 7] = 0.01
        mask = transmittance_mask(transparency)
        self.assertTrue(bool(mask[:8].all()))
        self.assertFalse(bool(mask[8:].any()))

    def test_render_tail_contribution_matches_compositing(self):
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
        alpha = 0.95 * torch.exp(-0.0005 * ((xx.double() - 16)**2 + (yy.double() - 16)**2))
        for n in range(1, 10):
            with self.subTest(n=n):
                color, depth = render(attributes(n))
                contributing = n if n < 8 else 4
                weight = (1 - (1 - alpha)**contributing).reshape(1, 1, 1024)
                expected_color = weight * torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64).reshape(3, 1, 1)
                torch.testing.assert_close(color, expected_color, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(depth, weight * 2, rtol=1e-12, atol=1e-12)

    def test_alpha_threshold_and_saturation(self):
        for opacity in (0.009, 0.01, 1.0):
            with self.subTest(opacity=opacity):
                color, depth = render(attributes(1, opacity))
                center = 16 * 32 + 16
                expected = 0.0 if opacity < 0.01 else min(opacity, 0.999)
                self.assertAlmostEqual(depth[0, 0, center].item(), expected * 2)
                self.assertAlmostEqual(color[0, 0, center].item(), expected * 0.2)

    def test_empty_tile_has_zero_outputs_and_gradients(self):
        attrs = tuple(value.requires_grad_() for value in attributes(1))
        color, depth = render_with_gsids(
            (32, 32), attrs, torch.tensor([0]), torch.empty(0, dtype=torch.int64), 32)
        self.assertEqual(tuple(color.shape), (3, 1, 1024))
        self.assertEqual(tuple(depth.shape), (1, 1, 1024))
        self.assertEqual(color.count_nonzero().item(), 0)
        self.assertEqual(depth.count_nonzero().item(), 0)
        for grad in torch.autograd.grad(color.sum() + depth.sum(), attrs):
            self.assertEqual(grad.count_nonzero().item(), 0)

    def test_reference_gradients_away_from_thresholds(self):
        attrs = tuple(value.requires_grad_() for value in attributes(1, 0.5))
        self.assertTrue(torch.autograd.gradcheck(
            lambda *values: tuple(value.sum() for value in render(values)), attrs))

    def test_unsupported_tile_size_is_explicit(self):
        with self.assertRaisesRegex(ValueError, 'tile_size=32'):
            render_with_gsids((64, 64), attributes(1), torch.tensor([1]), torch.tensor([0]), 64)


class TestRenderDiagnostics(unittest.TestCase):
    def compare(self, expected, actual):
        class Comparator(unittest.TestCase):
            calls = 0

            def assert_rtol_equal(self, first, second, prec=1e-5):
                self.calls += 1
                torch.testing.assert_close(torch.from_numpy(first), torch.from_numpy(second),
                                           rtol=prec, atol=prec)

            # assert_render_results calls the torch_npu comparator name.
            assertRtolEqual = assert_rtol_equal

            def run_test(self):
                assert_render_results(self, expected, actual, [1, 32, 32, 32])

        case = Comparator('run_test')
        result = unittest.TestResult()
        case.run(result)
        self.assertEqual(result.errors, [])
        return result, case.calls

    def test_diagnostics_preserve_comparator_defaults(self):
        outputs = namedtuple('Outputs', ['render_colors'])
        expected = outputs(torch.ones(2, 3))
        result, calls = self.compare(expected, outputs(torch.ones(2, 3) + 1e-6))
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(calls, 1)
        actual = torch.ones(2, 3)
        actual[1, 2] = 2
        result, calls = self.compare(expected, outputs(actual))
        self.assertEqual(len(result.failures), 1)
        message = result.failures[0][1]
        for token in ('render_colors', 'seed=1234', 'prec=1e-05', 'max_abs_error=1',
                      'max_rel_error=1', 'worst_abs_index=(1, 2)', 'expected=1', 'actual=2'):
            self.assertIn(token, message)

    def test_all_gradient_fields_are_checked_after_failures(self):
        outputs = namedtuple('Outputs', ['render_colors', 'render_depths', 'v_mean_ndc',
                                        'v_conics_0', 'v_conics_1', 'v_conics_2',
                                        'v_opacities', 'v_colors', 'v_depths'])
        result, calls = self.compare(outputs(*(torch.ones(1) for _ in outputs._fields)),
                                     outputs(*(torch.zeros(1) for _ in outputs._fields)))
        self.assertEqual(calls, 9)
        self.assertEqual(len(result.failures), 9)
        for name, (_, message) in zip(outputs._fields, result.failures):
            self.assertIn(name, message)

    def test_nonfinite_values_fail_before_comparison(self):
        outputs = namedtuple('Outputs', ['render_colors'])
        for value in (float('nan'), float('inf'), -float('inf')):
            for expected, actual in ((torch.tensor([value]), torch.tensor([value])),
                                     (torch.ones(1), torch.tensor([value]))):
                with self.subTest(value=value):
                    result, calls = self.compare(outputs(expected), outputs(actual))
                    self.assertEqual(calls, 0)
                    self.assertEqual(len(result.failures), 1)
                    self.assertIn('nonfinite_actual=1', result.failures[0][1])

    def test_shape_mismatch_fails_before_comparison(self):
        outputs = namedtuple('Outputs', ['render_colors'])
        result, calls = self.compare(outputs(torch.ones(2)), outputs(torch.ones(3)))
        self.assertEqual(calls, 0)
        self.assertEqual(len(result.failures), 1)
        self.assertIn('shape=', result.failures[0][1])

    def test_zero_reference_scalar_and_empty_tensors(self):
        outputs = namedtuple('Outputs', ['render_colors'])
        result, calls = self.compare(outputs(torch.tensor(0.0)), outputs(torch.tensor(1.0)))
        self.assertEqual(calls, 1)
        self.assertIn('max_rel_error=inf', result.failures[0][1])
        self.assertIn('worst_abs_index=()', result.failures[0][1])
        result, calls = self.compare(outputs(torch.empty(0)), outputs(torch.empty(0)))
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(calls, 0)


if __name__ == '__main__':
    unittest.main()
