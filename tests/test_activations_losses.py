"""Tests for src/activations.py and src/losses.py. Values are from docs/learning_roadmap.md and the lab notes."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activations import dsigmoid, dtanh, leaky_relu, relu, sigmoid, softmax, naive_softmax, tanh  # noqa: E402
from losses import bce, cce_from_logits, log_softmax, mse  # noqa: E402


class TestActivations(unittest.TestCase):
    def test_sigmoid_at_zero_and_derivative_maximum(self) -> None:
        self.assertAlmostEqual(float(sigmoid(np.array(0.0))), 0.5)
        self.assertAlmostEqual(float(dsigmoid(np.array(0.0))), 0.25)

    def test_sigmoid_derivative_never_exceeds_quarter(self) -> None:
        z = np.linspace(-10, 10, 2001)
        self.assertLessEqual(float(dsigmoid(z).max()), 0.25 + 1e-12)

    def test_tanh_values(self) -> None:
        self.assertAlmostEqual(float(tanh(np.array(1.0))), 0.7615941559557649)
        self.assertAlmostEqual(float(dtanh(np.array(1.0))), 0.41997434161402614)

    def test_relu_and_leaky(self) -> None:
        np.testing.assert_array_equal(relu(np.array([-2.0, 3.0])), [0.0, 3.0])
        self.assertAlmostEqual(float(leaky_relu(np.array(-2.0))), -0.02)

    def test_softmax_matches_notes(self) -> None:
        np.testing.assert_allclose(softmax(np.array([2.0, 1.0, 0.1])), [0.6590, 0.2424, 0.0986], atol=5e-4)

    def test_softmax_sums_to_one_and_is_shift_invariant(self) -> None:
        z = np.array([2.0, 1.0, 0.1])
        self.assertAlmostEqual(float(softmax(z).sum()), 1.0)
        np.testing.assert_allclose(softmax(z), softmax(z + 100.0))

    def test_naive_softmax_overflows_stable_does_not(self) -> None:
        z = np.array([1002.0, 1001.0, 1000.0])
        with np.errstate(over="ignore", invalid="ignore"):
            self.assertTrue(np.any(~np.isfinite(naive_softmax(z))))
        np.testing.assert_allclose(softmax(z), [0.6652, 0.2447, 0.0900], atol=5e-4)


class TestLosses(unittest.TestCase):
    def test_mse_value(self) -> None:
        self.assertAlmostEqual(mse(np.array([1.0, 0.0]), np.array([0.8, 0.3])), 0.065)

    def test_bce_values(self) -> None:
        self.assertAlmostEqual(float(bce(np.array(1.0), np.array(0.8))), 0.2231435513, places=6)
        self.assertAlmostEqual(float(bce(np.array(1.0), np.array(0.1))), 2.302585093, places=6)

    def test_bce_gradient_is_p_minus_y(self) -> None:
        # d/dz BCE(sigmoid(z)) = sigmoid(z) - y, checked by central differences.
        z, y, eps = 0.3, 1.0, 1e-6
        f = lambda zz: float(bce(np.array(y), sigmoid(np.array(zz))))
        numeric = (f(z + eps) - f(z - eps)) / (2 * eps)
        self.assertAlmostEqual(numeric, float(sigmoid(np.array(z))) - y, places=6)

    def test_cce_from_logits(self) -> None:
        self.assertAlmostEqual(cce_from_logits(np.array([2.0, 1.0, 0.1]), 0), 0.4170, places=4)
        self.assertAlmostEqual(cce_from_logits(np.array([1002.0, 1001.0, 1000.0]), 0), 0.4076, places=4)

    def test_log_softmax_is_finite_for_large_logits(self) -> None:
        self.assertTrue(np.all(np.isfinite(log_softmax(np.array([1e4, 0.0, -1e4])))))


if __name__ == "__main__":
    unittest.main()
