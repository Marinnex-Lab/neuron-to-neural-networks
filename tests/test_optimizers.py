"""Tests for src/optimizers.py on f(w) = w^2 (gradient 2w). Values from docs and Module 8."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optimizers import adam_step, momentum_step, rmsprop_step, sgd_step  # noqa: E402


def grad(w: np.ndarray) -> np.ndarray:
    return 2.0 * w


class TestOptimizers(unittest.TestCase):
    def test_sgd_sequence(self) -> None:
        w = np.array([1.0])
        seq = []
        for _ in range(3):
            w = sgd_step(w, grad(w), lr=0.1)
            seq.append(round(float(w[0]), 4))
        self.assertEqual(seq, [0.8, 0.64, 0.512])

    def test_sgd_lr_half_hits_minimum_in_one_step(self) -> None:
        w = sgd_step(np.array([1.0]), grad(np.array([1.0])), lr=0.5)
        self.assertAlmostEqual(float(w[0]), 0.0)

    def test_sgd_lr_above_one_diverges(self) -> None:
        w = np.array([1.0])
        for _ in range(3):
            w = sgd_step(w, grad(w), lr=1.1)
        self.assertAlmostEqual(float(w[0]), -1.728, places=3)

    def test_momentum_first_steps(self) -> None:
        w, v = np.array([1.0]), np.array([0.0])
        w, v = momentum_step(w, grad(w), v, lr=0.1, beta=0.9)
        self.assertAlmostEqual(float(w[0]), 0.8)
        w, v = momentum_step(w, grad(w), v, lr=0.1, beta=0.9)
        self.assertAlmostEqual(float(v[0]), 3.4, places=6)
        self.assertAlmostEqual(float(w[0]), 0.46, places=6)

    def test_rmsprop_first_step_by_hand(self) -> None:
        w, s = np.array([1.0]), np.array([0.0])
        w, s = rmsprop_step(w, grad(w), s, lr=0.1, rho=0.9)
        self.assertAlmostEqual(float(s[0]), 0.4, places=6)
        self.assertAlmostEqual(float(w[0]), 1.0 - 0.1 * 2.0 / np.sqrt(0.4), places=6)

    def test_adam_first_step_moves_by_lr(self) -> None:
        # With a steady gradient, the bias-corrected step is lr * g / |g| = lr.
        w, m, v = np.array([1.0]), np.array([0.0]), np.array([0.0])
        w, m, v = adam_step(w, grad(w), m, v, t=1, lr=0.1)
        self.assertAlmostEqual(float(w[0]), 0.9, places=6)
        self.assertAlmostEqual(float(m[0]), 0.2, places=6)
        self.assertAlmostEqual(float(v[0]), 0.004, places=8)

    def test_adam_second_step_values(self) -> None:
        w, m, v = np.array([1.0]), np.array([0.0]), np.array([0.0])
        w, m, v = adam_step(w, grad(w), m, v, t=1, lr=0.1)
        w, m, v = adam_step(w, grad(w), m, v, t=2, lr=0.1)
        self.assertAlmostEqual(float(w[0]), 0.8004, places=4)

    def test_adam_rejects_t_zero(self) -> None:
        with self.assertRaises(ValueError):
            adam_step(np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), t=0, lr=0.1)


if __name__ == "__main__":
    unittest.main()
