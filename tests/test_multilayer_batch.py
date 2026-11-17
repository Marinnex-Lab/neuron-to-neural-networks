"""Tests for batch training in src/mlp.py: matrix form vs loop form, and training on XOR."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlp import init_network  # noqa: E402

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y_XOR = np.array([0, 1, 1, 0], dtype=float)


class TestBatchGradients(unittest.TestCase):
    def test_matrix_form_equals_loop_form(self) -> None:
        net = init_network(2, 3, seed=7)
        g_mat, loss_mat = net.batch_gradients(X, Y_XOR)
        g_loop, loss_loop = net.batch_gradients_loop(X, Y_XOR)
        self.assertAlmostEqual(loss_mat, loss_loop, places=12)
        for k in g_mat:
            np.testing.assert_allclose(g_mat[k], g_loop[k], atol=1e-12, err_msg=k)

    def test_batch_gradients_match_finite_differences(self) -> None:
        net = init_network(2, 2, seed=3)
        grads, _ = net.batch_gradients(X, Y_XOR)
        eps = 1e-6
        for name, arr in net.params().items():
            numeric = np.zeros_like(arr)
            it = np.nditer(arr, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = arr[idx]
                arr[idx] = orig + eps
                lp = net.batch_loss(X, Y_XOR)
                arr[idx] = orig - eps
                lm = net.batch_loss(X, Y_XOR)
                arr[idx] = orig
                numeric[idx] = (lp - lm) / (2 * eps)
            np.testing.assert_allclose(grads[name], numeric, atol=1e-7, err_msg=name)


class TestXorTraining(unittest.TestCase):
    def test_loss_decreases_monotonically_for_small_lr(self) -> None:
        net = init_network(2, 3, seed=0)
        history = net.fit(X, Y_XOR, lr=0.05, steps=200)
        self.assertLess(history[-1], history[0])

    def test_three_hidden_units_solve_xor_for_seed_0(self) -> None:
        net = init_network(2, 3, seed=0)
        net.fit(X, Y_XOR, lr=0.5, steps=3000)
        self.assertEqual(net.accuracy(X, Y_XOR), 1.0)

    def test_predict_matches_per_sample_forward(self) -> None:
        net = init_network(2, 3, seed=1)
        per_sample = np.array([int(net.forward(x)["y_hat"][0] >= 0.5) for x in X])
        np.testing.assert_array_equal(net.predict(X), per_sample)


if __name__ == "__main__":
    unittest.main()
