"""Tests for src/mlp.py: hand-computed forward and backward values, and gradient checks."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlp import TwoLayerNet  # noqa: E402


def example_net() -> tuple[TwoLayerNet, np.ndarray, float]:
    net = TwoLayerNet(
        W1=np.array([[0.1, 0.4], [0.3, -0.2]]),
        b1=np.array([0.0, 0.1]),
        W2=np.array([[0.5], [-0.3]]),
        b2=np.array([0.2]),
    )
    return net, np.array([1.0, 0.5]), 1.0


class TestForward(unittest.TestCase):
    def test_forward_values_match_hand_calculation(self) -> None:
        net, x, _ = example_net()
        c = net.forward(x)
        np.testing.assert_allclose(c["z1"], [0.25, 0.40], atol=1e-12)
        np.testing.assert_allclose(c["h"], [0.5622, 0.5987], atol=1e-4)
        np.testing.assert_allclose(c["z2"], [0.3015], atol=1e-4)
        np.testing.assert_allclose(c["y_hat"], [0.5748], atol=1e-4)

    def test_loss_value(self) -> None:
        net, x, y = example_net()
        self.assertAlmostEqual(net.loss(x, y), 0.5537, places=4)

    def test_shapes(self) -> None:
        net, x, _ = example_net()
        c = net.forward(x)
        self.assertEqual(c["h"].shape, (2,))
        self.assertEqual(c["y_hat"].shape, (1,))


class TestBackward(unittest.TestCase):
    def test_gradients_match_hand_values(self) -> None:
        net, x, y = example_net()
        g, _ = net.backward(x, y)
        np.testing.assert_allclose(g["W1"], [[-0.0523, 0.0306], [-0.0262, 0.0153]], atol=1e-4)
        np.testing.assert_allclose(g["b1"], [-0.0523, 0.0306], atol=1e-4)
        np.testing.assert_allclose(g["W2"], [[-0.2391], [-0.2546]], atol=1e-4)
        np.testing.assert_allclose(g["b2"], [-0.4252], atol=1e-4)

    def test_gradients_match_finite_differences(self) -> None:
        net, x, y = example_net()
        g, _ = net.backward(x, y)
        eps = 1e-6
        for name, arr in net.params().items():
            numeric = np.zeros_like(arr)
            it = np.nditer(arr, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = arr[idx]
                arr[idx] = orig + eps
                lp = net.loss(x, y)
                arr[idx] = orig - eps
                lm = net.loss(x, y)
                arr[idx] = orig
                numeric[idx] = (lp - lm) / (2 * eps)
            np.testing.assert_allclose(g[name], numeric, atol=1e-7, err_msg=name)

    def test_one_gradient_step_lowers_loss(self) -> None:
        net, x, y = example_net()
        g, before = net.backward(x, y)
        lr = 0.5
        after_net = TwoLayerNet(
            W1=net.W1 - lr * g["W1"],
            b1=net.b1 - lr * g["b1"],
            W2=net.W2 - lr * g["W2"],
            b2=net.b2 - lr * g["b2"],
        )
        self.assertAlmostEqual(before, 0.5537, places=4)
        self.assertLess(after_net.loss(x, y), before)
        self.assertAlmostEqual(after_net.loss(x, y), 0.414, places=3)


if __name__ == "__main__":
    unittest.main()
