"""Tests for src/perceptron.py. Run with: python -m unittest discover -s tests -v"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perceptron import Perceptron, step  # noqa: E402

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
Y_AND = np.array([0, 0, 0, 1])
Y_OR = np.array([0, 1, 1, 1])
Y_XOR = np.array([0, 1, 1, 0])


class TestStep(unittest.TestCase):
    def test_boundary_is_one(self) -> None:
        # Convention from the notes: output 1 iff z >= 0, so z = 0 gives 1.
        np.testing.assert_array_equal(step(np.array([-0.1, 0.0, 0.1])), [0, 1, 1])


class TestPerceptron(unittest.TestCase):
    def test_and_is_learned(self) -> None:
        model = Perceptron(n_features=2, learning_rate=0.1, epochs=20).fit(X, Y_AND)
        self.assertEqual(model.accuracy(X, Y_AND), 1.0)

    def test_or_is_learned(self) -> None:
        model = Perceptron(n_features=2, learning_rate=0.1, epochs=20).fit(X, Y_OR)
        self.assertEqual(model.accuracy(X, Y_OR), 1.0)

    def test_xor_is_not_learned(self) -> None:
        # XOR is not linearly separable (proof in docs/learning_roadmap.md, Module 3).
        model = Perceptron(n_features=2, learning_rate=0.1, epochs=20).fit(X, Y_XOR)
        self.assertLess(model.accuracy(X, Y_XOR), 1.0)
        self.assertGreater(min(model.errors_per_epoch[-5:]), 0)

    def test_and_error_sequence_matches_hand_trace(self) -> None:
        # Hand-traced in docs: epochs 1 and 2 give 2 and 3 mistakes.
        model = Perceptron(n_features=2, learning_rate=0.1, epochs=2).fit(X, Y_AND)
        self.assertEqual(model.errors_per_epoch, [2, 3])

    def test_learning_rate_scales_weights_not_decisions(self) -> None:
        # With zero initialization, a positive learning rate does not change the
        # predicted labels in exact arithmetic. Use learning rates that are exact
        # in binary (0.5 and 1.0) to avoid rounding at z = 0.
        small = Perceptron(n_features=2, learning_rate=0.5, epochs=20).fit(X, Y_AND)
        large = Perceptron(n_features=2, learning_rate=1.0, epochs=20).fit(X, Y_AND)
        self.assertEqual(small.errors_per_epoch, large.errors_per_epoch)
        np.testing.assert_array_equal(small.predict(X), large.predict(X))


if __name__ == "__main__":
    unittest.main()
