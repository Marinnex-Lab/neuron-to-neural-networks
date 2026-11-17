"""Tests for src/transformer.py. Expected values are the hand-checked values in docs and Module 9."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformer import attention, ffn, layer_norm, residual_ffn_block, split_heads  # noqa: E402

X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=float)
WQ = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
WK = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
WV = np.array([[1, 1], [0, 1], [1, 0], [0, 0.5]], dtype=float)
W1 = np.array([[0.5, -0.2, 0.1, 0.3], [0.2, 0.4, -0.1, 0.0], [-0.3, 0.1, 0.2, 0.5], [0.1, 0.0, 0.3, -0.2]])
W2 = np.array([[0.2, 0.1, 0.0, 0.0], [0.0, 0.3, 0.0, 0.0], [0.4, -0.1, 0.2, 0.0], [0.1, 0.2, 0.0, 0.1]])


class TestAttention(unittest.TestCase):
    def test_projections_match_notes(self) -> None:
        np.testing.assert_array_equal(X @ WQ, [[2, 0], [0, 2], [1, 1]])
        np.testing.assert_array_equal(X @ WK, [[0, 2], [2, 0], [1, 1]])
        np.testing.assert_array_equal(X @ WV, [[2, 1], [0, 1.5], [1, 2]])

    def test_weights_and_output_match_notes(self) -> None:
        out, A = attention(X @ WQ, X @ WK, X @ WV)
        np.testing.assert_allclose(A[0], [0.0454, 0.7679, 0.1867], atol=5e-4)
        np.testing.assert_allclose(out[0], [0.2775, 1.5707], atol=5e-4)
        np.testing.assert_allclose(A[2], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
        self.assertEqual(out.shape, (3, 2))

    def test_rows_of_weights_sum_to_one(self) -> None:
        _, A = attention(X @ WQ, X @ WK, X @ WV)
        np.testing.assert_allclose(A.sum(axis=1), [1, 1, 1])

    def test_causal_first_token_sees_only_itself(self) -> None:
        out, A = attention(X @ WQ, X @ WK, X @ WV, causal=True)
        np.testing.assert_allclose(A[0], [1, 0, 0])
        np.testing.assert_allclose(out[0], [2, 1])
        np.testing.assert_allclose(out[1], [1.8884, 1.0279], atol=5e-4)
        self.assertTrue(np.all(np.triu(A, k=1) == 0))

    def test_scaling_keeps_scores_moderate(self) -> None:
        # Unscaled scores for d_k = 2 are 4 and 2; scaled by sqrt(2) they are 2.8284 and 1.4142.
        Q, K = X @ WQ, X @ WK
        S = Q @ K.T / np.sqrt(Q.shape[-1])
        self.assertAlmostEqual(S[0, 1], 4 / np.sqrt(2))


class TestBlocks(unittest.TestCase):
    def test_layer_norm_zero_mean_unit_variance(self) -> None:
        Z = layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]))
        self.assertAlmostEqual(float(Z.mean()), 0.0, places=8)
        self.assertAlmostEqual(float(Z.var()), 1.0, places=4)

    def test_layer_norm_values(self) -> None:
        np.testing.assert_allclose(layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]))[0],
                                   [-1.3416, -0.4472, 0.4472, 1.3416], atol=5e-4)

    def test_ffn_token_one_hand_check(self) -> None:
        x = np.array([1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(ffn(x[None, :], W1, W2)[0], [0.14, 0.19, 0.02, 0.10], atol=1e-12)

    def test_residual_block_matches_notes(self) -> None:
        # The notes round to 2 decimals. The layer-norm epsilon (1e-5) moves the exact
        # values by about 4e-6, so the tolerance here is 1e-4.
        Y = residual_ffn_block(X, W1, W2)
        np.testing.assert_allclose(Y[0], [1.14, 0.19, 1.02, 0.10], atol=1e-4)
        self.assertEqual(Y.shape, (3, 4))


class TestHeads(unittest.TestCase):
    def test_split_heads_shape(self) -> None:
        H = split_heads(np.zeros((5, 8)), n_heads=2)
        self.assertEqual(H.shape, (2, 5, 4))

    def test_split_heads_rejects_bad_width(self) -> None:
        with self.assertRaises(ValueError):
            split_heads(np.zeros((5, 6)), n_heads=4)


if __name__ == "__main__":
    unittest.main()
