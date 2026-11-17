"""Attention, feed-forward network, residual connections and layer norm (docs: Module 9).

Shapes: tokens are rows. X has shape (T, d_model). Q, K have shape (T, d_k); V has shape (T, d_v).

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

Causal masking sets scores for future positions to -inf before the softmax.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from activations import softmax

Array = NDArray[np.float64]


def attention(Q: Array, K: Array, V: Array, causal: bool = False) -> tuple[Array, Array]:
    """Scaled dot-product attention. Returns (output (T, d_v), weights (T, T))."""
    d_k = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    if causal:
        T = S.shape[0]
        future = np.triu(np.ones((T, T), dtype=bool), k=1)
        S = np.where(future, -np.inf, S)
    A = softmax(S, axis=-1)
    return A @ V, A


def layer_norm(z: Array, eps: float = 1e-5) -> Array:
    """Normalize each row (each token) over its features."""
    mu = z.mean(axis=-1, keepdims=True)
    var = z.var(axis=-1, keepdims=True)
    return (z - mu) / np.sqrt(var + eps)


def ffn(x: Array, W1: Array, W2: Array) -> Array:
    """Position-wise feed-forward network with a ReLU hidden layer (no biases, as in the notes)."""
    return np.maximum(0.0, x @ W1) @ W2


def residual_ffn_block(X: Array, W1: Array, W2: Array) -> Array:
    """Pre-norm residual block: Y = X + FFN(LayerNorm(X))."""
    return X + ffn(layer_norm(X), W1, W2)


def split_heads(X: Array, n_heads: int) -> Array:
    """Split the feature axis into heads: (T, d_model) -> (n_heads, T, d_model // n_heads)."""
    T, d_model = X.shape
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    return X.reshape(T, n_heads, d_model // n_heads).transpose(1, 0, 2)
