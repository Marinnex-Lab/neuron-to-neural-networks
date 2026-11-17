"""Loss functions (docs: Module 6).

- mse: mean squared error for regression.
- bce: binary cross-entropy for a probability p = sigmoid(z).
- log_softmax / cce_from_logits: categorical cross-entropy from logits, stable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def mse(y: Array, y_hat: Array) -> float:
    return float(np.mean((y_hat - y) ** 2))


def bce(y: Array, p: Array, eps: float = 1e-12) -> Array:
    """Element-wise binary cross-entropy. p is clipped to avoid log(0)."""
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def log_softmax(z: Array) -> Array:
    """log softmax(z) = z - log sum exp(z), computed with the max shift."""
    m = np.max(z)
    return z - (m + np.log(np.sum(np.exp(z - m))))


def cce_from_logits(z: Array, t: int) -> float:
    """Categorical cross-entropy for one example, logits z, true class index t."""
    return float(-log_softmax(z)[t])
