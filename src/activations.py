"""Activation functions and their derivatives (docs: Module 4).

Conventions:
- step(z) = 1 where z >= 0, else 0. Its derivative is 0 almost everywhere.
- ReLU'(0) is set to 0 by convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def step(z: Array) -> NDArray[np.int64]:
    return (z >= 0).astype(np.int64)


def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-z))


def dsigmoid(z: Array) -> Array:
    """sigma'(z) = sigma(z) * (1 - sigma(z)); maximum 0.25 at z = 0."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: Array) -> Array:
    return np.tanh(z)


def dtanh(z: Array) -> Array:
    """tanh'(z) = 1 - tanh(z)^2; maximum 1 at z = 0."""
    return 1.0 - np.tanh(z) ** 2


def relu(z: Array) -> Array:
    return np.maximum(0.0, z)


def drelu(z: Array) -> Array:
    return (z > 0).astype(np.float64)


def leaky_relu(z: Array, alpha: float = 0.01) -> Array:
    return np.where(z >= 0, z, alpha * z)


def dleaky_relu(z: Array, alpha: float = 0.01) -> Array:
    return np.where(z >= 0, 1.0, alpha)


def softmax(z: Array, axis: int = -1) -> Array:
    """Numerically stable softmax: subtract the maximum before exponentiating."""
    shifted = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / np.sum(e, axis=axis, keepdims=True)


def naive_softmax(z: Array) -> Array:
    """Unstable version, kept only to demonstrate overflow in the notebook."""
    e = np.exp(z)
    return e / np.sum(e)
