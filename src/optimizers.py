"""Single-step update rules (docs: Module 8).

Each function takes the current parameters, the gradient, and optimizer state,
and returns the updated values. Nothing is stored globally.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def sgd_step(theta: Array, grad: Array, lr: float) -> Array:
    return theta - lr * grad


def momentum_step(theta: Array, grad: Array, v: Array, lr: float, beta: float = 0.9) -> tuple[Array, Array]:
    v_new = beta * v + grad
    return theta - lr * v_new, v_new


def rmsprop_step(
    theta: Array, grad: Array, s: Array, lr: float, rho: float = 0.9, eps: float = 1e-8
) -> tuple[Array, Array]:
    s_new = rho * s + (1.0 - rho) * grad ** 2
    return theta - lr * grad / (np.sqrt(s_new) + eps), s_new


def adam_step(
    theta: Array,
    grad: Array,
    m: Array,
    v: Array,
    t: int,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Array, Array, Array]:
    """t counts steps from 1. Bias correction uses 1 - beta^t."""
    if t < 1:
        raise ValueError("t must be >= 1")
    m_new = beta1 * m + (1.0 - beta1) * grad
    v_new = beta2 * v + (1.0 - beta2) * grad ** 2
    m_hat = m_new / (1.0 - beta1 ** t)
    v_hat = v_new / (1.0 - beta2 ** t)
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps), m_new, v_new
