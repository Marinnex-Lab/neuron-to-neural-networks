"""Two-layer network n_in -> n_hidden -> 1 with sigmoid units and BCE loss.

Single-example forward and backward pass written out by hand (docs: Modules 5-7).

Forward:
    z1 = x W1 + b1          h = sigmoid(z1)
    z2 = h W2 + b2          y_hat = sigmoid(z2)
    L = BCE(y, y_hat)

Backward (BCE + sigmoid gives dL/dz2 = y_hat - y):
    dz2 = y_hat - y
    dW2 = outer(h, dz2)                 db2 = dz2
    dh  = W2[:, 0] * dz2
    dz1 = dh * h * (1 - h)              # sigmoid derivative h(1 - h)
    dW1 = outer(x, dz1)                 db1 = dz1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from activations import sigmoid
from losses import bce

Array = NDArray[np.float64]


@dataclass
class TwoLayerNet:
    W1: Array  # shape (n_in, n_hidden)
    b1: Array  # shape (n_hidden,)
    W2: Array  # shape (n_hidden, 1)
    b2: Array  # shape (1,)

    def forward(self, x: Array) -> dict[str, Array]:
        z1 = x @ self.W1 + self.b1
        h = sigmoid(z1)
        z2 = h @ self.W2 + self.b2
        y_hat = sigmoid(z2)
        return {"z1": z1, "h": h, "z2": z2, "y_hat": y_hat}

    def loss(self, x: Array, y: float) -> float:
        y_hat = self.forward(x)["y_hat"][0]
        return float(bce(np.array(y), np.array(y_hat)))

    def backward(self, x: Array, y: float) -> tuple[dict[str, Array], float]:
        """Return gradients {'W1','b1','W2','b2'} and the loss for one example."""
        cache = self.forward(x)
        h, y_hat = cache["h"], cache["y_hat"]
        loss = float(bce(np.array(y), np.array(y_hat[0])))

        dz2 = y_hat - y                      # shape (1,)
        dW2 = np.outer(h, dz2)               # shape (n_hidden, 1)
        db2 = dz2                            # shape (1,)
        dh = self.W2[:, 0] * dz2             # shape (n_hidden,)
        dz1 = dh * h * (1.0 - h)             # shape (n_hidden,)
        dW1 = np.outer(x, dz1)               # shape (n_in, n_hidden)
        db1 = dz1                            # shape (n_hidden,)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return grads, loss

    def params(self) -> dict[str, Array]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def batch_gradients_loop(self, X: Array, y: Array) -> tuple[dict[str, Array], float]:
        """Reference version: one sample at a time, then average. Used to test the matrix form."""
        grads_sum = {k: np.zeros_like(v) for k, v in self.params().items()}
        loss_sum = 0.0
        for x_i, y_i in zip(X, y):
            g, loss_i = self.backward(x_i, float(y_i))
            for k in grads_sum:
                grads_sum[k] += g[k]
            loss_sum += loss_i
        n = len(X)
        return {k: v / n for k, v in grads_sum.items()}, loss_sum / n

    def batch_gradients(self, X: Array, y: Array) -> tuple[dict[str, Array], float]:
        """Matrix form of the backward pass for a whole batch (rows of X, labels y of shape (N,)).

        dZ2 = (Y_hat - y) / N,   dW2 = H^T dZ2,   db2 = sum(dZ2)
        dH  = dZ2 W2^T,          dZ1 = dH * H (1 - H)
        dW1 = X^T dZ1,           db1 = sum(dZ1)
        """
        n = len(X)
        z1 = X @ self.W1 + self.b1
        H = sigmoid(z1)
        Y_hat = sigmoid(H @ self.W2 + self.b2)              # shape (N, 1)
        y_col = y.reshape(-1, 1).astype(np.float64)
        loss = float(np.mean(bce(y_col, Y_hat)))
        dZ2 = (Y_hat - y_col) / n
        dW2 = H.T @ dZ2
        db2 = dZ2.sum(axis=0)
        dH = dZ2 @ self.W2.T
        dZ1 = dH * H * (1.0 - H)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, loss

    def batch_loss(self, X: Array, y: Array) -> float:
        Y_hat = sigmoid(sigmoid(X @ self.W1 + self.b1) @ self.W2 + self.b2)
        return float(np.mean(bce(y.reshape(-1, 1).astype(np.float64), Y_hat)))

    def fit(self, X: Array, y: Array, lr: float, steps: int) -> list[float]:
        """Full-batch gradient descent. Returns the loss before each step, then the final loss."""
        history = [self.batch_loss(X, y)]
        for _ in range(steps):
            grads, _ = self.batch_gradients(X, y)
            for name, arr in self.params().items():
                arr -= lr * grads[name]
            history.append(self.batch_loss(X, y))
        return history

    def predict(self, X: Array) -> Array:
        """Predicted labels in {0, 1} (threshold 0.5 on the output probability)."""
        Y_hat = sigmoid(sigmoid(X @ self.W1 + self.b1) @ self.W2 + self.b2)
        return (Y_hat[:, 0] >= 0.5).astype(np.int64)

    def accuracy(self, X: Array, y: Array) -> float:
        return float(np.mean(self.predict(X) == y))


def init_network(n_in: int, n_hidden: int, seed: int, scale: float = 1.0) -> TwoLayerNet:
    """Random initialization: uniform in [-scale/sqrt(n_in), scale/sqrt(n_in)]. Seeded for reproducibility."""
    rng = np.random.default_rng(seed)
    limit = scale / np.sqrt(n_in)
    return TwoLayerNet(
        W1=rng.uniform(-limit, limit, size=(n_in, n_hidden)),
        b1=np.zeros(n_hidden),
        W2=rng.uniform(-1.0 / np.sqrt(n_hidden), 1.0 / np.sqrt(n_hidden), size=(n_hidden, 1)),
        b2=np.zeros(1),
    )
