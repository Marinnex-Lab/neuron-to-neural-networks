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
