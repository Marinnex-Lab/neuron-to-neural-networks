"""Single perceptron with a step activation and the perceptron learning rule.

Mathematics (see docs/learning_roadmap.md, Module 2):

    z    = w . x + b
    y_hat = 1 if z >= 0 else 0
    error = t - y_hat
    w   <- w + lr * error * x
    b   <- b + lr * error
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


def step(z: NDArray[np.float64]) -> NDArray[np.int64]:
    """Step activation: 1 where z >= 0, otherwise 0."""
    return (z >= 0).astype(np.int64)


@dataclass
class Perceptron:
    """A single neuron with a step activation, trained by the perceptron rule.

    Parameters
    ----------
    n_features:
        Number of inputs.
    learning_rate:
        Step size for each update (eta).
    epochs:
        Number of full passes over the training data.
    """

    n_features: int
    learning_rate: float = 0.1
    epochs: int = 20
    weights: NDArray[np.float64] = field(init=False)
    bias: float = field(init=False)
    errors_per_epoch: list[int] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        # Zero initialization: deterministic, so runs are reproducible without a seed.
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0

    def net_input(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Weighted sum z = X w + b for every row of X (shape (n_samples,))."""
        return X @ self.weights + self.bias

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predicted labels in {0, 1}."""
        return step(self.net_input(X))

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> "Perceptron":
        """Train with the perceptron rule. Records mistakes made in each epoch."""
        self.errors_per_epoch = []
        for _ in range(self.epochs):
            mistakes = 0
            for x_i, target in zip(X, y):
                y_hat = int(step(np.array([x_i @ self.weights + self.bias]))[0])
                error = int(target) - y_hat
                if error != 0:
                    mistakes += 1
                    self.weights = self.weights + self.learning_rate * error * x_i
                    self.bias = self.bias + self.learning_rate * error
            self.errors_per_epoch.append(mistakes)
        return self

    def accuracy(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        """Fraction of correctly predicted labels."""
        return float(np.mean(self.predict(X) == y))
