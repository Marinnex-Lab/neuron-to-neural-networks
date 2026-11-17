# Learning roadmap

This document lists the modules, what each one covers, and whether it is implemented in this repository.

Status key: **Implemented** (code runs and is tested), **Notes only** (written as explanation, no verified code here), **Planned** (not yet written).

| Module | Topic | Status |
|---|---|---|
| 1 | Artificial neuron | Notes only (source notes are in the lab folder) |
| 2 | Perceptron and learning rule | Implemented in `src/perceptron.py` |
| 3 | AND, OR, XOR | Implemented in `experiments/logic_gates.py`; proof below |
| 4 | Activation functions | Planned |
| 5 | Multilayer networks | Planned |
| 6 | Loss functions | Planned |
| 7 | Calculus and backpropagation | Planned |
| 8 | Optimization | Planned |
| 9 | Transformers and LLMs | Planned |

## Module 1: Artificial neuron

A neuron computes a weighted sum and applies an activation:

$$z = \mathbf{w}^\top \mathbf{x} + b, \qquad y = f(z)$$

- $\mathbf{x}$: inputs. $\mathbf{w}$: weights. $b$: bias. $z$: pre-activation. $f$: activation. $y$: output.

## Module 2: Perceptron

A perceptron is a neuron with a step activation, $\hat{y} = 1$ if $z \ge 0$ and $0$ otherwise, trained with the perceptron rule:

$$\text{error} = t - \hat{y}, \qquad \mathbf{w} \leftarrow \mathbf{w} + \eta \cdot \text{error} \cdot \mathbf{x}, \qquad b \leftarrow b + \eta \cdot \text{error}$$

- $t$: target. $\eta$: learning rate. One pass over the data is an epoch.
- With zero initialization, the AND run in this repository gives 2 and 3 mistakes in the first two epochs, then 0.

## Module 3: AND, OR, XOR

### Truth tables

| $x_1$ | $x_2$ | AND | OR | XOR |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

### Linear separability

A dataset is linearly separable if some $\mathbf{w}, b$ satisfy $\mathbf{w}^\top \mathbf{x} + b \ge 0$ for every positive point and $< 0$ for every negative point.

- AND is separable: $\mathbf{w} = (1, 1)$, $b = -1.5$.
- OR is separable: $\mathbf{w} = (1, 1)$, $b = -0.5$.

### Proof that XOR is not separable

Suppose $\mathbf{w} = (w_1, w_2)$ and $b$ separate XOR. The four conditions are:

1. $(0,0) \to 0$: $b < 0$
2. $(0,1) \to 1$: $w_2 + b \ge 0$
3. $(1,0) \to 1$: $w_1 + b \ge 0$
4. $(1,1) \to 0$: $w_1 + w_2 + b < 0$

Adding conditions 2 and 3 gives $w_1 + w_2 + 2b \ge 0$. Condition 4 gives $w_1 + w_2 < -b$, so $w_1 + w_2 + 2b < b$. Together, $0 \le w_1 + w_2 + 2b < b$, so $b > 0$. This contradicts condition 1. No separating line exists.

### What the experiment shows

- AND and OR reach 0 mistakes and stay there.
- XOR never reaches 0 mistakes. The perceptron keeps moving its boundary and cannot settle. This matches the proof.

## Module 4 to 9

Planned. Each module will state its mathematics, give a numerical example, and link to code only once the code has been run and tested.

## Caveats

- The experiments use one learning rate (0.1), one epoch count (20), and zero initialization. The results are specific to these settings.
- Floating-point rounding can change a decision when a point lies exactly on the boundary. The AND run reaches 0 mistakes at epoch 4 because of rounding at the point $(1,0)$. Exact binary learning rates give a different sequence.
