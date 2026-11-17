# Project overview

## Purpose

This repository explains how neural networks work from first principles, starting from one artificial neuron. Each concept is shown in three forms: the mathematics, a numerical example worked by hand, and code that reproduces the example. Results come from scripts in this repository, not from typed-in numbers.

## Audience

- Computer science students and beginners in machine learning.
- Developers who want to understand what a framework computes.
- Anyone who wants to connect the basics to Transformers and LLMs.

Readers need basic programming. Advanced calculus and linear algebra are introduced where needed.

## Scope

| Area | Included | Not included |
|---|---|---|
| Single neuron, perceptron, AND/OR/XOR | Yes | |
| Activations, losses, forward pass, gradient descent | Yes | |
| Backpropagation (two-layer network) | Yes | Deep (many-layer) training |
| Multilayer XOR and learning-rate experiments | Yes | Large benchmarks |
| Attention and one Transformer block | Yes | Training a language model |
| Deep-learning frameworks | No | PyTorch, TensorFlow and scikit-learn are not used, so the calculations stay visible |

## Design decisions

- **Layout.** The master plan proposed separate folders for each gate experiment. All three gate experiments share one script, `experiments/logic_gates.py`, because they use the same model and settings. Only the multilayer and learning-rate experiments have their own folders.
- **Notebook 11.** A notebook on attention and one Transformer block was added beyond the ten-notebook plan, because attention is central to modern language models.
- **Tests.** The code is tested with the standard library `unittest` module. The 48 tests check hand-computed values, finite-difference gradients, and the matrix form against the loop form.
- **Random seeds.** Initialization uses explicit seeds. The perceptron starts from zero, so it needs no seed.

## Status

See the status table in the [README](../README.md#development-status). Anything listed as planned has no verified code or results yet.

## Reproducing everything

```bash
python -m unittest discover -s tests -v
python experiments/logic_gates.py
python experiments/multilayer_network/run.py
python experiments/learning_rate_study/run.py
```

The multilayer experiment takes about two minutes. The learning-rate study takes about twenty seconds.
