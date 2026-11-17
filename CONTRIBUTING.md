# Contributing

Thank you for reading. Contributions that improve correctness, clarity, or reproducibility are welcome.

## Principles

1. **Explain why, not only what.** Every formula should say what it computes, why it is needed, and how it affects the whole network.
2. **Separate status from claims.** Label content as implemented, notes only, or planned. Do not describe planned work as finished.
3. **Run before reporting.** Any number, figure, or result in a document must come from a script in this repository that has been run. Do not type results by hand.
4. **Keep dependencies small.** The core stack is NumPy and Matplotlib. Do not use PyTorch, TensorFlow, or scikit-learn to hide the calculations.
5. **Use credible sources.** Cite original papers, textbooks, or official documentation. Do not invent references.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Before you open a change

```bash
python -m unittest discover -s tests -v
python experiments/logic_gates.py
```

Both must succeed. If a figure or report changes, explain why in the change description.

## Code style

- Python 3.9 or newer. Use type hints for public functions.
- Keep modules in `src/` small and tested. Put exploration in `notebooks/`.
- Use controlled, deterministic settings. If randomness is used, set and report the seed.

## Commit messages

Use short imperative messages, for example `Add gradient check for two-layer network`.

## Scope

This repository is an educational project. Changes that claim research results must include the script that produced them and the settings used.
