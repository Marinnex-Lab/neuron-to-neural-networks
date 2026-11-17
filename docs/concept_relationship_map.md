# Concept relationship map

The map shows how the concepts depend on each other. An arrow from A to B means B uses A, or A explains a limit of B.

```mermaid
flowchart TD
    N[Neuron: z = w.x + b, y = f z] --> P[Perceptron: step + learning rule]
    N --> L[Layer: Z = XW + b]
    P --> LS[Linear separability]
    LS --> XOR[XOR is not separable]
    XOR --> H[Hidden layer needed]
    H --> L
    L --> F[Forward pass: stacked layers]
    F --> ACT[Activation: nonlinearity]
    ACT -.->|without it, layers collapse to one| L
    ACT --> DER[Activation derivative]
    F --> LOSS[Loss: MSE, BCE, CCE]
    LOSS --> G[Output error: p - y]
    G --> BP[Backpropagation: chain rule]
    DER --> BP
    BP --> GD[Gradient descent]
    GD --> OPT[Optimizers: momentum, RMSProp, Adam]
    OPT --> LR[Learning rate and stability]
    BP --> VG[Vanishing and exploding gradients]
    INIT[Initialization and normalization] --> VG
    VG --> RES[Residual connections and layer norm]
    L --> ATT[Attention: softmax(QK^T/sqrt d_k)V]
    ATT --> TB[Transformer block]
    RES --> TB
    TB --> LLM[Next-token prediction with cross-entropy]
    LOSS --> LLM
```

## Reading the map

- **Linear separability** explains the perceptron's limit. It motivates the hidden layer.
- **Nonlinearity** is what makes depth useful. The dotted line shows that removing it collapses the layers.
- **Output error** $p-y$ is the signal that starts backpropagation. The same form appears for BCE and cross-entropy.
- **Attention** replaces the fixed weights of a dense layer with weights computed from the input.

## Notebook for each node

| Node | Notebook or module |
|---|---|
| Neuron | `notebooks/01_artificial_neuron.ipynb` |
| Perceptron, separability | `notebooks/02_perceptron_and_linear_separability.ipynb` |
| XOR, hidden layer | `notebooks/03_and_or_xor_experiments.ipynb`, `notebooks/09_multilayer_network.ipynb` |
| Activations | `notebooks/04_activation_functions.ipynb` |
| Forward pass | `notebooks/05_forward_propagation.ipynb` |
| Losses | `notebooks/06_loss_functions.ipynb` |
| Gradient descent, optimizers | `notebooks/07_gradient_descent.ipynb` |
| Backpropagation | `notebooks/08_backpropagation.ipynb` |
| Experiment design | `notebooks/10_neural_network_experiments.ipynb` |
| Attention, transformer block | `notebooks/11_attention_and_transformer_block.ipynb` |
