# Mathematical prerequisites

Each item lists where it is first used. Work through the items in order if you are new to the mathematics.

| Topic | What you need | Used in |
|---|---|---|
| Functions | input, output, composition | every module |
| Summation | $\sum_i a_i$ | weighted sums, losses |
| Exponential and logarithm | $e^x$, $\log x$, $\log(ab)=\log a+\log b$ | sigmoid, softmax, cross-entropy |
| Vectors and dot product | $\mathbf{w}^\top\mathbf{x}=\sum_i w_i x_i$ | neuron, attention scores |
| Matrices and shapes | $(m\times n)(n\times p)=(m\times p)$, transpose | layers, batches, attention |
| Derivatives | $f'(x)$ as a slope and as a sensitivity | activations, gradients |
| Partial derivatives and gradient | $\partial f/\partial w_i$, $\nabla f$ | gradient descent |
| Chain rule | $dL/dw = dL/dz\cdot dz/dw$ | backpropagation |
| Probability basics | probability, expectation, likelihood | losses, softmax |
| Floating-point limits | overflow, rounding, machine epsilon | stable softmax, log-sum-exp, quantization |

## Practice questions

1. Compute $\dfrac{d}{dx}(3x+1)^2$ at $x=1$. (Answer: $24$.)
2. What is the shape of $AB$ if $A$ is $4\times5$ and $B$ is $5\times3$? (Answer: $4\times3$.)
3. Why does $\log(0)$ cause a problem in cross-entropy? (It is $-\infty$, so the loss is infinite.)

The module notebooks state the mathematics they need, so these items can also be learned in context.
