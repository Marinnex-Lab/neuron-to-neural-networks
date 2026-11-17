# References

Each entry was checked against a web search of bibliographic records, publisher or proceedings pages, arXiv, or ADS, during the final review of this repository. Where a detail could not be confirmed, the entry says so under **Note**. Check the entry against the publisher before a formal citation.

## Foundational work

- **Rosenblatt, F. (1958).** The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408. DOI: [10.1037/h0042519](https://doi.org/10.1037/h0042519). Introduces the perceptron and its learning procedure.
- **Novikoff, A. B. J. (1962).** On convergence proofs on perceptrons. *Proceedings of the Symposium on the Mathematical Theory of Automata*, 12, 615–622. **Note:** the symposium paper itself was not located. A related Office of Naval Research technical report by Novikoff (1963) is also cited in the literature. The title is cited differently by different sources, so check the proceedings directly. This entry is the source for the convergence bound quoted in the notes.
- **Minsky, M., & Papert, S. (1969).** *Perceptrons: An Introduction to Computational Geometry*. MIT Press. Shows the limits of single-layer perceptrons, including the XOR case. A later expanded edition appeared in 1988.
- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning representations by back-propagating errors. *Nature*, 323, 533–536. DOI: [10.1038/323533a0](https://doi.org/10.1038/323533a0). Popularizes backpropagation for training hidden layers.

## Textbooks

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. Standard reference covering the material in this repository. The title and publisher were not separately verified in this review.
- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Probabilistic view of losses and likelihoods. Not separately verified in this review.

## Overview article

- **LeCun, Y., Bengio, Y., & Hinton, G. (2015).** Deep learning. *Nature*, 521(7553), 436–444. DOI: [10.1038/nature14539](https://doi.org/10.1038/nature14539). A review of multi-layer representation learning and backpropagation. **Note:** a published critique disputes some historical attributions in this article. Read it as one account, not as a neutral history.

## Representation and approximation

- **Cybenko, G. (1989).** Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314. DOI: [10.1007/BF02551274](https://doi.org/10.1007/BF02551274). Universal approximation for a single hidden layer with sigmoidal units (existence, not learnability).
- **Hornik, K. (1991).** Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257. DOI: [10.1016/0893-6080(91)90009-T](https://doi.org/10.1016/0893-6080(91)90009-T). Extends the approximation result to general bounded nonconstant activations.
- **Telgarsky, M. (2016).** Benefits of depth in neural networks. *Proceedings of the 29th Conference on Learning Theory (COLT)*, PMLR 49, 1517–1539. arXiv: [1602.04485](https://arxiv.org/abs/1602.04485). Depth separation results: some functions need exponentially more units if the depth is reduced.

## Initialization, activations and normalization

- **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR 9, 249–256. Source of the Xavier/Glorot initialization.
- **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. arXiv: [1502.01852](https://arxiv.org/abs/1502.01852). Source of the He initialization for ReLU networks. **Note:** the search did not confirm the venue, so cite the arXiv version unless you confirm the conference.
- **Hendrycks, D., & Gimpel, K. (2016).** Gaussian error linear units (GELUs). arXiv: [1606.08415](https://arxiv.org/abs/1606.08415). Defines the GELU activation.
- **Ioffe, S., & Szegedy, C. (2015).** Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv: [1502.03167](https://arxiv.org/abs/1502.03167). **Note:** the search did not confirm the ICML venue, so cite the arXiv version unless you confirm it.
- **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).** Layer normalization. arXiv: [1607.06450](https://arxiv.org/abs/1607.06450). Normalizes across the features of each example, used in Transformers.

## Optimization

- **Kingma, D. P., & Ba, J. (2015).** Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. arXiv: [1412.6980](https://arxiv.org/abs/1412.6980).
- **Reddi, S. J., Kale, S., & Kumar, S. (2018).** On the convergence of Adam and beyond. *International Conference on Learning Representations (ICLR)*. Shows a convex counterexample for Adam and proposes variants.

## Regularization and efficiency

- **Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014).** Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(56), 1929–1958.
- **Hinton, G., Vinyals, O., & Dean, J. (2015).** Distilling the knowledge in a neural network. arXiv: [1503.02531](https://arxiv.org/abs/1503.02531). Originally a NIPS 2014 Deep Learning Workshop paper. Introduces distillation with temperature-scaled soft targets.
- **Jacob, B., Kligys, S., Chen, B., et al. (2018).** Quantization and training of neural networks for efficient integer-arithmetic-only inference. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2704–2713. DOI: [10.1109/CVPR.2018.00286](https://doi.org/10.1109/CVPR.2018.00286). Integer-only inference with 8-bit weights and activations.

## Transformers

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*. arXiv: [1706.03762](https://arxiv.org/abs/1706.03762). Introduces the Transformer and scaled dot-product attention. **Note:** the arXiv and proceedings abstracts report slightly different BLEU figures, so take numbers from the version you cite.
