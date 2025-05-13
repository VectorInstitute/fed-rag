# Retriever Fine-Tuning with LSR

RAG systems integrate three key components: a knowledge store, a retriever model,
and a generator model. Effective fine-tuning approaches should enhance not just
individual components, but the cohesive performance of the entire system.

This tutorial focuses on the LM-Supervised Retriever (LSR) fine-tuning method, which
was first introduced in Shi, Weijia et al. (2023)[^1] and later generalized in
Lin, Xi Victoria et al. (2023)[^2]. With this method the retriever is fine-tuned
by leveraging signals from the language model (i.e., generator).

## The LSR Method

$$
\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
$$

<!-- References -->
[^1]: Shi, Weijia, et al. "Replug: Retrieval-augmented black-box language models."
  arXiv preprint arXiv:2301.12652 (2023).
[^2]: Lin, Xi Victoria, et al. "Ra-dit: Retrieval-augmented dual instruction tuning."
  The Twelfth International Conference on Learning Representations. 2023.
