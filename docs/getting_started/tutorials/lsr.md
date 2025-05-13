# Retriever Fine-Tuning with LSR

RAG systems integrate three key components: a knowledge store, a retriever model,
and a generator model. Effective fine-tuning approaches should enhance not just
individual components, but the cohesive performance of the entire system.

This tutorial focuses on the LM-Supervised Retriever (LSR) fine-tuning method, which
was first introduced in Shi, Weijia et al. (2023)[^1] and later generalized in
Lin, Xi Victoria et al. (2023)[^2]. With this method the retriever is fine-tuned
by leveraging signals from the language model (i.e., generator).

## The LSR Method

The LSR method is applied over a training dataset of (query, response) pairs. It
involves the computation of two probability distributions for every pair, namely:
a probability distribution derived the from the retrieval scores of each of the
retrieved knowledge nodes, and a probability distribution derived from the LLM
generator conditioned on prompt and each of the knowledge node's context.

Mathematically, let $(q, r)$ represent the query and response, respectively of a
given training example. Moreover let $c_i$ represent the context from the $i$-th
knowledge node retrieved by the RAG system for query, $q$.

### Retrieval probabilities

The retrieval probability distribution is defined by applying the softmax function
to the retrieval scores:

$$
p_{R}(c_i|q) = \frac{\exp s(q, c_i)}{\sum_{j=1}^k \exp s(q, c_j)}, \quad i=1,\ldots,k,
$$

where $s(q, c_i)$ represents the similarity score between query, $q$ and context,
$c_i$.

### Generation (LLM) probabilities

Similarly, the probabilities derived from the LLM involves another application of
the softmax function, but this time over the probabilities that the LLM generator
produces the ground-truth response, $r$ when given the input sequence of $q \circ c_i$
for $i=1,\ldots,k$, where "$\circ$" is the concatenation operator. That is,

$$
p_{LSR}(c_i|q,r) = \frac{\exp (P_{LLM}(r| q \circ c_i)/\tau)}{\sum_{j=1}^k \exp(P_{LLM}(r | q \circ c_j)/\tau)}, \quad i=1,\ldots,k,
$$

where $\tau$ is a temperature hyperparameter.

### LSR Training objective

The LSR loss is defined as the Kullback-Liebler divergence between these two probability
distributions:

$$
\mathcal{L}_{LSR} = \mathbb{E}_{(q,r)\in\mathcal{D}_{\text{train}}} KL\big(p_{R}(c|q)\|p_{LSR}(c|q,r)\big).
$$

In minimizing the LSR loss, we are adapting the retriever model to assign higher
scores to the knowledge nodes that increase the generator's likelihood of producing
the ground-truth response, $r$.

## Notes on the FedRAG Implementation of LSR

In FedRAG, we implement the LSR method with the typical coordination of a data collator
and a trainer. The `DataCollatorForLSR` takes a batch of (query, response) pairs
and produces the PyTorch tensors for retrieval scores as well as LLM scores for
each knowledge node retrieved by the `RAGSystem`. This batch is then used by the
`TrainerForLSR` which computes the `LSRLoss` for the batch and then performs the
optimization step for the given training step.

<!-- References -->
[^1]: Shi, Weijia, et al. "Replug: Retrieval-augmented black-box language models."
  arXiv preprint arXiv:2301.12652 (2023).
[^2]: Lin, Xi Victoria, et al. "Ra-dit: Retrieval-augmented dual instruction tuning."
  The Twelfth International Conference on Learning Representations. 2023.
