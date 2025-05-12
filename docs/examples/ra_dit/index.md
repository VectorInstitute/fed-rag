# A comprehensive implementation of RA-DIT

<!-- markdownlint-disable-file MD041 MD033 MD042 -->

<a target="_blank" href="https://github.com/VectorInstitute/fed-rag/tree/main/examples/ra-dit">
  <img src="https://img.shields.io/badge/view_in_github-black?logo=github&style=flat" alt="View in Github"/>
</a>

We consider the paper "RA-DIT: Retrieval-Augmented Dual Instruction Tuning" by Lin,
Xi Victoria et al. (2023)[^1] and implement simplified versions of their experiments
using FedRAG. In this work, the authors build a RAG system and fine-tune both
the generator and retriever using a diverse question-answering (QA) datasets.
Their experimental results demonstrate that a fine-tuned RAG system consistently
outperforms two key baselines: a standalone generator LLM and an un-fine-tuned RAG
system. These findings highlight the substantial benefits of applying the RA-DIT
approach to enhance RAG system performance.

This comprehensive implementation demonstrates the key concepts and
techniques from the original research while adapting them for practical demonstration.
More specifically, in this example, we:

1. [Build a Qdrant Knowledge Store](./qdrant_knowledge_store_wikipedia.md) — Take
  artifacts derived from Wikipedia to populate a`QdrantKnowledgeStore`.

2. [Fine-tune with QA datasets](./qdrant_knowledge_store_wikipedia.md) — Build a
  [`RAGSystem`](../../api_reference/rag_system/index.md) and fine-tune it with
  some QA datasets using LSR and RALT trainers.

3. [Evaluate with Benchmarks](./benchmarking.md) — Benchmark our fine-tuned RAG system
  on MMLU and compare it to a few appropriate baselines.

4. [Federated Fine-tuning](./federated_finetune.md) — Demonstrate how we can go
  from centralized to federated fine-tuning of our RAG system.

!!! note
    Federated fine-tuning was not considered in Lin, Xi Victoria et al (2023)[^1].

<!-- References -->
[^1]: Lin, Xi Victoria, et al. "Ra-dit: Retrieval-augmented dual instruction tuning."
The Twelfth International Conference on Learning Representations. 2023.
