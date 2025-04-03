# Essentials

To get to know FedRAG a bit better and understand its purpose, we provide the
answers to the following four essential questions.

## Four Essential Questions

### What is RAG?

Retrieval-Augmented Generation (RAG) is a widely used technique that addresses a main drawback of Large Language Models (LLM), which is that they're trained on
historical corpora and thus answering user queries that heavily rely on recent data
are not really possible. Further, using the parametric knowledge of LLMs alone
has yielded subpar performance on knowledge-intensive benchmarks.

RAG provides access to relevant (and potentially more recent) non-parametric
knowledge (i.e. data) that are stored in _Knowledge Stores_ to the LLM so that it
can use it in order to more accurately respond to user queries.

### What is Federated Learning?

Federated Learning (FL) is a technique for building machine learning (as well as
deep learning) models when the data is decentralized. Rather than first centralizing
the datasets to a central location, which may not be possible due to strict data
residency regulations or may be uneconomical due to the significant monetary costs in moving massive datasets, FL enables collaborative model building by facilitating
the sharing of the model weights between the data providers.

### Why Federated Fine-Tuning of RAG?

Fine-tuning is a technique that is used to enhance the performance of LLMs by
speciliazing its general capabilities towards a specific domain. It has also been
shown that fine-tuning the model components of RAG systems, namely the generator
and retriever, on domain-specific datasets can lead to its overall improved
performance.

Accessing fine-tuning datasets may be challenging. And, in situations where the
data is dispersed across several nodes, and centralizing is either not possible
or uneconomical, the fine-tuning of these RAG systems can be made possible through
FL.

### Who is FedRAG for?

FedRAG is for the model builders, data scientists, and researchers who wish to fine-tune
their RAG systems on their own datasets.

!!! note "Note — FedRAG also enables centralized RAG fine-tuning"

    While the main premise for FedRAG is FL, that centralized fine-tuning is also
    supported (as our readers shall see shortly in the upcoming Quick Start example).
