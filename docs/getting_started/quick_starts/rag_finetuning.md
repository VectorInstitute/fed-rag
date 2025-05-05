# Fine-tune a RAG System

<!-- markdownlint-disable-file MD033 -->

In this quick start, we'll go over how we can take the RAG system we built from
the previous quick start example, and fine-tune it.

!!! note
    For this fine-tuning tutorial, you'll need the `huggingface` extra installed.
    If you haven't added it yet, run:

    `pip install fed-rag[huggingface]`

    This provides access to the HuggingFace models and training utilities we'll
    use for both retriever and generator fine-tuning.

## Retrieval-Augmented Generator Fine-tuning

The objective of retrieval-augmented generator fine-tuning is to adapt the LLM
to make more effective use of the retrieved passages or chunks. We accomplish this
by using the causal language modelling task on instruction examples that contain
a user query, some retrieved context passage, and finally a response.

Performing this kind of fine-tuning is made easy using FedRAG abstractions as
demonstrated below.

``` py title="retrieval-augmented fine-tuning"
from fed_rag.trainers.huggingface.ralt import HuggingFaceTrainerForRALT

rag_system = ...  # RAG system from previous quick start example
train_dataset = [
    {
        "query": [
            "What is machine learning?",
            "Tell me about climate change",
            "How do computers work?",
        ],
        "response": [
            "Machine learning is a field of AI focused on algorithms that learn from data.",
            "Climate change refers to long-term shifts in temperatures and weather patterns.",
            "Computers work by processing information using logic gates and electronic components.",
        ],
    }
]
trainer = HuggingFaceTrainerForRALT(rag_system, train_dataset)
train_result = trainer.train()
print(f"loss: {train_result.loss}")
```

## Language-Model Supervised Retriever Fine-tuning

In language-model supervised retriever fine-tuning or LSR for short, the goal is
to adapt the retriever model using the log probabilities outputted by the LLM
generator.

``` py title="lm-supervised retriever fine-tuning"
from fed_rag.trainers.huggingface.lsr import HuggingFaceTrainerForLSR

rag_system = ...  # RAG system from previous quick start example
train_dataset = [
    {
        "query": [
            "What is machine learning?",
            "Tell me about climate change",
            "How do computers work?",
        ],
        "response": [
            "Machine learning is a field of AI focused on algorithms that learn from data.",
            "Climate change refers to long-term shifts in temperatures and weather patterns.",
            "Computers work by processing information using logic gates and electronic components.",
        ],
    }
]
trainer = HuggingFaceTrainerForLSR(rag_system, train_dataset)
train_result = trainer.train()
print(f"loss: {train_result.loss}")
```
