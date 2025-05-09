# Fine-tune a RAG System

<!-- markdownlint-disable-file MD033 MD046 -->

In this quick start, we'll go over how we can take the RAG system we built from
the previous quick start example, and fine-tune it.

!!! note
    For this fine-tuning tutorial, you'll need the `huggingface` extra installed.
    If you haven't added it yet, run:

    `pip install fed-rag[huggingface]`

    This provides access to the HuggingFace models and training utilities we'll
    use for both retriever and generator fine-tuning.

## The Train Dataset

Training a RAG system requires a train dataset that is familiarly shaped as a question-answering
dataset.

```py title="training examples for RAG fine-tuning"
from datasets import Dataset

train_dataset = Dataset.from_dict(  # (1)!
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
)
```

1. A train example is essentially a (`query`, `response`) pair.

## Define our Trainer objects

To perform RAG fine-tuning, FedRAG offers both a [`BaseGeneratorTrainer`](../../api_reference/trainers/index.md)
and a [`BaseRetrieverTrainer`](../../api_reference/trainers/index.md) that incorporate
the training logic for each of these respective RAG components.

For this quick start, we make use of the following trainers:

- [`HuggingFaceTrainerForRALT`](../../api_reference/trainers/huggingface.md) — A
  generator trainer that fine-tunes the LLM using retrieval-augmented instruction
  examples.
- [`HuggingFaceTrainerForLSR`](../../api_reference/trainers/huggingface.md) — A
  retriever trainer that fine-tunes the retriever model using retrieval chunk scores
  and the log probabilities derived from the generator LLM using the ground truth
  response.

``` py title="retrieval-augmented fine-tuning"
from fed_rag.trainers.huggingface.ralt import HuggingFaceTrainerForRALT
from fed_rag.trainers.huggingface.lsr import HuggingFaceTrainerForLSR


rag_system = ...  # from previous quick start
generator_trainer = HuggingFaceTrainerForRALT(
    rag_system=rag_system,
    train_dataset=train_dataset,
)
retriever_trainer = HuggingFaceTrainerForLSR(
    rag_system=rag_system,
    train_dataset=train_dataset,
)
```

## Define our Trainer Manager object

To orchestrate training between the two RAG components, FedRAG offers a manager
class called [`BaseTrainerManager`](../../api_reference/trainer_managers/index.md).
The training manager contains logic to prepare the component and system for the
specific training task (i.e., retriever or generator), and also contains a simple
method to transform the task into a federated one.

```py title="training with managers"
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager

manager = HuggingFaceRAGTrainerManager(
    mode="retriever",  # (1)!
    retriever_trainer=retriever_trainer,
    generator_trainer=generator_trainer,
)
train_result = manager.train()
print(f"loss: {train_result.loss}")

# get your federated learning task (optional)
fl_task = manager.get_federated_task()
```

1. Mode can be "retriever" or "generator"—see [`RAGTrainMode`](../../api_reference/trainer_managers/index.md)
