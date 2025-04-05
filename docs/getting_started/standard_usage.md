# Standard Usage

The standard usage pattern for FedRAG aligns to a natural RAG fine-tuning workflow,
and looks as follows:

1. Build a [`RAGSystem`](../api_reference/rag_system/index.md)
2. Create a [`RAGFinetuningDataset`](../api_reference/finetuning_datasets/index.md)
3. Define a training loop and evaluation function and decorate both of these with
the appropriate [`decorators`](../api_reference/decorators/index.md).
4. Create an [`FLTask`](../api_reference/fl_tasks/index.md)
5. Spin up FL servers and clients to begin federated fine-tuning!

In the following subsections, we briefly elaborate on what's involved in each of
these listed steps.

!!! note
    Steps 1. through 3. are—minus the decoration of trainers and testers—typical
    steps one would perform for a centralized RAG fine-tuning task.

!!! tip
    Before proceeding to federated learning, one should verify that the centralized
    task runs as intended with a representative dataset. In fact, centralized learning
    represents a standard baseline with which to compare federated learning results.

## Build a `RAGSystem`

Building a [`RAGSystem`](../api_reference/rag_system/index.md) involves defining
a [`Retriever`](../api_reference/retrievers/index.md),
[`KnowledgeStore`](../api_reference/knowledge_stores/index.md) as well as
[`Generator`](../api_reference/generators/index.md), and subsequently supplying
these along with a [`RAGConfig`](../api_reference/rag_system/index.md) (to define
parameters such a `top_k`) to the [`RAGSystem`](../api_reference/rag_system/index.md)
constructor.

## Create a `RAGFinetuningDataset`

With a `RAGSystem` in place, we can create a fine-tuning dataset using examples
that contain queries and their associated answers. In retrieval-augmented generator
fine-tuning, we process each example by calling the `RAGSystem.retrieve()` method
with the query to fetch relevant knowledge nodes from the connected `KnowledgeStore`.
These contextual nodes enhance each example, creating a collection of
retrieval-augmented examples that form the RAG fine-tuning dataset for generator
model training. Our how-to guides provide detailed instructions on performing this
type of fine-tuning, as well as other approaches.

## Define a training loop and evaluation function

Like any model training process, a training loop establishes how the model learns
from the dataset. Since RAG systems are essentially assemblies of component models
(namely retriever and generator), we need to define a specific training loop to effectively learn from RAG fine-tuning datasets.

The lift to transform this from a centralized task is to a federated one is minimal
with FedRAG, and amounts to the application of trainer and tester decorators on
the respective functions.

``` py title="decorating training loops"
from fed_rag.decorators import federate


@federate.trainer.pytorch
def training_loop():
    ...
```

## Create an `FLTask`

## Spin up FL servers and clients
