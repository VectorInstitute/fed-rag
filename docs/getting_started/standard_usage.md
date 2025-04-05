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

``` py title="building a rag system"
from fed_rag import RAGSystem, RAGConfig

# three main components
retriever = ...
knowledge_store = ...
generator = ...

rag = RAGSystem(
    generator=generator,
    retriever=retriever,
    knowledge_store=knowledge_store,
    rag_config=RAGConfig(top_k=2),
)
```

## Create a `RAGFinetuningDataset`

With a `RAGSystem` in place, we can create a fine-tuning dataset using examples
that contain queries and their associated answers. In retrieval-augmented generator
fine-tuning, we process each example by calling the `RAGSystem.retrieve()` method
with the query to fetch relevant knowledge nodes from the connected `KnowledgeStore`.
These contextual nodes enhance each example, creating a collection of
retrieval-augmented examples that form the RAG fine-tuning dataset for generator
model training. Our how-to guides provide detailed instructions on performing this
type of fine-tuning, as well as other approaches.

``` py title="creating a RAG fine-tuning dataset"
from fed_rag.utils.data import build_finetune_dataset

examples: list[dict[str, str]] = [{"query": ..., "answer": ...}, ...]

dataset = build_finetune_dataset(
    rag_system=rag_system, examples=examples, ...  # (1)!
)
```

1. Check the [API Reference](../api_reference/finetuning_datasets/index.md) for
the remaining required parameters

## Define a training loop and evaluation function

Like any model training process, a training loop establishes how the model learns
from the dataset. Since RAG systems are essentially assemblies of component models
(namely retriever and generator), we need to define a specific training loop to effectively learn from RAG fine-tuning datasets.

The lift to transform this from a centralized task is to a federated one is minimal
with FedRAG, and the first step towards this endeavour amounts to the application
of trainer and tester [`decorators`](../api_reference/decorators/index.md)
on the respective functions.

``` py title="decorating training loops and evaluation functions"
from fed_rag.decorators import federate


@federate.trainer.pytorch
def training_loop():
    ...


@federate.tester.pytorch
def evaluate():
    ...
```

These decorators perform inspection on these functions to automatically parse
the model as well as training and validation datasets.

## Create an `FLTask`

The final step in the federation transformation involves building an
[`FLTask`](../api_reference/fl_tasks/index.md) using the decorated trainer and
evaluation function.

``` py title="defining the FL task"
from fed_rag.fl_tasks.pytorch import PyTorchFLTask

# use from_trainer_tester class method
fl_task = PyTorchFLTask.from_trainer_and_tester(
    trainer=decorated_trainer, tester=decorated_tester  # (1)!
)
```

1. decorated with `federate.trainer.pytorch` and `federate.tester.pytorch`, respectively

## Spin up FL servers and clients

With an `FLTask`, we can obtain an FL server as well as clients. Starting a server
and required number of clients will commence the federated training.

``` py title="getting server and clients"
import flwr as fl  # (1)!

# federate generator fine-tuning
model = rag_system.generator.model

# server
server = fl_task.server(model, ...)  # (2)!

# client
client = fl_task.client(...)  # (3)!

# the below commands are blocking and would need to be run in separate processes
fl.server.start_server(server=server, server_address="[::]:8080")
fl.client.start_client(client=clients, server_address="[::]:8080")
```

1. `flwr` is the backend federated learning framework for FedRAG and comes included
with the installation of `fed-rag`.
2. Can pass in FL aggregation strategy, otherwise defaults to federated averaging.
3. Requires the same arguments as the centralized `training_loop`.

!!! note
    Under the hood, `FLTask.server()` and `FLTask.client()` build `~flwr.Server`
    and `~flwr.Client` objects, respectively.
