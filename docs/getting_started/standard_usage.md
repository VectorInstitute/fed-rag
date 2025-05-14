# Standard Usage

The standard usage pattern for fine-tuning a RAG system with FedRAG follows the
below listed steps:

1. Build a `train_dataset` that contains examples of (query, response) pairs.
2. Specify a retriever trainer as well as a generator trainer.
3. Construct a RAG trainer manager and invoke the `train()` method
4. (Optional) Get the associated `FLTask` `RAGTrainerManager.get_federated_task()`

!!! info
    These steps assume that you have already constructed your `RAGSystem` that
    you intend to fine-tune.

!!! info
    The below code snippets require the `hugginface` extra to be installed, which
    can be done via a `pip install fed-rag[huggingface]`.

## Build a `train_dataset`

For now, all FedRAG trainers deal with datasets that comprise of examples with
(query, answer) pairs.

```py title="Example: a train dataset for HuggingFace"
from datasets import Dataset

train_dataset = Dataset.from_dict(
    {
        "query": ["a query", "another query", ...],
        "response": [
            "reponse to a query",
            "another response to another query",
            ...,
        ],
    }
)
```

## Specify a retriever and generator trainer

FedRAG trainer classes bear the responsibility of training the associated retriever
or generator on the training dataset. It has an attached data collator that takes
a batch of the training dataset and applies the "forward" pass of the RAG system
(i.e., retrieval from the knowledge store and if required, the subsequent generation
step), and returns the `~torch.Tensors` required for computing the desire loss.

These trainer classes take your `RAGSystem` as input amongst possibly other
parameters.

```py title="Example HuggingFaceTrainers"
from fed_rag.trainers.huggingface import (
    HuggingFaceTrainerForRALT,
    HuggingFaceTrainerForLSR,
)

retriever_trainer = HuggingFaceTrainerForLSR(rag_system)
generator_trainer = HuggingFaceTrainerForRALT(rag_system)
```

## Create a `RAGTrainerManager`

The trainer manager class is responsible for orchestrating the training of the RAG
system.

```py title="Example HuggingFaceRAGTrainerManager"
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager

trainer_manager = HuggingFaceRAGTrainerManager(
    mode="retriever",
    retriever_trainer=retriever_trainer,
    generator_trainer=generator_trainer,
)

# train
result = trainer_manager.train()
print(result.loss)
```

!!! note
    Alternating training of the retriever and generator can be done by modifying
    the `mode` attribute of the manager and calling `train()`. In the future, the
    trainer manager will be able to orchestrate between retriever and generator
    fine-tuning within a single epoch.

## (Optional) Get the `FLTask` for federated training

FedRAG trainer managers offer a simple way to get the associated `FLTask` for
federated fine-tuning.

```py title="Convert centralized to federated task"
fl_task = trainer_manager.get_federated_task()  # (1)!
```

1. This will return an `FLTask` for either the retriever trainer or the generator
trainer task, depending on the `mode` that the trainer manager is currently set on.

### Spin up FL servers and clients

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
fl.client.start_client(client=client, server_address="[::]:8080")
```

1. `flwr` is the backend federated learning framework for FedRAG and comes included
with the installation of `fed-rag`.
2. Can pass in FL aggregation strategy, otherwise defaults to federated averaging.
3. Requires the same arguments as the centralized `training_loop`.

!!! note
    Under the hood, `FLTask.server()` and `FLTask.client()` build `~flwr.Server`
    and `~flwr.Client` objects, respectively.
