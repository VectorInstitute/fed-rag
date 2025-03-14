"""Federate RAG via RA-DIT."""

from typing import Literal

import torch
from datasets import Dataset

# fedrag
from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask

from .retriever_train_loop import (
    rag_system,
    retriever_evaluate,
    retriever_train_loop,
)

# Create your FLTask
retrieval_fl_task = HuggingFaceFLTask.from_trainer_and_tester(
    trainer=retriever_train_loop, tester=retriever_evaluate
)

## 1. construct a server
model = rag_system.retriever.query_encoder
server = retrieval_fl_task.server(model=model)

### 2. construct a client trainer
clients = []
for i in range(2):
    train_dataset = Dataset.from_dict(
        {
            "text1": ["My first sentence", "Another pair"],
            "text2": ["My second sentence", "Unrelated sentence"],
            "label": [0.8, 0.3],
        }
    )
    device = torch.device("cuda:0")

    client = retrieval_fl_task.client(
        # train params
        model=model,
        train_data=train_dataset,
        val_data=train_dataset,
        device=device,
    )
    clients.append(client)


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def start_component(
    component: Literal["server", "client_1", "client_2"]
) -> None:
    """For starting any of the FL Task components."""
    import flwr as fl

    if component == "server":
        fl.server.start_server(server=server, server_address="[::]:8080")
    elif component == "client_1":
        fl.client.start_client(client=clients[0], server_address="[::]:8080")
    elif component == "client_2":
        fl.client.start_client(client=clients[1], server_address="[::]:8080")
    else:
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(start_component)
