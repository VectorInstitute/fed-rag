"""Federate RAG via RA-DIT."""

from typing import Literal

import torch
from datasets import Dataset, load_dataset

# fedrag
from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask

from .rag_system import main as get_rag_system
from .retriever_train_loop import (
    retriever_evaluate,
    retriever_train_loop,
)
from .generator_train_loop import generator_train_loop, generator_evaluate

# RAG System

tasks = ["retriever", "generator"]


# Models and Train/Test Functions
def get_model(task):
    model_name = "/model-weights/Llama-3.2-1B"
    rag_system = get_rag_system(model_name)

    if task == "retriever":
        return rag_system.retriever.query_encoder
    elif task == "generator":
        return rag_system.generator.model
    else:
        raise ValueError("Unsupported task")


trainers = {"retriever": retriever_train_loop, "generator": generator_train_loop}
testers = {"retriever": retriever_evaluate, "generator": generator_evaluate}

# Create your FLTasks
fl_tasks = {
    key: HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainers[key], tester=testers[key]
    )
    for key in tasks
}


## Servers
def build_server(task: Literal["retriever", "generator"]):
    fl_task = fl_tasks[task]
    return fl_task.server(model=get_model(task))


## Clients
datasets = {
    "retriever": {
        "train_dataset": Dataset.from_dict(
            {
                "text1": ["My first sentence", "Another pair"],
                "text2": ["My second sentence", "Unrelated sentence"],
                "label": [0.8, 0.3],
            }
        ),
        "val_dataset": Dataset.from_dict(
            {
                "text1": ["My third sentence"],
                "text2": ["My fourth sentence"],
                "label": [0.99],
            }
        ),
    },
    "generator": {
        "train_dataset": load_dataset("stanfordnlp/imdb", split="train[:20]"),
        "val_dataset": load_dataset("stanfordnlp/imdb", split="test[:10]"),
    },
}


# dummy setup with clients using the same datasets
def build_client(task: Literal["retriever", "generator"], device_id: str):
    fl_task = fl_tasks[task]
    return fl_task.client(
        model=get_model(task),
        train_data=datasets[task]["train_dataset"],
        val_data=datasets[task]["val_dataset"],
        device=torch.device(f"cuda:{device_id}"),
    )


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def start(
    task: Literal["retriever", "generator"],
    component: Literal["server", "client_1", "client_2"],
) -> None:
    """For starting any of the FL Task components."""
    import flwr as fl

    if task not in ["retriever", "generator"]:
        raise ValueError("Unrecognized task.")

    server = build_server(task)

    if component == "server":
        fl.server.start_server(server=server, server_address="[::]:8080")
    elif component == "client_1":
        fl.client.start_client(
            client=build_client(task=task, device_id="0"), server_address="[::]:8080"
        )
    elif component == "client_2":
        fl.client.start_client(
            client=build_client(task=task, device_id="1"), server_address="[::]:8080"
        )
    else:
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(start)
