"""Federate RAG via RA-DIT."""

# logging
from logging import INFO
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from flwr.common.logger import log
from ra_dit.trainers_and_testers.generator import (
    generator_evaluate,
    generator_train_loop,
)
from ra_dit.trainers_and_testers.retriever import (
    retriever_evaluate,
    retriever_train_loop,
)

# fedrag
from fed_rag.fl_tasks.huggingface import (
    HuggingFaceFlowerClient,
    HuggingFaceFlowerServer,
    HuggingFaceFLTask,
)

from .rag_system import main as get_rag_system

# import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

tasks = ["retriever", "generator"]


# Models and Train/Test Functions
def get_model(
    task: str, retriever_id: str, generator_id: str
) -> torch.nn.Module:
    rag_system = get_rag_system(retriever_id, generator_id)

    if task == "retriever":
        return rag_system.retriever.query_encoder
    elif task == "generator":
        return rag_system.generator.model
    else:
        raise ValueError("Unsupported task")


trainers = {
    "retriever": retriever_train_loop,
    "generator": generator_train_loop,
}
testers = {"retriever": retriever_evaluate, "generator": generator_evaluate}

# Create your FLTasks
fl_tasks = {
    key: HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainers[key], tester=testers[key]
    )
    for key in tasks
}


## Servers
def build_server(
    task: Literal["retriever", "generator"],
    retriever_id: str,
    generator_id: str,
) -> HuggingFaceFlowerServer:
    fl_task = fl_tasks[task]
    model = get_model(task, retriever_id, generator_id)
    log(INFO, f"Server model loaded into device: {model.device}")
    return fl_task.server(model=model)


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
def build_client(
    task: Literal["retriever", "generator"],
    retriever_id: str,
    generator_id: str,
    device_id: str,
) -> HuggingFaceFlowerClient:
    fl_task = fl_tasks[task]
    device = torch.device(f"cuda:{device_id}")
    model = get_model(task, retriever_id, generator_id).to(device)
    log(INFO, f"Client model loaded into device: {model.device}")
    return fl_task.client(
        model=model,
        train_data=datasets[task]["train_dataset"],
        val_data=datasets[task]["val_dataset"],
        device=device,
    )


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def start(
    task: Literal["retriever", "generator"],
    component: Literal["server", "client_1", "client_2"],
    retriever_id: str = "dragon",
    generator_id: str = "llama2_7b",
) -> None:
    """For starting any of the FL Task components."""
    import flwr as fl

    if task not in ["retriever", "generator"]:
        raise ValueError("Unrecognized task.")

    grpc_max_message_length = int(512 * 1024 * 1024 * 3.75)

    if component == "server":
        server = build_server(task, retriever_id, generator_id)
        fl.server.start_server(
            server=server,
            server_address="[::]:8080",
            grpc_max_message_length=grpc_max_message_length,
        )
    elif component == "client_1":
        log(INFO, "Starting client_1")
        fl.client.start_client(
            client=build_client(
                task=task,
                retriever_id=retriever_id,
                generator_id=generator_id,
                device_id="0",
            ),
            server_address="[::]:8080",
            grpc_max_message_length=grpc_max_message_length,
        )
    elif component == "client_2":
        fl.client.start_client(
            client=build_client(
                task=task,
                retriever_id=retriever_id,
                generator_id=generator_id,
                device_id="1",
            ),
            server_address="[::]:8080",
            grpc_max_message_length=grpc_max_message_length,
        )
    else:
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(start)
