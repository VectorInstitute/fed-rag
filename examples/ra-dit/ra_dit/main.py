"""Federate RAG via RA-DIT."""

# logging
import logging
from typing import Literal

import torch
from datasets import Dataset, load_dataset
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

tasks = ["retriever", "generator"]

main_logger = logging.getLogger("ra_dit.main")


# Models and Train/Test Functions
def get_model(
    task: str,
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
    server: bool = False,
) -> torch.nn.Module:
    main_logger.info(
        f"Getting model: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}' "
        f"server={server}"
    )
    rag_system = get_rag_system(retriever_id, generator_id, generator_variant)

    if task == "retriever":
        return rag_system.retriever.query_encoder
    elif task == "generator":
        if server:
            # lazy load model, so we can still set device map to cpu
            rag_system.generator.load_model_kwargs.update(device_map="cpu")
            rag_system.generator.load_base_model_kwargs.update(
                device_map="cpu"
            )
        return rag_system.generator.model
    else:
        main_logger.error(f"Got unsupported task: '{task}'")
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
    generator_variant: Literal["plain", "lora", "qlora"],
) -> HuggingFaceFlowerServer:
    main_logger.info(
        f"Building FL server: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )
    fl_task = fl_tasks[task]
    model = get_model(
        task, retriever_id, generator_id, generator_variant, server=True
    )
    main_logger.info(f"Server model loaded into device: {model.device}")
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
        "train_dataset": load_dataset(
            "nerdai/fedrag-commonsense-qa", split="train[:10]"
        ),
        "val_dataset": load_dataset(
            "nerdai/fedrag-commonsense-qa", split="test[:10]"
        ),
    },
}


# dummy setup with clients using the same datasets
def build_client(
    task: Literal["retriever", "generator"],
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
) -> HuggingFaceFlowerClient:
    main_logger.info(
        f"Building client: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}' "
    )
    fl_task = fl_tasks[task]
    model = get_model(task, retriever_id, generator_id, generator_variant)
    return fl_task.client(
        model=model,
        train_data=datasets[task]["train_dataset"],
        val_data=datasets[task]["val_dataset"],
        peft_config=model.active_peft_config,
    )


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def start(
    task: Literal["retriever", "generator"],
    component: Literal["server", "client_1", "client_2"],
    retriever_id: str = "dragon",
    generator_id: str = "llama2_7b",
    generator_variant: Literal["plain", "lora", "qlora"] = "qlora",
) -> None:
    """For starting any of the FL Task components."""
    main_logger.info(
        f"Starting FL component '{component}' for: task='{task}', retriever_id='{retriever_id}', "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )
    import flwr as fl

    if task not in ["retriever", "generator"]:
        main_logger.error(f"Got unsupported task: '{task}'")
        raise ValueError("Unrecognized task.")

    grpc_max_message_length = int(512 * 1024 * 1024 * 3.75)

    if component == "server":
        server = build_server(
            task, retriever_id, generator_id, generator_variant
        )
        main_logger.info(
            "Successfully built FL server, commencing server start."
        )
        fl.server.start_server(
            server=server,
            server_address="[::]:8080",
            grpc_max_message_length=grpc_max_message_length,
        )
        main_logger.info("FL server successfully completed training session.")
    elif component in ["client_1", "client_2"]:
        client = build_client(
            task=task,
            retriever_id=retriever_id,
            generator_id=generator_id,
            generator_variant=generator_variant,
        )
        main_logger.info(
            f"Successfully built FL client for '{component}', commencing client start."
        )
        fl.client.start_client(
            client=client,
            server_address="[::]:8080",
            grpc_max_message_length=grpc_max_message_length,
        )
        main_logger.info(
            f"FL client '{component}' has successfully completed local training."
        )
    else:
        main_logger.error(f"Got unsupported component: '{component}'")
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(start)
