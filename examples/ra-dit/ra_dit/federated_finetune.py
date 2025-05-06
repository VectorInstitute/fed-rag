"""Centralized RAG Fine-Tuning"""

import logging
from typing import Literal

import torch
from datasets import Dataset

from fed_rag.fl_tasks.huggingface import (
    HuggingFaceFlowerClient,
    HuggingFaceFlowerServer,
)
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager
from fed_rag.trainers.huggingface.lsr import HuggingFaceTrainerForLSR
from fed_rag.trainers.huggingface.ralt import HuggingFaceTrainerForRALT

from .rag_system import main as get_rag_system

GRPC_MAX_MESSAGE_LENGTH = int(512 * 1024 * 1024 * 3.75)
TASKS = ["retriever", "generator"]


logger = logging.getLogger("ra_dit.centralized")

# Define your train examples. You need more than just two examples...
train_dataset = Dataset.from_dict(
    {
        "query": [
            "What is machine learning?",
            "Tell me about climate change",
            "How do computers work?",
            "Explain quantum physics",
        ],
        "response": [
            "Machine learning is a field of AI focused on algorithms that learn from data.",
            "Climate change refers to long-term shifts in temperatures and weather patterns.",
            "Computers work by processing information using logic gates and electronic components.",
            "Quantum physics studies matter and energy at the smallest scales of atoms and subatomic particles.",
        ],
    }
)

val_dataset = Dataset.from_dict(
    {
        "query": ["Yet another query"],
        "response": ["Yet another response"],
    }
)


# Models and Train/Test Functions
def get_train_manager(
    task: str,
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
    retriever_checkpoint_path: str | None = None,
    generator_checkpoint_path: str | None = None,
) -> torch.nn.Module:
    logger.info(
        f"Getting train manager for: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )

    rag_system = get_rag_system(
        retriever_id,
        generator_id,
        generator_variant,
        retriever_checkpoint_path,
        generator_checkpoint_path,
    )
    retriever_trainer = HuggingFaceTrainerForLSR(rag_system, train_dataset)
    generator_trainer = HuggingFaceTrainerForRALT(rag_system, train_dataset)
    return HuggingFaceRAGTrainerManager(
        mode=task,
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )


def build_client(
    train_manager: HuggingFaceRAGTrainerManager,
) -> HuggingFaceFlowerClient:
    logger.info("Building client")
    fl_task = train_manager.get_federated_task()
    model = train_manager.model
    return fl_task.client(
        model=model, train_dataset=train_dataset, val_dataset=val_dataset
    )


def build_server(
    train_manager: HuggingFaceRAGTrainerManager,
) -> HuggingFaceFlowerServer:
    logger.info("Building server")
    fl_task = train_manager.get_federated_task()
    model = train_manager.model
    return fl_task.server(model=model)


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def main(
    task: Literal["retriever", "generator"],
    component: Literal["server", "client_1", "client_2"],
    retriever_id: str = "dragon",
    generator_id: str = "llama2_7b",
    generator_variant: Literal["plain", "lora", "qlora"] = "qlora",
    retriever_checkpoint_path: str | None = None,
    generator_checkpoint_path: str | None = None,
) -> None:
    """For starting any of the FL Task components.

    Example:
        # retriever federated fine-tuning (LSR)
        ## server
        `python -m ra_dit.federated_finetune --task retriever --component server`

        ## client 1
        `python -m ra_dit.federated_finetune --task retriever --component client_1`

        ## client 1
        `python -m ra_dit.federated_finetune --task retriever --component client_2`
    """
    logger.info(
        f"Starting FL component '{component}' for: task='{task}', retriever_id='{retriever_id}', "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )
    import flwr as fl

    if task not in ["retriever", "generator"]:
        logger.error(f"Got unsupported task: '{task}'")
        raise ValueError("Unrecognized task.")

    manager = get_train_manager(
        task=task,
        retriever_id=retriever_id,
        generator_id=generator_id,
        generator_variant=generator_variant,
        retriever_checkpoint_path=retriever_checkpoint_path,
        generator_checkpoint_path=generator_checkpoint_path,
    )

    if component == "server":
        server = build_server(manager)
        logger.info("Successfully built FL server, commencing server start.")
        fl.server.start_server(
            server=server,
            server_address="[::]:8080",
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )
        logger.info("FL server successfully completed training session.")
    elif component in ["client_1", "client_2"]:
        client = build_client(manager)
        logger.info(
            f"Successfully built FL client for '{component}', commencing client start."
        )
        fl.client.start_client(
            client=client,
            server_address="[::]:8080",
            grpc_max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        )
        logger.info(
            f"FL client '{component}' has successfully completed local training."
        )
    else:
        logger.error(f"Got unsupported component: '{component}'")
        raise ValueError("Unrecognized component.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
