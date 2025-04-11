"""Centralized RAG Fine-tuning RA-DIT."""

# logging
import logging
from typing import Any, Literal

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

from fed_rag.types.results import TrainResult

from .rag_system import main as get_rag_system

tasks = ["retriever", "generator"]

logger = logging.getLogger("ra_dit.centralized")


# Models and Train/Test Functions
def get_model(
    task: str,
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
) -> torch.nn.Module:
    logger.info(
        f"Getting model: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )
    rag_system = get_rag_system(retriever_id, generator_id, generator_variant)

    if task == "retriever":
        return rag_system.retriever.query_encoder
    elif task == "generator":
        return rag_system.generator.model
    else:
        logger.error(f"Got unsupported task: '{task}'")
        raise ValueError("Unsupported task")


trainers = {
    "retriever": retriever_train_loop,
    "generator": generator_train_loop,
}
testers = {"retriever": retriever_evaluate, "generator": generator_evaluate}


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
def execute_trainer(
    task: Literal["retriever", "generator"],
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
) -> Any:
    logger.info(
        f"Building client: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}' "
    )
    model = get_model(task, retriever_id, generator_id, generator_variant)
    trainer = trainers["generator"]
    train_result = trainer(
        model=model,
        train_data=datasets[task]["train_dataset"],
        val_data=datasets[task]["val_dataset"],
        peft_config=model.active_peft_config,
    )
    return train_result


# NOTE: The code below is merely for building a quick CLI to start server, and clients.
def start(
    task: Literal["retriever", "generator"],
    retriever_id: str = "dragon",
    generator_id: str = "llama2_7b",
    generator_variant: Literal["plain", "lora", "qlora"] = "qlora",
) -> TrainResult:
    """For starting any of the FL Task components."""
    logger.info(
        f"Executing trainer for: task='{task}', retriever_id='{retriever_id}', "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )

    if task not in ["retriever", "generator"]:
        logger.error(f"Got unsupported task: '{task}'")
        raise ValueError("Unrecognized task.")

    train_result = execute_trainer(
        task=task,
        retriever_id=retriever_id,
        generator_id=generator_id,
        generator_variant=generator_variant,
    )
    logger.info("Successfully executed trainer.")
    logger.debug(f"Train result: {train_result}")

    return train_result


if __name__ == "__main__":
    import fire

    fire.Fire(start)
