"""Centralized RAG Fine-Tuning"""

import logging
from typing import Literal

import torch
from datasets import Dataset

from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager
from fed_rag.trainers.huggingface.lsr import HuggingFaceTrainerForLSR
from fed_rag.trainers.huggingface.ralt import HuggingFaceTrainerForRALT
from fed_rag.types.results import TrainResult

from .rag_system import main as get_rag_system

tasks = ["retriever", "generator"]

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
) -> torch.nn.Module:
    logger.info(
        f"Getting train manager for: task='{task}', retriver_id='{retriever_id}' "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}'"
    )

    rag_system = get_rag_system(retriever_id, generator_id, generator_variant)
    retriever_trainer = HuggingFaceTrainerForLSR(rag_system, train_dataset)
    generator_trainer = HuggingFaceTrainerForRALT(rag_system, train_dataset)
    return HuggingFaceRAGTrainerManager(
        mode=task,
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )


def main(
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

    manager = get_train_manager(
        task=task,
        retriever_id=retriever_id,
        generator_id=generator_id,
        generator_variant=generator_variant,
    )
    train_result = manager.train()
    logger.info("Successfully executed trainer.")
    logger.debug(f"Train result: {train_result}")

    return train_result


if __name__ == "__main__":
    import fire

    fire.Fire(main)
