"""Builds fine-tuning datasets for RA-DIT."""

import json
import logging
import time
from pathlib import Path
from typing import Literal

from ra_dit.rag_system import main as get_rag_system

from fed_rag.types.rag_system import RAGSystem
from fed_rag.utils.data import build_finetune_dataset
from fed_rag.utils.data.finetuning_datasets.huggingface import (
    HuggingFaceRAGFinetuningDataset,
)

logger = logging.getLogger("ra_dit.datasets")

finetune_example_template = """<instruction>
Below is a user query and some background context. Write an answer to the user query.
</instruction>

<query>
{query}
</query>

<context>
{context}
</context>

<response>
{answer}
</response>
"""


QA_DATA_PATH = Path(__file__).parents[1].absolute() / "data" / "qa"
QA_DATA_REGISTRY: dict[str, str] = {
    "mock": "mock_qa_examples.jsonl",
    "commonsense_qa": "commonsense_qa.jsonl",
}
HF_HUB_PREFIX = "nerdai/fedrag-"


def _load_qa_dataset(name: str) -> list[dict[str, str]]:
    data_file_path = QA_DATA_PATH / QA_DATA_REGISTRY[name]
    examples = []
    with open(data_file_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main(
    retriever_id: str,
    generator_id: str,
    generator_variant: Literal["plain", "lora", "qlora"],
    qa_dataset_name: str,
    push_to_hub: bool = False,
) -> RAGSystem:
    """Build RAG Fine-Tuning Dataset."""
    logger.info(
        f"Creating fine-tuning dataset: retriever_id='{retriever_id}', "
        f"generator_id='{generator_id}', generator_variant='{generator_variant}' "
        f"qa_dataset_name={qa_dataset_name}, push_to_hub={push_to_hub}"
    )
    start_time = time.time()

    # load in QA dataset
    examples = _load_qa_dataset(qa_dataset_name)
    logger.info(f"Successfully loadding QA dataset: {qa_dataset_name}")
    logger.debug(f"Examples has {len(examples)} elements")

    # build rag system
    rag_system = get_rag_system(
        retriever_id=retriever_id,
        generator_id=generator_id,
        generator_variant="qlora",  # unused
    )
    logger.info("RAGSystem successfully constructed")
    logger.debug(f"Using retriever: {type(rag_system.retriever).__name__}")
    logger.debug(f"Using generator: {type(rag_system.generator).__name__}")

    # generate retrieval-augmented instruction tuning dataset
    unwrapped_tokenizer = rag_system.generator.tokenizer.unwrapped
    logger.debug(f"Tokenizer vocabulary size: {len(unwrapped_tokenizer)}")
    logger.debug(
        f"Tokenizer model max length: {unwrapped_tokenizer.model_max_length}"
    )

    # find eos_token_id
    try:
        eos_token = unwrapped_tokenizer.special_tokens_map.get("eos_token")
        logger.info(f"Successfully found tokenizer eos_token: {eos_token}")
    except KeyError:
        raise ValueError("Tokenizer doesn't have an `eos_token`.")
    eos_token_ix = unwrapped_tokenizer.all_special_tokens.index(eos_token)
    eos_token_id = unwrapped_tokenizer.all_special_ids[eos_token_ix]

    # build dataset
    dataset: HuggingFaceRAGFinetuningDataset = build_finetune_dataset(
        rag_system=rag_system,
        examples=examples,
        answer_key="answer",
        query_key="question",
        eos_token_id=eos_token_id,
        finetune_example_template=finetune_example_template,
        return_dataset="hf",
    )
    logger.info("Fine-tuning dataset successfully created")
    logger.info(
        f"Dataset creation took {time.time() - start_time:.2f} seconds"
    )
    logger.debug(f"Dataset has {len(dataset)} fine-tuning examples")
    logger.info("Fine-tuning dataset successfully created")
    logger.debug(
        f"Started with {len(examples)} qa examples, "
        f"finished with {len(dataset)} retrieval-augmented examples"
    )

    if push_to_hub:
        hf_dataset_name = HF_HUB_PREFIX + qa_dataset_name.replace("_", "-")
        dataset.push_to_hub(hf_dataset_name)
        logger.info(
            f"Succesfully pushed to HF hub, dataset: {hf_dataset_name}"
        )

    return dataset


if __name__ == "__main__":
    import fire

    def custom_serializer(obj: object) -> str:
        """Fire has an issue in finding the __str__ method for Datasets. I've
        logged an issue in their Github for this. If addressed, we can get rid
        of this.

        https://github.com/google/python-fire/issues/595
        """
        if hasattr(obj, "__str__"):
            return str(obj)
        else:
            return ""

    fire.Fire(main, serialize=custom_serializer)
