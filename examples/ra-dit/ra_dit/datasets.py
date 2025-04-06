"""Builds fine-tuning datasets for RA-DIT."""

import json
from pathlib import Path
from typing import Literal

from ra_dit.rag_system import main as get_rag_system

from fed_rag.types.rag_system import RAGSystem
from fed_rag.utils.data import build_finetune_dataset
from fed_rag.utils.data.finetuning_datasets.huggingface import (
    HuggingfaceRAGFinetuningDataset,
)

finetune_example_template = """<instruction>
...
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


QA_DATA_PATH = Path(__file__).parents[2].absolute() / "data" / "qa"
QA_DATA_REGISTRY: dict[str, str] = {}


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
) -> RAGSystem:
    """Build RAG Fine-Tuning Dataset."""

    # load in QA dataset
    examples = _load_qa_dataset(qa_dataset_name)

    # build rag system
    rag_system = get_rag_system(
        retriever_id=retriever_id,
        generator_id=generator_id,
        generator_variant="qlora",  # unused
    )

    # generate retrieval-augmented instruction tuning dataset
    unwrapped_tokenizer = rag_system.generator.tokenizer.unwrapped

    # find eos_token_id
    try:
        eos_token = unwrapped_tokenizer.special_tokens_map.get("eos_token")
    except KeyError:
        raise ValueError("Tokenizer doesn't have an `eos_token`.")
    eos_token_ix = unwrapped_tokenizer.all_special_tokens.index(eos_token)
    eos_token_id = unwrapped_tokenizer.all_special_ids[eos_token_ix]

    # build dataset
    dataset: HuggingfaceRAGFinetuningDataset = build_finetune_dataset(
        rag_system=rag_system,
        examples=examples,
        answer_key="response",
        eos_token_id=eos_token_id,
        finetune_example_template=finetune_example_template,
        return_dataset="hf",
    )

    return dataset


if __name__ == "__main__":
    import fire

    dataset = fire.Fire(main)
    print(len(dataset))
    print(dataset[0])
