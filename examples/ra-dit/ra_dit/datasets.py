"""Builds fine-tuning datasets for RA-DIT."""

<<<<<<< HEAD
import json
from pathlib import Path
=======
>>>>>>> 3075ca4 (scaffolding ra_dit.datasets)
from typing import Literal

from ra_dit.generators import GENERATORS
from ra_dit.knowledge_stores import KNOWLEDGE_STORES
from ra_dit.retrievers import RETRIEVERS

from fed_rag.types.rag_system import RAGConfig, RAGSystem
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
<<<<<<< HEAD
{answer}
=======
{response}
>>>>>>> 3075ca4 (scaffolding ra_dit.datasets)
</response>
"""


<<<<<<< HEAD
QA_DATA_REGISTRY = {"mock": "mock_qa_examples.jsonl"}
QA_DATA_PATH = Path(__file__).parents[1].absolute() / "data"


def _load_qa_dataset(name: str) -> list[dict[str, str]]:
    data_file_path = QA_DATA_PATH / QA_DATA_REGISTRY[name]
    examples = []
    with open(data_file_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples
=======
def _load_qa_dataset(name: str) -> list[dict[str, str]]:
    return []
>>>>>>> 3075ca4 (scaffolding ra_dit.datasets)


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
    retriever = RETRIEVERS[retriever_id]
    knowledge_store = KNOWLEDGE_STORES[f"from_{retriever_id}"]
    generator = GENERATORS[generator_id][generator_variant]

    rag_config = RAGConfig(top_k=2)
    rag_system = RAGSystem(
        knowledge_store=knowledge_store,  # knowledge store loaded from knowledge_store.py
        generator=generator,
        retriever=retriever,
        rag_config=rag_config,
    )

    # generate retrieval-augmented instruction tuning dataset
<<<<<<< HEAD
    unwrapped_tokenizer = generator.tokenizer.unwrapped

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
=======
    _unwrapped_tokenizer = generator.tokenizer.unwrapped
    eos_token_id = ...
    dataset: HuggingfaceRAGFinetuningDataset = build_finetune_dataset(
        rag_system=rag_system,
        examples=examples,
        eos_token_id=eos_token_id,
        finetune_example_template=...,
>>>>>>> 3075ca4 (scaffolding ra_dit.datasets)
        return_dataset="hf",
    )

    return dataset
<<<<<<< HEAD


if __name__ == "__main__":
    import fire

    dataset = fire.Fire(main)
    print(len(dataset))
    print(dataset[0])
=======
>>>>>>> 3075ca4 (scaffolding ra_dit.datasets)
