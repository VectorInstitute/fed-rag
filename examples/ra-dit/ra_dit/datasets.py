"""Builds fine-tuning datasets for RA-DIT."""

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
{response}
</response>
"""


def _load_qa_dataset(name: str) -> list[dict[str, str]]:
    return []


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
    _unwrapped_tokenizer = generator.tokenizer.unwrapped
    eos_token_id = ...
    dataset: HuggingfaceRAGFinetuningDataset = build_finetune_dataset(
        rag_system=rag_system,
        examples=examples,
        eos_token_id=eos_token_id,
        finetune_example_template=...,
        return_dataset="hf",
    )

    return dataset
