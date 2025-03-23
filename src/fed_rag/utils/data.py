"""Data utils"""

from enum import Enum
from typing import Any, Sequence

from fed_rag.types.rag_system import RAGSystem

DEFAULT_FINETUNE_EXAMPLE_TEMPLATE = "{query} {context} {answer}"


class ReturnType(str, Enum):
    PYTORCH = "pt"
    HUGGINGFACE = "hf"


def build_finetine_dataset(
    rag_system: RAGSystem,
    examples: Sequence[dict],
    finetune_example_template: str = DEFAULT_FINETUNE_EXAMPLE_TEMPLATE,
    query_key: str = "query",
    answer_key: str = "answer",
    return_dataset: ReturnType = "pt",
    num_workers: int = 1,
) -> Any:
    # query rag
    finetuning_instances = []
    for example in examples:
        source_nodes = rag_system.retrieve(query=example[query_key])
        for source in source_nodes:
            finetune_instance = finetune_example_template.format(
                query=example[query_key],
                answer=example[answer_key],
                context=[source.node.get_content()["text_content"]],
            )
            # tokenize to get input_ids and target_ids
            tokenizer = rag_system.generator.tokenizer
            eos_token: int = ...
            input_ids = tokenizer.encode(finetune_instance)
            target_ids = input_ids[1:] + [eos_token]

            finetuning_instances.append(finetune_instance)
