"""Data utils"""

from enum import Enum
from typing import Any, Sequence

import torch
from typing_extensions import assert_never

from fed_rag.types.rag_system import RAGSystem
from fed_rag.utils.data.finetuning_datasets import PyTorchRAGFinetuningDataset

DEFAULT_FINETUNE_EXAMPLE_TEMPLATE = "{query} {context} {answer}"


class ReturnType(str, Enum):
    PYTORCH = "pt"
    HUGGINGFACE = "hf"
    TEXT = "txt"


def build_finetune_dataset(
    rag_system: RAGSystem,
    examples: Sequence[dict],
    eos_token_id: int,
    finetune_example_template: str = DEFAULT_FINETUNE_EXAMPLE_TEMPLATE,
    query_key: str = "query",
    answer_key: str = "answer",
    return_dataset: ReturnType = ReturnType.PYTORCH,
) -> Any:
    """Generates the finetuning dataset using the supplied rag_system and examples."""

    if (
        isinstance(return_dataset, str)
        and return_dataset not in ReturnType._value2member_map_.keys()
    ):
        raise ValueError(
            "Invalid `return_type` specified."
        )  # TODO: give a proper exception to this

    inputs_list = []
    targets_list = []
    attention_mask_list = []
    finetuning_instances = []
    for example in examples:
        # retrieve
        source_nodes = rag_system.retrieve(query=example[query_key])
        total_sum_scores = sum(s.score for s in source_nodes)

        # parallel in-context retrieval-augmentation creates
        # top_k separated finetuning instances
        for source in source_nodes:
            finetune_instance_text = finetune_example_template.format(
                query=example[query_key],
                answer=example[answer_key],
                context=source.node.get_content()["text_content"],
            )
            finetuning_instances.append(finetune_instance_text)
            _weight = source.score / total_sum_scores

            # tokenize to get input_ids and target_ids
            tokenizer = rag_system.generator.tokenizer
            encode_result = tokenizer.encode(finetune_instance_text)
            input_ids = encode_result["input_ids"]
            attention_mask = encode_result["attention_mask"]
            target_ids = input_ids[1:] + [eos_token_id]

            inputs_list.append(input_ids)
            targets_list.append(target_ids)
            attention_mask_list.append(attention_mask)

    if return_dataset == ReturnType.TEXT:
        return finetuning_instances
    elif return_dataset == ReturnType.PYTORCH:
        return PyTorchRAGFinetuningDataset(
            input_ids=[torch.Tensor(el) for el in inputs_list],
            target_ids=[torch.Tensor(el) for el in targets_list],
        )
    elif return_dataset == ReturnType.HUGGINGFACE:
        # needs `fed-rag[huggingface]` extra to be installed
        # this import will fail if not installed
        from fed_rag.utils.data.finetuning_datasets.huggingface import (
            HuggingFaceRAGFinetuningDataset,
        )

        return HuggingFaceRAGFinetuningDataset.from_inputs(
            input_ids=inputs_list,
            target_ids=targets_list,
            attention_mask=attention_mask_list,
        )
    else:
        assert_never(return_dataset)  # pragma: no cover
