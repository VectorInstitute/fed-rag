"""HuggingFace Data Collator For Retrieval-Augmented Generator Training"""

from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field

from fed_rag.base.data_collator import BaseDataCollator
from fed_rag.exceptions import DataCollatorError, MissingExtraError
from fed_rag.types.rag_system import RAGSystem
from fed_rag.utils.huggingface import _validate_rag_system

try:
    from transformers.data.data_collator import DataCollatorMixin

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    # Create a dummy class with a different name to avoid the redefinition
    class _DummyDataCollatorMixin:
        """Dummy placeholder when transformers is not available."""

        pass

    DataCollatorMixin = _DummyDataCollatorMixin  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedTokenizer


DEFAULT_EXAMPLE_TEMPLATE = """
You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<warning>
Only use the the provided context if its relevant to answer the question. Otherwise,
ignore it and use your parametric knowledge to answer the question.
</warning>

<question>
{query}
</question>

<context>
{context}
</context>

<response>
{response}
</response>
"""


class DataCollatorForRALT(DataCollatorMixin, BaseDataCollator):
    """A HuggingFace DataCollator for LM-Supervised Retrieval."""

    example_template: str = Field(default=DEFAULT_EXAMPLE_TEMPLATE)
    default_return_tensors: str = Field(default="pt")

    def __init__(
        self,
        rag_system: RAGSystem,
        example_template: str | None = None,
        default_return_tensors: str = "pt",
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        _validate_rag_system(rag_system)

        example_template = example_template or DEFAULT_EXAMPLE_TEMPLATE

        super().__init__(
            rag_system=rag_system,
            example_template=example_template,
            default_return_tensors=default_return_tensors,
            **kwargs,
        )

    def _apply_padding(
        self,
        max_length: int,
        inputs_list: list[list[int]],
        attention_mask_list: list[list[int]],
        tokenizer: "PreTrainedTokenizer",
    ) -> dict[str, torch.Tensor]:
        """Applys left padding for causal lm modelling."""

        # First convert lists to tensors if not already
        input_ids_tensors = [torch.tensor(ids) for ids in inputs_list]
        attention_mask_tensors = [
            torch.tensor(mask) for mask in attention_mask_list
        ]
        labels_tensors = [
            torch.tensor(ids) for ids in inputs_list
        ]  # Labels are the same as input_ids for causal LM

        # Get pad token ID
        if tokenizer.pad_token is not None:
            if tokenizer.pad_token_id < 0:
                raise DataCollatorError(
                    "Asking to pad but the tokenizer has a value for pad_token_id < 0."
                )
            pad_token_id = tokenizer.pad_token_id
        else:
            if tokenizer.eos_token_id is not None:
                pad_token_id = tokenizer.eos_token_id
            else:
                raise DataCollatorError(
                    "Asking to pad but the tokenizer does not have a padding token "
                    "nor an eos token that can potentially be used in its place."
                )

        # Create padded tensors
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for input_ids, attention_mask, labels in zip(
            input_ids_tensors, attention_mask_tensors, labels_tensors
        ):
            # Calculate padding needed
            pad_len = max_length - len(input_ids)

            if pad_len > 0:
                # Create padding tensors
                padding = torch.full(
                    (pad_len,), pad_token_id, dtype=input_ids.dtype
                )
                mask_padding = torch.zeros(pad_len, dtype=attention_mask.dtype)
                label_padding = torch.full(
                    (pad_len,), -100, dtype=labels.dtype
                )  # -100 to ignore in loss calculation

                # Apply left padding
                padded_input = torch.cat([padding, input_ids])
                padded_mask = torch.cat([mask_padding, attention_mask])
                padded_label = torch.cat([label_padding, labels])
            else:
                # No padding needed
                padded_input = input_ids
                padded_mask = attention_mask
                padded_label = labels

            padded_input_ids.append(padded_input)
            padded_attention_mask.append(padded_mask)
            padded_labels.append(padded_label)

        # Stack into batch tensors
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Any]:
        """Use the features of the dataset in order to get the `input_ids` and `labels`.

        Steps:
            1. process the features using the RAG system and example template to create
               the retrieval-augmented lm fine-tuning text
            2. apply padding and get required ~torch.Tensors


        Args:
            features (list[Any]): Should contain a 'query' and 'response' field.
            return_tensors (_type_, optional): supports right now only 'pt'

        Returns:
            dict[str, Any]: a dictionary of ~torch.Tensors with keys 'input_ids'
                and 'labels'

        Note that each ('query', 'response') pair generates rag_system.config.top_k
        fine-tuning instance for RALT.
        """
        return_tensors = (
            return_tensors if return_tensors else self.default_return_tensors
        )
        if return_tensors != "pt":
            raise DataCollatorError(
                f"Framework '{return_tensors}' not recognized!"
            )

        # STEP 1 — use rag system to build the RALT fine-tuning texts
        finetuning_instances = []
        inputs_list = []
        attention_mask_list = []
        max_length = 0
        for example in features:
            # retrieve
            source_nodes = self.rag_system.retrieve(query=example["query"])
            total_sum_scores = sum(s.score for s in source_nodes)

            # parallel in-context retrieval-augmentation creates
            # top_k separated finetuning instances
            for source in source_nodes:
                finetune_instance_text = self.example_template.format(
                    query=example["query"],
                    response=example["response"],
                    context=source.node.get_content()["text_content"],
                )
                finetuning_instances.append(finetune_instance_text)
                _weight = source.score / total_sum_scores

                # tokenize to get input_ids and target_ids
                tokenizer = self.rag_system.generator.tokenizer

                encode_result = tokenizer.encode(finetune_instance_text)
                input_ids = encode_result["input_ids"]
                attention_mask = encode_result["attention_mask"]

                current_input_len = len(input_ids)
                if current_input_len > max_length:
                    max_length = current_input_len

                inputs_list.append(input_ids)
                attention_mask_list.append(attention_mask)

        # padding — apply left padding
        padded_features = self._apply_padding(
            max_length=max_length,
            inputs_list=inputs_list,
            attention_mask_list=attention_mask_list,
            tokenizer=tokenizer.unwrapped,
        )

        return padded_features
