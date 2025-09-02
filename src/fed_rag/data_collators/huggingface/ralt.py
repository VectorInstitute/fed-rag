"""HuggingFace Data Collator For Retrieval-Augmented Generator Training"""

from typing import TYPE_CHECKING, Any

import torch
from pydantic import Field

from fed_rag import NoEncodeRAGSystem, RAGSystem
from fed_rag.base.data_collator import BaseDataCollator
from fed_rag.exceptions import DataCollatorError, MissingExtraError
from fed_rag.utils.huggingface import _validate_rag_system

try:
    from transformers.data.data_collator import DataCollatorMixin

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    # Create a dummy class with a different name to avoid the redefinition
    class _DummyDataCollatorMixin:
        """Mixin providing placeholder data collation functionality.

        This class serves as a base mixin for implementing data collation methods. It is
        intended to be extended by other classes where data collation is required. This
        class does not implement any methods or attributes itself.
        """

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
    """
    Data collator class for Retrieval-Augmented Language Tuning (RALT).

    This class is responsible for processing dataset features to create proper
    inputs and labels for the fine-tuning of a retrieval-augmented language model.
    It uses an example template and a RAG (Retrieval-Augmented Generation) system
    to build and encode fine-tuning instances, and applies padding to align the
    training data.

    Attributes:
        example_template(str): A string template used to format fine-tuning instances.
        default_return_tensors(str): The default framework type for returned tensors ('pt').
        model_dtype(torch.dtype|None): The model's data type (e.g., torch.float32). Initialized from
            the generator model in the RAG system, if available.
        rag_system(RAGSystem|NoEncodeRAGSystem): An instance of a RAG system supporting the retrieval and in-context
            augmentation for fine-tuning.
    """

    example_template: str = Field(default=DEFAULT_EXAMPLE_TEMPLATE)
    default_return_tensors: str = Field(default="pt")
    model_dtype: torch.dtype | None = None
    rag_system: RAGSystem | NoEncodeRAGSystem

    def __init__(
        self,
        rag_system: RAGSystem | NoEncodeRAGSystem,
        example_template: str | None = None,
        default_return_tensors: str = "pt",
        **kwargs: Any,
    ):
        """
        Initialize an instance with a RAG system, example template, and optional parameters.

        Validates the RAG system and ensures required dependencies are installed.

        Args:
            rag_system (RAGSystem | NoEncodeRAGSystem): The RAG system to be used, which can
                be either a standard RAGSystem or a NoEncodeRAGSystem.
            example_template (str | None, optional): Template for examples. Defaults to None;
                if not specified, a predefined template is used.
            default_return_tensors (str, optional): Default tensor format (e.g., "pt" for PyTorch).
                Defaults to "pt".
            **kwargs (Any): Additional keyword arguments passed to the superclass or used
                during initialization.

        Raises:
            MissingExtraError: If required Hugging Face dependencies are missing.
        """
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        _validate_rag_system(rag_system)

        example_template = example_template or DEFAULT_EXAMPLE_TEMPLATE
        # get generator model type
        try:
            model_dtype = rag_system.generator.model.dtype
        except AttributeError:
            model_dtype = torch.float32  # fallback

        super().__init__(
            rag_system=rag_system,
            example_template=example_template,
            default_return_tensors=default_return_tensors,
            model_dtype=model_dtype,
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
            "input_ids": torch.stack(padded_input_ids).long(),
            "attention_mask": torch.stack(padded_attention_mask).to(
                self.model_dtype
            ),
            "labels": torch.stack(padded_labels).long(),
        }

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Any]:
        """Prepare input tensors for fine-tuning using RAG system features.

        Converts a list of features into input tensors suitable for retrieval-augmented
        language model (RALT) fine-tuning. Each ('query', 'response') pair generates
        `rag_system.config.top_k` fine-tuning instances.

        Args:
            features (list[dict[str, Any]]): List of examples, each containing
                'query' and 'response' fields.
            return_tensors (str | None, optional): Tensor framework to use. Only 'pt'
                is supported. Defaults to None (uses `self.default_return_tensors`).

        Returns:
            dict[str, Any]: Dictionary of PyTorch tensors with keys:
                - 'input_ids': Token IDs.
                - 'labels': Target IDs.

        Raises:
            DataCollatorError: If `return_tensors` is not 'pt'.

        Note:
            Applies left-padding to all sequences to ensure uniform length.
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
