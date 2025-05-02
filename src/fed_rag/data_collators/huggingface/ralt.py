"""HuggingFace Data Collator For Retrieval-Augmented Generator Training"""

from typing import Any, cast

from pydantic import Field

from fed_rag.base.data_collator import BaseDataCollator
from fed_rag.exceptions import DataCollatorError, MissingExtraError
from fed_rag.types.rag_system import RAGSystem
from fed_rag.utils.huggingface import _validate_rag_system

try:
    import transformers.data.data_collator as transformers_data_collators
    from transformers.data.data_collator import (
        DataCollatorForLanguageModeling,
        DataCollatorMixin,
    )

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    # Create a dummy class with a different name to avoid the redefinition
    class _DummyDataCollatorMixin:
        """Dummy placeholder when transformers is not available."""

        pass

    class DataCollatorForLanguageModeling:  # type: ignore[no-redef]
        """Dummy placeholder when transformers is not available."""

        pass

    DataCollatorMixin = _DummyDataCollatorMixin  # type: ignore


DEFAULT_EXAMPLE_TEMPLATE = """
You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

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

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Any]:
        """Use the features of the dataset in order to get the `input_ids` and `labels`.

        Steps:
            1. process the features using the RAG system and example template to create
               the retrieval-augmented lm fine-tuning text
            2. pass this processed features to ~transformers.DataCollatorForLanguageModeling


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
        inputs_list = []
        targets_list = []
        attention_mask_list = []
        finetuning_instances = []
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
                unwrapped_tokenizer = (
                    self.rag_system.generator.tokenizer.unwrapped
                )
                try:
                    eos_token = unwrapped_tokenizer.special_tokens_map.get(
                        "eos_token"
                    )
                except KeyError:
                    raise DataCollatorError(
                        "Tokenizer doesn't have an `eos_token`."
                    )
                eos_token_ix = unwrapped_tokenizer.all_special_tokens.index(
                    eos_token
                )
                eos_token_id = unwrapped_tokenizer.all_special_ids[
                    eos_token_ix
                ]

                encode_result = tokenizer.encode(finetune_instance_text)
                input_ids = encode_result["input_ids"]
                attention_mask = encode_result["attention_mask"]
                target_ids = input_ids[1:] + [eos_token_id]

                inputs_list.append(input_ids)
                targets_list.append(target_ids)
                attention_mask_list.append(attention_mask)

        processed_features = {
            "input_ids": inputs_list,
            "attention_mask": attention_mask_list,
            "target_ids": targets_list,
        }

        # STEP 2 — Use ~transformers.DataCollatorForLanguageModeling
        data_collator_for_lm = DataCollatorForLanguageModeling(
            tokenizer=unwrapped_tokenizer,
            mlm=False,  # we could implement masking here on instructions
        )
        # bring back proper typing
        data_collator_for_lm = cast(
            transformers_data_collators.DataCollatorForLanguageModeling,
            data_collator_for_lm,
        )

        return data_collator_for_lm.torch_call(processed_features)  # type: ignore[no-any-return]
