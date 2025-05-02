"""HuggingFace Data Collator For LM-Supervised Retriever Training"""

from typing import Any

import torch
from pydantic import Field

from fed_rag.base.data_collator import BaseDataCollator
from fed_rag.exceptions import MissingExtraError
from fed_rag.exceptions.core import FedRAGError
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


DEFAULT_PROMPT_TEMPLATE = """
You are a helpful assistant. Given the user's question, provide a succinct
and accurate response. If context is provided, use it in your answer if it helps
you to create the most accurate response.

<question>
{query}
</question>

<context>
{context}
</context>

"""

DEFAULT_TARGET_TEMPLATE = """
<response>
{response}
</response>
"""


class DataCollatorForLSR(DataCollatorMixin, BaseDataCollator):
    """A HuggingFace DataCollator for LM-Supervised Retrieval."""

    prompt_template: str = Field(default=DEFAULT_PROMPT_TEMPLATE)
    target_template: str = Field(default=DEFAULT_TARGET_TEMPLATE)
    default_return_tensors: str = Field(default="pt")

    def __init__(
        self,
        rag_system: RAGSystem,
        prompt_template: str | None = None,
        target_template: str | None = None,
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

        prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        target_template = target_template or DEFAULT_TARGET_TEMPLATE

        super().__init__(
            rag_system=rag_system,
            prompt_template=prompt_template,
            target_template=target_template,
            default_return_tensors=default_return_tensors,
            **kwargs,
        )

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Any]:
        """Use the features of the dataset in order to get the retrieval and lm-scores.


        Args:
            features (list[Any]): Should contain a 'query' and 'reponse' field.
            return_tensors (_type_, optional): supports right now only 'pt'

        Returns:
            dict[str, Any]: a dictionary of ~torch.Tensors with keys 'retrieval_scores'
                and 'lm_scores'
            Note that each ('query', 'response') pair generates one fine-tuning instance for LSR.
        """
        return_tensors = (
            return_tensors if return_tensors else self.default_return_tensors
        )
        if return_tensors != "pt":
            raise FedRAGError(f"Framework '{return_tensors}' not recognized!")

        # use rag system to get scores
        batch_retriever_scores = []
        batch_lm_scores = []
        for example in features:
            query = example.get("query")
            response = example.get("response")

            # retriever scores
            source_nodes = self.rag_system.retrieve(query)
            retriever_scores = torch.tensor([n.score for n in source_nodes])

            # lm supervised scores
            lm_scores = []
            for chunk in source_nodes:
                prompt = self.prompt_template.format(
                    query=query,
                    context=chunk.node.get_content()["text_content"],
                )
                target = self.target_template.format(response=response)
                lm_score = (
                    self.rag_system.generator.compute_target_sequence_proba(
                        prompt=prompt, target=target
                    )
                )
                lm_scores.append(lm_score)
            lm_scores = torch.stack(lm_scores, dim=0)

            # append to batch
            batch_retriever_scores.append(retriever_scores)
            batch_lm_scores.append(lm_scores)

        # create torch.Tensors
        retrieval_scores = torch.stack(batch_retriever_scores, dim=0)
        lm_scores = torch.stack(batch_lm_scores, dim=0)

        return {"retrieval_scores": retrieval_scores, "lm_scores": lm_scores}
