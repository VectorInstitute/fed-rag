"""HuggingFace PeftModel Generator"""

from typing import Any

import torch

from fed_rag.exceptions import MissingExtraError
from fed_rag.exceptions.core import FedRAGError
from fed_rag.types.rag_system import RAGSystem

try:
    from transformers.data.data_collator import DataCollatorMixin

    _has_huggingface = True
except (ModuleNotFoundError, MissingExtraError):
    _has_huggingface = False

    # Create a dummy class with a different name to avoid the redefinition
    class _DummyDataCollatorMixin:
        """Dummy placeholder when transformers is not available."""

        pass

    DataCollatorMixin = _DummyDataCollatorMixin  # type: ignore


def _validate_rag_system(rag_system: RAGSystem) -> None:
    # Skip validation if environment variable is set
    import os

    if os.environ.get("FEDRAG_SKIP_VALIDATION") == "1":
        return

    from fed_rag.generators.huggingface import (
        HFPeftModelGenerator,
        HFPretrainedModelGenerator,
    )
    from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
        HFSentenceTransformerRetriever,
    )

    if not isinstance(
        rag_system.generator, HFPretrainedModelGenerator
    ) and not isinstance(rag_system.generator, HFPeftModelGenerator):
        raise FedRAGError(
            "Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator"
        )

    if not isinstance(rag_system.retriever, HFSentenceTransformerRetriever):
        raise FedRAGError("Retriever must be a HFSentenceTransformerRetriever")


class DataCollatorForLSR(DataCollatorMixin):
    """A HuggingFace DataCollator for LM-Supervised Retrieval."""

    def __init__(self, rag_system: RAGSystem, prompt_template: str):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        _validate_rag_system(rag_system)

        super().__init__()
        self.default_return_tensors = "pt"
        self.rag_system = rag_system
        self.prompt_template = prompt_template

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
                target = f"\n{response}\n</response>"
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
