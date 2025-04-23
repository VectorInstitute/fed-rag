"""HuggingFace PeftModel Generator"""

from typing import TYPE_CHECKING, Any

from fed_rag.generators.hf_peft_model import HFPeftModelGenerator
from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator
from fed_rag.retrievers.hf_sentence_transformer import (
    HFSentenceTransformerRetriever,
)
from fed_rag.types.rag_system import RAGSystem

try:
    from transformers.data.data_collator import DataCollatorMixin

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from transformers.data.data_collator import DataCollatorMixin


class DataCollatorForLSR(DataCollatorMixin):
    """A HuggingFace DataCollator for LM-Supervised Retrieval."""

    def __init__(self, rag_system: RAGSystem):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise ValueError(msg)

        if not isinstance(
            rag_system.generator, HFPretrainedModelGenerator
        ) and isinstance(rag_system.generator, HFPeftModelGenerator):
            raise ValueError(
                "Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator."
            )

        if not isinstance(
            rag_system.retriever, HFSentenceTransformerRetriever
        ):
            raise ValueError(
                "Retriever must be a HFSentenceTransformerRetriever."
            )

        super().__init__()
        self.default_return_tensors = "pt"
        self.rag_system = rag_system

    def __call__(
        self, features: list[dict[str, Any]], return_tensors: str | None = None
    ) -> dict[str, Any]:
        """Use the features of the dataset in order to get the retrieval and lm-scores.


        Args:
            features (list[Any]): Should contain a 'query' and 'reponse' field.
            return_tensors (_type_, optional): supports right now only 'pt'

        Returns:
            dict[str, Any]: a dictionary of ~torch.Tensors with keys 'retrieval_scores'
                and 'lm-scores'
            Note that each ('query', 'response') pair generates one fine-tuning instance for LSR.
        """
        return_tensors = (
            return_tensors if return_tensors else self.default_return_tensors
        )
        if return_tensors != "pt":
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

        # use rag system to get scores
        for example in features:
            query = example.get("query")
            _response = example.get("response")

            # retriever scores
            source_nodes = self.rag_system.retrieve(query)

            # lm supervised scores
            for chunk in source_nodes:
                # 1. setup sequence kv cache with query + chunk context

                # 2. get probability of response sequence
                ...

        return {}
