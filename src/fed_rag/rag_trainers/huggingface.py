"""HuggingFace RAG Trainer"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import model_validator

from fed_rag.base.rag_trainer import BaseRAGTrainer
from fed_rag.exceptions import MissingExtraError
from fed_rag.exceptions.core import FedRAGError
from fed_rag.types.rag_system import RAGSystem

try:
    from datasets import Dataset
    from transformers import TrainingArguments

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:
    from datasets import Dataset


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


# Define trainer function type hints
RetrieverTrainFn = Callable[[RAGSystem, Dataset, TrainingArguments], Any]
GeneratorTrainFn = Callable[[RAGSystem, Dataset, TrainingArguments], Any]


class HuggingFaceRAGTrainer(BaseRAGTrainer):
    """HuggingFace RAG Trainer"""

    train_dataset: Dataset
    retriever_training_args: TrainingArguments
    generator_training_args: TrainingArguments
    retriever_train_fn: Optional[RetrieverTrainFn] = None
    generator_train_fn: Optional[GeneratorTrainFn] = None

    def __init__(
        self,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

    @model_validator(mode="after")
    def validate_training_args(self) -> "HuggingFaceRAGTrainer":
        _validate_rag_system(self.rag_system)

        return self
