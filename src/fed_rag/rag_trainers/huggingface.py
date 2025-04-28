"""HuggingFace RAG Trainer"""

from typing import TYPE_CHECKING, Any, Optional

from fed_rag.base.rag_trainer import BaseRAGTrainer, RAGTrainMode
from fed_rag.exceptions import MissingExtraError
from fed_rag.exceptions.core import FedRAGError
from fed_rag.types.rag_system import RAGSystem

try:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

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


class HFRAGTrainer(BaseRAGTrainer):
    """HuggingFace RAG Trainer"""

    def __init__(
        self,
        rag_system: RAGSystem,
        mode: RAGTrainMode,
        train_dataset: "Dataset",
        retriever_trainer: Optional["Trainer"] = None,
        retriever_training_args: Optional["TrainingArguments"] = None,
        generator_trainer: Optional["Trainer"] = None,
        generator_training_args: Optional["TrainingArguments"] = None,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        Trainer.__init__(self, **kwargs)  # Pass appropriate params to Trainer
        BaseRAGTrainer.__init__(
            self, rag_system=rag_system, mode=mode, **kwargs
        )

        _validate_rag_system(rag_system)

        # Custom training functions
        self.retriever_training_args = retriever_training_args
        self.retriever_trainer = retriever_trainer
        self.generator_training_args = generator_training_args
        self.generator_trainer = generator_trainer
        self.train_dataset = train_dataset
