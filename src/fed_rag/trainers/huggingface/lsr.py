"""HuggingFace LM-Supervised Retriever Trainer"""

from typing import TYPE_CHECKING, Any

from pydantic import field_validator

from fed_rag.base.trainer import BaseTrainer
from fed_rag.exceptions import TrainerError
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin

if TYPE_CHECKING:  # pragma: no-cover
    from sentence_transformers import SentenceTransformer


class HuggingFaceLSRTrainer(HuggingFaceTrainerMixin, BaseTrainer):
    """HuggingFace LM-Supervised Retriever Trainer."""

    @field_validator("model", mode="before")
    @classmethod
    def validate_mode(cls, v: Any) -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        if not isinstance(v, SentenceTransformer):
            raise TrainerError(
                "For `HuggingFaceLSRTrainer`, attribute `model` must be of type "
                "`~sentence_transformers.SentenceTransformer`."
            )
        return v

    def train(self) -> None:
        pass

    def evaluate(self) -> None:
        pass
