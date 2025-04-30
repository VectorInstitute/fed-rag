"""HuggingFace LM-Supervised Retriever Trainer"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import PrivateAttr, field_validator, model_validator

from fed_rag.base.trainer import BaseTrainer
from fed_rag.exceptions import MissingExtraError, TrainerError
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult

try:
    from sentence_transformers import SentenceTransformerTrainer

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
    )
    from transformers import TrainingArguments
    from transformers.trainer_utils import TrainOutput


class LSRSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(self, *args: Any, **kwargs: Any):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: "SentenceTransformer",
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        return super().compute_loss(
            model, inputs, return_outputs, num_items_in_batch
        )


class HuggingFaceLSRTrainer(HuggingFaceTrainerMixin, BaseTrainer):
    """HuggingFace LM-Supervised Retriever Trainer."""

    _hf_trainer: Optional["SentenceTransformerTrainer"] = PrivateAttr(
        default=None
    )

    def __init__(
        self,
        rag_system: RAGSystem,
        model: "SentenceTransformer",
        train_dataset: "Dataset",
        training_arguments: Optional["TrainingArguments"] = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            rag_system=rag_system,
            training_arguments=training_arguments,
            **kwargs,
        )

    @model_validator(mode="after")
    def set_hf_trainer(self) -> "HuggingFaceLSRTrainer":
        self._hf_trainer = LSRSentenceTransformerTrainer(
            self.model, args=self.training_arguments
        )

        return self

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

    def train(self) -> TrainResult:
        output: TrainOutput = self.hf_trainer_obj.train()
        return TrainResult(loss=output.training_loss)

    def evaluate(self) -> TestResult:
        # TODO: implement this
        raise NotImplementedError

    @property
    def hf_trainer_obj(self) -> "SentenceTransformerTrainer":
        return self._hf_trainer
