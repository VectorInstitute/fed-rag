"""HuggingFace Retrieval-Augmented Generator Trainer"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import PrivateAttr, model_validator

from fed_rag.base.trainer import BaseGeneratorTrainer
from fed_rag.data_collators.huggingface.ralt import DataCollatorForRALT
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult
from fed_rag.utils.huggingface import _validate_rag_system

try:
    from transformers import Trainer

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    class Trainer:  # type: ignore[no-redef]
        """Dummy placeholder when transformers is not available."""

        pass


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_utils import TrainOutput


class HuggingFaceTrainerForRALT(HuggingFaceTrainerMixin, BaseGeneratorTrainer):
    """HuggingFace Trainer for Retrieval-Augmented LM Training/Fine-Tuning."""

    _hf_trainer: Optional["Trainer"] = PrivateAttr(default=None)

    def __init__(
        self,
        rag_system: RAGSystem,
        train_dataset: "Dataset",
        training_arguments: Optional["TrainingArguments"] = None,
        **kwargs: Any,
    ):
        super().__init__(
            train_dataset=train_dataset,
            rag_system=rag_system,
            training_arguments=training_arguments,
            **kwargs,
        )

    @model_validator(mode="after")
    def set_private_attributes(self) -> "HuggingFaceTrainerForRALT":
        # if made it to here, then this import is available
        from transformers import Trainer

        # validate rag system
        _validate_rag_system(self.rag_system)

        self._hf_trainer = Trainer(
            model=self.model,
            args=self.training_arguments,
            data_collator=DataCollatorForRALT(rag_system=self.rag_system),
        )

        return self

    def train(self) -> TrainResult:
        output: TrainOutput = self.hf_trainer_obj.train()
        return TrainResult(loss=output.training_loss)

    def evaluate(self) -> TestResult:
        # TODO: implement this
        raise NotImplementedError

    @property
    def hf_trainer_obj(self) -> "Trainer":
        return self._hf_trainer
