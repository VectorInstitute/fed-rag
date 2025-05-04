"""HuggingFace Retrieval-Augmented Generator Trainer"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import PrivateAttr, model_validator

from fed_rag.base.trainer import BaseGeneratorTrainer
from fed_rag.data_collators.huggingface.ralt import DataCollatorForRALT
from fed_rag.exceptions import MissingExtraError
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


def _get_default_training_args() -> "TrainingArguments":
    from transformers import TrainingArguments

    return TrainingArguments(remove_unused_columns=False)


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
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        if training_arguments is None:
            training_arguments = _get_default_training_args()
        else:
            training_arguments.remove_unused_columns = (
                False  # pragma: no cover
            )

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
            train_dataset=self.train_dataset,
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
