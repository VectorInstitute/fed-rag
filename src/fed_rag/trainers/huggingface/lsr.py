"""HuggingFace LM-Supervised Retriever Trainer"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from pydantic import PrivateAttr, model_validator

from fed_rag.base.trainer import BaseRetrieverTrainer
from fed_rag.data_collators.huggingface import DataCollatorForLSR
from fed_rag.exceptions import (
    InvalidDataCollatorError,
    InvalidLossError,
    MissingExtraError,
    MissingInputTensor,
    TrainerError,
)
from fed_rag.loss.pytorch.lsr import LSRLoss
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult
from fed_rag.utils.huggingface import _validate_rag_system

try:
    from sentence_transformers import SentenceTransformerTrainer

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    class SentenceTransformerTrainer:  # type: ignore[no-redef]
        """Dummy placeholder when sentence transformers is not available."""

        pass


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
    )
    from transformers import TrainingArguments
    from transformers.trainer_utils import TrainOutput


class LSRSentenceTransformerTrainer(SentenceTransformerTrainer):
    def __init__(
        self,
        *args: Any,
        data_collator: DataCollatorForLSR,
        loss: Optional[LSRLoss] = None,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)

        # set loss
        if loss is None:
            loss = LSRLoss()
        else:
            if not isinstance(loss, LSRLoss):
                raise InvalidLossError(
                    "`LSRSentenceTransformerTrainer` must use ~fed_rag.loss.LSRLoss`."
                )

        if not isinstance(data_collator, DataCollatorForLSR):
            raise InvalidDataCollatorError(
                "`LSRSentenceTransformerTrainer` must use ~fed_rag.data_collators.DataCollatorForLSR`."
            )

        super().__init__(
            *args, loss=loss, data_collator=data_collator, **kwargs
        )

    def collect_scores(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "retrieval_scores" not in inputs:
            raise MissingInputTensor(
                "Collated `inputs` are missing key `retrieval_scores`"
            )

        if "lm_scores" not in inputs:
            raise MissingInputTensor(
                "Collated `inputs` are missing key `lm_scores`"
            )

        retrieval_scores = inputs.get("retrieval_scores")
        lm_scores = inputs.get("lm_scores")

        return retrieval_scores, lm_scores

    def compute_loss(
        self,
        model: "SentenceTransformer",
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute LSR loss.

        NOTE: the forward pass of the model is taken care of in the DataCollatorForLSR.

        Args:
            model (SentenceTransformer): _description_
            inputs (dict[str, torch.Tensor  |  Any]): _description_
            return_outputs (bool, optional): _description_. Defaults to False.
            num_items_in_batch (Any | None, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, Any]]: _description_
        """
        retrieval_scores, lm_scores = self.collect_scores(inputs)
        loss = self.loss(retrieval_scores, lm_scores)

        # inputs are actually the outputs of RAGSystem's "forward" pass
        return (loss, inputs) if return_outputs else loss


class HuggingFaceTrainerForLSR(HuggingFaceTrainerMixin, BaseRetrieverTrainer):
    """HuggingFace LM-Supervised Retriever Trainer."""

    _hf_trainer: Optional["SentenceTransformerTrainer"] = PrivateAttr(
        default=None
    )

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
    def set_private_attributes(self) -> "HuggingFaceTrainerForLSR":
        # if made it to here, then this import is available
        from sentence_transformers import SentenceTransformer

        # validate rag system
        _validate_rag_system(self.rag_system)

        # validate model
        if not isinstance(self.model, SentenceTransformer):
            raise TrainerError(
                "For `HuggingFaceTrainerForLSR`, attribute `model` must be of type "
                "`~sentence_transformers.SentenceTransformer`."
            )

        self._hf_trainer = LSRSentenceTransformerTrainer(
            model=self.model,
            args=self.training_arguments,
            data_collator=DataCollatorForLSR(rag_system=self.rag_system),
        )

        return self

    def train(self) -> TrainResult:
        output: TrainOutput = self.hf_trainer_obj.train()
        return TrainResult(loss=output.training_loss)

    def evaluate(self) -> TestResult:
        # TODO: implement this
        raise NotImplementedError

    @property
    def hf_trainer_obj(self) -> "SentenceTransformerTrainer":
        return self._hf_trainer
