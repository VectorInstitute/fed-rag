"""Hugging Face Trainer for ReSearch"""

from typing import TYPE_CHECKING, Any, Optional

from pydantic import PrivateAttr, model_validator

from fed_rag.base.trainer import BaseGeneratorTrainer
from fed_rag.exceptions import MissingExtraError
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.types import RAGSystem, TestResult, TrainResult
from fed_rag.utils.huggingface import _validate_rag_system

try:
    from trl import GRPOConfig, GRPOTrainer

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False

    class GRPOTrainer:  # type: ignore[no-redef]
        """Dummy placeholder when transformers is not available."""

        pass


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer


def _get_default_training_args() -> "GRPOConfig":
    from trl import GRPOConfig

    return GRPOConfig()


class HuggingFaceTrainerForReSearch(
    HuggingFaceTrainerMixin, BaseGeneratorTrainer
):
    """Hugging Face Trainer for ReSearch.

    This class is an implementation of the ReSearch framework introduced in the
    paper:
        Chen, Mingyang, et al. "Research: Learning to reason with search for
        llms via reinforcement learning." arXiv preprint arXiv:2503.19470 (2025).

    In summary, the method aims to provide the LLM generator reasoning capabilities
    from scratch using reinforcement learning methods—i.e., GRPO. In addition,
    to text-based reasoning, ReSearch incorporates search/retrieval in the
    rollout.
    """

    _hf_trainer: Optional["GRPOTrainer"] = PrivateAttr(default=None)

    def __init__(
        self,
        rag_system: RAGSystem,
        train_dataset: "Dataset",
        training_arguments: Optional["GRPOConfig"] = None,
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
    def set_private_attributes(self) -> "HuggingFaceTrainerForReSearch":
        from trl import GRPOTrainer

        _validate_rag_system(self.rag_system)

        self._hf_trainer = GRPOTrainer(
            self.model,
            processing_class=self.rag_system.generator.tokenizer.unwrapped,
            reward_funcs=HuggingFaceTrainerForReSearch.rollout_reward_func,
        )

        return self

    def train(self) -> TrainResult:
        raise NotImplementedError

    def evaluate(self) -> TestResult:
        raise NotImplementedError

    @property
    def hf_trainer_obj(self) -> "GRPOTrainer":
        return self._hf_trainer

    @staticmethod
    def rollout_reward_func(
        completions: list[Any], **kwargs: Any
    ) -> list[float]:
        raise NotImplementedError
