"""HuggingFace Trainer Mixin"""

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict

from fed_rag.exceptions import MissingExtraError

try:
    from datasets import Dataset
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedModel, Trainer, TrainingArguments

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from transformers import TrainingArguments


# Define the protocol for runtime checking
@runtime_checkable
class HuggingFaceTrainerProtocol(Protocol):
    train_dataset: "Dataset"
    training_arguments: Optional["TrainingArguments"]

    def model(
        self,
    ) -> Union["SentenceTransformer", "PreTrainedModel", "PeftModel"]:
        pass  # pragma: no cover


class HuggingFaceTrainerMixin(BaseModel, ABC):
    """HuggingFace Trainer Mixin."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    train_dataset: "Dataset"
    training_arguments: Optional["TrainingArguments"] = None

    def __init__(
        self,
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
        super().__init__(
            train_dataset=train_dataset,
            training_arguments=training_arguments,
            **kwargs,
        )

    @property
    @abstractmethod
    def hf_trainer_obj(self) -> "Trainer":
        """A ~transformers.Trainer object."""
