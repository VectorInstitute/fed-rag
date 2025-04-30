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

from pydantic import BaseModel, ConfigDict, model_validator

from fed_rag.exceptions import MissingExtraError
from fed_rag.utils.huggingface import _validate_rag_system

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
    model: Union["SentenceTransformer", "PreTrainedModel", "PeftModel"]
    train_dataset: "Dataset"
    training_arguments: Optional["TrainingArguments"]


class HuggingFaceTrainerMixin(BaseModel, ABC):
    """HuggingFace Trainer Mixin."""

    model: Union["SentenceTransformer", "PreTrainedModel", "PeftModel"]
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    train_dataset: "Dataset"
    training_arguments: Optional["TrainingArguments"] = None

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        if not _has_huggingface:
            msg = (
                f"`{self.__class__.__name__}` requires `huggingface` extra to be installed. "
                "To fix please run `pip install fed-rag[huggingface]`."
            )
            raise MissingExtraError(msg)
        super().__init__(*args, **kwargs)

    @model_validator(mode="after")
    def validate_training_args(self) -> "HuggingFaceTrainerMixin":
        if hasattr(self, "rag_system"):
            _validate_rag_system(self.rag_system)

        return self

    @property
    @abstractmethod
    def hf_trainer_obj(self) -> "Trainer":
        """A ~transformers.Trainer object."""
