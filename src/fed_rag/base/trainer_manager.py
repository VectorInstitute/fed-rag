"""Base RAG Trainer Manager"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from fed_rag.exceptions import UnsupportedTrainerMode
from fed_rag.types.rag_system import RAGSystem

from .fl_task import BaseFLTask


class RAGTrainMode(str, Enum):
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    INTERLEAVED = "interleaved"


class BaseRAGTrainerManager(BaseModel, ABC):
    """Base RAG Trainer Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rag_system: RAGSystem
    mode: RAGTrainMode

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        try:
            # Try to convert to enum
            mode = RAGTrainMode(v)
            return mode
        except ValueError:
            # Catch the ValueError from enum conversion and raise your custom error
            raise UnsupportedTrainerMode(
                f"Unsupported RAG train mode: {v}. "
                f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
            )

    @abstractmethod
    def _prepare_retriever_for_training(
        self, freeze_context_encoder: bool = True, **kwargs: Any
    ) -> None:
        """Prepare retriever model for training."""

    @abstractmethod
    def _prepare_generator_for_training(self, **kwargs: Any) -> None:
        """Prepare generator model for training."""

    @abstractmethod
    def _train_retriever(self, **kwargs: Any) -> Any:
        """Train loop for retriever."""

    @abstractmethod
    def _train_generator(self, **kwargs: Any) -> Any:
        """Train loop for generator."""

    @abstractmethod
    def train(self, **kwargs: Any) -> Any:
        """Train loop for rag system."""

    @abstractmethod
    def get_federated_task(self) -> BaseFLTask:
        """Get the federated task."""
