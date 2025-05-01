"""Base Trainer"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult


class BaseTrainer(BaseModel, ABC):
    """Base Trainer Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rag_system: RAGSystem
    train_dataset: Any
    _model = PrivateAttr()

    @abstractmethod
    def train(self) -> TrainResult:
        """Train loop."""

    @abstractmethod
    def evaluate(self) -> TestResult:
        """Evaluation"""

    @abstractmethod
    def _get_model_from_rag_system(self) -> Any:
        """Get the model from the RAG system."""

    @model_validator(mode="after")
    def set_model(self) -> "BaseTrainer":
        self._model = self._get_model_from_rag_system()
        return self

    @property
    def model(self) -> Any:
        """Return the model to be trained."""
        return self._model

    @model.setter
    def model(self, v: Any) -> None:
        """Set the model to be trained."""
        self._model = v


class BaseRetrieverTrainer(BaseTrainer, ABC):
    """Base Retriever Trainer Class."""

    def _get_model_from_rag_system(self) -> Any:
        if self.rag_system.retriever.encoder:
            return self.rag_system.retriever.encoder
        else:
            return (
                self.rag_system.retriever.query_encoder
            )  # only update query encoder


class BaseGeneratorTrainer(BaseTrainer, ABC):
    """Base Retriever Trainer Class."""

    def _get_model_from_rag_system(self) -> Any:
        return self.rag_system.generator.model
