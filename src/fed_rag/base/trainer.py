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

    @abstractmethod
    def train(self) -> TrainResult:
        """Train loop."""

    @abstractmethod
    def evaluate(self) -> TestResult:
        """Evaluation"""

    @property
    @abstractmethod
    def model(self) -> Any:
        """Return the model to be trained.

        NOTE: this should be a component of the RAG System.
        """

    @model.setter
    @abstractmethod
    def model(self, v: Any) -> None:
        """Set the model to be trained."""
        raise NotImplementedError("Setting model not supported by default")


class BaseRetrieverTrainer(BaseTrainer, ABC):
    """Base Retriever Trainer Class."""

    _model = PrivateAttr()

    @model_validator(mode="after")
    def set_model(self) -> "BaseRetrieverTrainer":
        if self.rag_system.retriever.encoder:
            self._model = self.rag_system.retriever.encoder
        else:
            self._model = (
                self.rag_system.retriever.query_encoder
            )  # only update query encoder

        return self

    @property
    def model(self) -> Any:
        """Return the model to be trained."""
        return self._model

    @model.setter
    def model(self, v: Any) -> None:
        self._model = v


class BaseGeneratorTrainer(BaseTrainer, ABC):
    """Base Retriever Trainer Class."""

    _model = PrivateAttr()

    @model_validator(mode="after")
    def set_model(self) -> "BaseGeneratorTrainer":
        self._model = self.rag_system.generator.model
        return self

    @property
    def model(self) -> Any:
        """Return the model to be trained."""
        return self._model

    @model.setter
    def model(self, v: Any) -> None:
        self._model = v
