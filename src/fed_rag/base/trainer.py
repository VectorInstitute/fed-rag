"""Base trainer classes for RAG system components."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from fed_rag import NoEncodeRAGSystem, RAGSystem
from fed_rag.data_structures.results import TestResult, TrainResult


class BaseTrainer(BaseModel, ABC):
    """Base Trainer Class.

    This abstract class provides the interface for creating Trainer objects that
    implement different training strategies.

    Attributes:
        rag_system: The RAG system to be trained.
        train_dataset: Dataset used for training.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    rag_system: RAGSystem
    train_dataset: Any
    _model = PrivateAttr()

    @abstractmethod
    def train(self) -> TrainResult:
        """Trains the model.

        Returns:
            TrainResult: The result of model training.
        """

    @abstractmethod
    def evaluate(self) -> TestResult:
        """Evaluates the model.

        Returns:
            TestResult: The result of model evaluation.
        """

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
    """Base trainer for retriever components of RAG systems.

    This trainer focuses specifically on training the retriever's encoder
    components, either the full encoder or just the query encoder depending
    on the retriever configuration.
    """

    def _get_model_from_rag_system(self) -> Any:
        if self.rag_system.retriever.encoder:
            return self.rag_system.retriever.encoder
        else:
            return (
                self.rag_system.retriever.query_encoder
            )  # only update query encoder


class BaseGeneratorTrainer(BaseTrainer, ABC):
    """Base trainer for generator component of RAG systems.

    This trainer focuses specifically on training the generator model.

    Attributes:
        rag_system: The RAG system to be trained. Can also be a `NoEncodeRAGSytem`.
    """

    rag_system: RAGSystem | NoEncodeRAGSystem

    def _get_model_from_rag_system(self) -> Any:
        return self.rag_system.generator.model
