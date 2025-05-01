"""Base Trainer"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

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


class BaseRetrieverTrainer(BaseTrainer, ABC):
    """Base Retriever Trainer Class."""

    @property
    def model(self) -> Any:
        """Return the model to be trained."""
        if self.rag_system.retriever.encoder:
            return self.rag_system.retriever.encoder
        else:
            return (
                self.rag_system.retriever.query_encoder
            )  # only update query encoder


class BaseGeneratorTrainer(BaseTrainer, ABC):
    """Base Retriever Trainer Class."""

    @property
    def model(self) -> Any:
        """Return the model to be trained."""
        return self.rag_system.generator.model
