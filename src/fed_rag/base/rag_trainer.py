"""Base RAG Trainer"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from fed_rag.types.rag_system import RAGSystem


class RAGTrainMode(str, Enum):
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    INTERLEAVED = "interleaved"


class BaseRAGTrainer(ABC):
    """Base RAG Trainer Class."""

    def __init__(
        self, rag_system: RAGSystem, mode: RAGTrainMode, **kwargs: Any
    ):
        self.rag_system = rag_system
        self.mode = mode

    @abstractmethod
    def _prepare_retriever_for_training(
        self, decoder: bool = False, **kwargs: Any
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
