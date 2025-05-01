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
