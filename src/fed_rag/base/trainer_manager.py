"""Base RAG Trainer Manager"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import assert_never

from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.exceptions import (
    InconsistentRAGSystems,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)

from .fl_task import BaseFLTask


class RAGTrainMode(str, Enum):
    """Modes for training RAG systems."""

    RETRIEVER = "retriever"
    GENERATOR = "generator"


class BaseRAGTrainerManager(BaseModel, ABC):
    """Manages and orchestrates RAG system training workflows.

    Coordinates training of retriever and generator components based on
    the specified training mode. Handles trainer selection and execution
    without maintaining RAG system state.

    Attributes:
        mode: The training mode specifying which component to train.
        retriever_trainer: Trainer for the retriever component, if applicable.
        generator_trainer: Trainer for the generator component, if applicable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    mode: RAGTrainMode
    retriever_trainer: BaseRetrieverTrainer | None = None
    generator_trainer: BaseGeneratorTrainer | None = None

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """Validate the supplied mode."""
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

    # Validate trainer presence
    @model_validator(mode="after")
    def validate_trainers(self) -> "BaseRAGTrainerManager":
        """Validate trainer requirements."""
        # Validate trainer presence based on mode
        if (
            self.mode == RAGTrainMode.RETRIEVER
            and self.retriever_trainer is None
        ):
            raise UnspecifiedRetrieverTrainer(
                "Retriever trainer must be set when in retriever mode"
            )
        if (
            self.mode == RAGTrainMode.GENERATOR
            and self.generator_trainer is None
        ):
            raise UnspecifiedGeneratorTrainer(
                "Generator trainer must be set when in generator mode"
            )

        return self

    @model_validator(mode="after")
    def validate_trainers_consistency(self) -> "BaseRAGTrainerManager":
        """Validate that trainers use consistent RAG systems if both are present."""
        if (
            self.retriever_trainer is not None
            and self.generator_trainer is not None
        ):
            # Check if both trainers have the same RAG system reference
            if id(self.retriever_trainer.rag_system) != id(
                self.generator_trainer.rag_system
            ):
                raise InconsistentRAGSystems(
                    "Inconsistent RAG systems detected between retriever and generator trainers. "
                    "Both trainers must use the same RAG system instance for consistent training."
                )

        return self

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

    @property
    def model(self) -> Any:
        """Return the model to be trained."""
        match self.mode:
            case RAGTrainMode.RETRIEVER:
                trainer = cast(BaseRetrieverTrainer, self.retriever_trainer)
            case RAGTrainMode.GENERATOR:
                trainer = cast(BaseGeneratorTrainer, self.generator_trainer)
            case _:  # pragma: no cover
                assert_never(self.mode)  # pragma: no cover
        return trainer.model
