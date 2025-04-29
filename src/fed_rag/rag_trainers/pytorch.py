"""PyTorch RAG Trainer"""

from typing import Any, Callable, Optional, cast

import torch.nn as nn
from pydantic import BaseModel, Field, model_validator
from torch.utils.data import DataLoader
from typing_extensions import assert_never

from fed_rag.base.rag_trainer import BaseRAGTrainer
from fed_rag.decorators import federate
from fed_rag.exceptions.rag_trainer import (
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
)
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult


class TrainingArgs(BaseModel):
    """Arguments for training."""

    learning_rate: float | None = None
    batch_size: int | None = None
    num_epochs: int | None = None
    warmup_steps: int | None = None
    weight_decay: float | None = None
    custom_kwargs: dict[str, Any] = Field(default_factory=dict)


# Define trainer function type hints
RetrieverTrainFn = Callable[[RAGSystem, DataLoader, TrainingArgs], Any]
GeneratorTrainFn = Callable[[RAGSystem, DataLoader, TrainingArgs], Any]


class PyTorchRAGTrainer(BaseRAGTrainer):
    train_dataloader: DataLoader
    retriever_training_args: TrainingArgs = Field(
        default_factory=lambda: TrainingArgs()
    )
    generator_training_args: TrainingArgs = Field(
        default_factory=lambda: TrainingArgs()
    )
    retriever_train_fn: Optional[RetrieverTrainFn] = None
    generator_train_fn: Optional[GeneratorTrainFn] = None

    @model_validator(mode="after")
    def validate_training_args(self) -> "PyTorchRAGTrainer":
        # Convert dict args to Pydantic models if needed
        if isinstance(self.retriever_training_args, dict):
            self.retriever_training_args = TrainingArgs.model_validate(
                self.retriever_training_args
            )

        if isinstance(self.generator_training_args, dict):
            self.generator_training_args = TrainingArgs.model_validate(
                self.generator_training_args
            )

        return self

    def _prepare_generator_for_training(self, **kwargs: Any) -> None:
        self.rag_system.generator.model.train()

        # freeze retriever
        if self.rag_system.retriever.encoder:
            self.rag_system.retriever.encoder.eval()

        if self.rag_system.retriever.context_encoder:
            self.rag_system.retriever.context_encoder.eval()

        if self.rag_system.retriever.query_encoder:
            self.rag_system.retriever.query_encoder.eval()

    def _prepare_retriever_for_training(
        self, freeze_context_encoder: bool = True, **kwargs: Any
    ) -> None:
        if self.rag_system.retriever.encoder:
            self.rag_system.retriever.encoder.train()

        if self.rag_system.retriever.query_encoder:
            self.rag_system.retriever.query_encoder.train()

        if self.rag_system.retriever.context_encoder:
            if freeze_context_encoder:
                self.rag_system.retriever.context_encoder.eval()
            else:
                self.rag_system.retriever.context_encoder.train()

        # freeze generator
        self.rag_system.generator.model.eval()

    def _train_retriever(self, **kwargs: Any) -> None:
        self._prepare_retriever_for_training()
        if self.retriever_train_fn:
            self.retriever_train_fn(
                self.rag_system,
                self.train_dataloader,
                self.retriever_training_args,
            )
        else:
            raise UnspecifiedRetrieverTrainer(
                "Attempted to perform retriever trainer with an unspecified trainer function."
            )

    def _train_generator(self, **kwargs: Any) -> None:
        self._prepare_generator_for_training()
        if self.generator_train_fn:
            self.generator_train_fn(
                self.rag_system,
                self.train_dataloader,
                self.generator_training_args,
            )
        else:
            raise UnspecifiedGeneratorTrainer(
                "Attempted to perform generator trainer with an unspecified trainer function."
            )

    def train(self, **kwargs: Any) -> None:
        if self.mode == "retriever":
            self._train_retriever()
        elif self.mode == "generator":
            self._train_generator()
        else:
            assert_never(self.mode)  # pragma: no cover

    def _get_federated_trainer(self) -> tuple[Callable, nn.Module]:
        if self.mode == "retriever":
            if self.retriever_train_fn is None:
                raise UnspecifiedRetrieverTrainer(
                    "Cannot federate an unspecified retriever trainer function."
                )
            retriever_train_fn = self.retriever_train_fn

            if self.rag_system.retriever.encoder:
                retriever_module = self.rag_system.retriever.encoder
            else:
                retriever_module = self.rag_system.retriever.query_encoder
                retriever_module = cast(nn.Module, retriever_module)

            # Create a standalone function for federation
            def train_wrapper(
                _mdl: nn.Module,
                _train_dataloader: DataLoader,
                _val_dataloader: DataLoader,
            ) -> TrainResult:
                _ = retriever_train_fn(
                    self.rag_system,
                    self.train_dataloader,
                    self.retriever_training_args,
                )
                return TrainResult(loss=0)

            return federate.trainer.pytorch(train_wrapper), retriever_module

        elif self.mode == "generator":
            if self.generator_train_fn is None:
                raise UnspecifiedGeneratorTrainer(
                    "Cannot federate an unspecified generator trainer function."
                )
            generator_train_fn = self.generator_train_fn

            generator_module = self.rag_system.generator.model

            # Create a standalone function for federation
            def train_wrapper(
                _mdl: nn.Module,
                _train_dataloader: DataLoader,
                _val_dataloader: DataLoader,
            ) -> TrainResult:
                _ = generator_train_fn(
                    self.rag_system,
                    self.train_dataloader,
                    self.generator_training_args,
                )
                # TODO get loss from out
                return TrainResult(loss=0)

            return federate.trainer.pytorch(train_wrapper), generator_module
        else:
            assert_never(self.mode)  # pragma: no cover

    def get_federated_task(self) -> PyTorchFLTask:
        federated_trainer, _module = self._get_federated_trainer()

        # TODO: add logic for getting evaluator/tester and then federate it as well
        # federated_tester = self.get_federated_tester(tester_decorator)
        # For now, using a simple placeholder test function
        def test_fn(_mdl: nn.Module, _dataloader: DataLoader) -> TestResult:
            # Implement simple testing or return a placeholder
            return TestResult(loss=0.42, metrics={})  # pragma: no cover

        federated_tester = federate.tester.pytorch(test_fn)

        return PyTorchFLTask.from_trainer_and_tester(
            trainer=federated_trainer,
            tester=federated_tester,
        )
