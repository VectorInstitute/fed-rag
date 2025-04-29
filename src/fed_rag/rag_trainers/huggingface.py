"""HuggingFace RAG Trainer"""

from typing import TYPE_CHECKING, Any, Callable, Optional, cast

from pydantic import model_validator
from typing_extensions import assert_never

from fed_rag.base.rag_trainer import BaseRAGTrainer
from fed_rag.decorators import federate
from fed_rag.exceptions import (
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
)
from fed_rag.exceptions.core import FedRAGError
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult

try:
    from datasets import Dataset
    from transformers import TrainingArguments

    _has_huggingface = True
except ModuleNotFoundError:
    _has_huggingface = False


if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from transformers import TrainingArguments

    from fed_rag.fl_tasks.huggingface import HFModelType, HuggingFaceFLTask


def _validate_rag_system(rag_system: RAGSystem) -> None:
    # Skip validation if environment variable is set
    import os

    if os.environ.get("FEDRAG_SKIP_VALIDATION") == "1":
        return

    from fed_rag.generators.huggingface import (
        HFPeftModelGenerator,
        HFPretrainedModelGenerator,
    )
    from fed_rag.retrievers.huggingface.hf_sentence_transformer import (
        HFSentenceTransformerRetriever,
    )

    if not isinstance(
        rag_system.generator, HFPretrainedModelGenerator
    ) and not isinstance(rag_system.generator, HFPeftModelGenerator):
        raise FedRAGError(
            "Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator"
        )

    if not isinstance(rag_system.retriever, HFSentenceTransformerRetriever):
        raise FedRAGError("Retriever must be a HFSentenceTransformerRetriever")


# Define trainer function type hints
RetrieverTrainFn = Callable[
    [RAGSystem, "Dataset", Optional["TrainingArguments"]], Any
]
GeneratorTrainFn = Callable[
    [RAGSystem, "Dataset", Optional["TrainingArguments"]], Any
]


class HuggingFaceRAGTrainer(BaseRAGTrainer):
    """HuggingFace RAG Trainer"""

    train_dataset: "Dataset"
    retriever_training_args: Optional["TrainingArguments"] = None
    generator_training_args: Optional["TrainingArguments"] = None
    retriever_train_fn: Optional[RetrieverTrainFn] = None
    generator_train_fn: Optional[GeneratorTrainFn] = None

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
    def validate_training_args(self) -> "HuggingFaceRAGTrainer":
        _validate_rag_system(self.rag_system)

        return self

    def _prepare_generator_for_training(self, **kwargs: Any) -> None:
        pass  # no-op

    def _prepare_retriever_for_training(
        self, freeze_context_encoder: bool = True, **kwargs: Any
    ) -> None:
        pass  # no-op

    def _train_retriever(self, **kwargs: Any) -> None:
        self._prepare_retriever_for_training()
        if self.retriever_train_fn:
            self.retriever_train_fn(
                self.rag_system,
                self.train_dataset,
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
                self.train_dataset,
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

    def _get_federated_trainer(self) -> tuple[Callable, "HFModelType"]:
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
            retriever_module = cast("SentenceTransformer", retriever_module)

            # Create a standalone function for federation
            def train_wrapper(
                _mdl: HFModelType,
                _train_dataset: Dataset,
                _val_dataloader: Dataset,
            ) -> TrainResult:
                _ = retriever_train_fn(
                    self.rag_system,
                    self.train_dataset,
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
                _mdl: HFModelType,
                _train_dataset: Dataset,
                _val_dataloader: Dataset,
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

    def get_federated_task(self) -> "HuggingFaceFLTask":
        from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask

        federated_trainer, _module = self._get_federated_trainer()

        # TODO: add logic for getting evaluator/tester and then federate it as well
        # federated_tester = self.get_federated_tester(tester_decorator)
        # For now, using a simple placeholder test function
        def test_fn(_mdl: "HFModelType", _dataset: Dataset) -> TestResult:
            # Implement simple testing or return a placeholder
            return TestResult(loss=0.42, metrics={})  # pragma: no cover

        federated_tester = federate.tester.pytorch(test_fn)

        return HuggingFaceFLTask.from_trainer_and_tester(
            trainer=federated_trainer,
            tester=federated_tester,
        )
