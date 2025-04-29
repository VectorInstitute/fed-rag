import re
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch
from transformers import TrainingArguments

from fed_rag.base.rag_trainer import BaseRAGTrainer, RAGTrainMode
from fed_rag.base.retriever import BaseRetriever
from fed_rag.exceptions import (
    FedRAGError,
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)
from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask
from fed_rag.generators.huggingface import HFPeftModelGenerator
from fed_rag.rag_trainers.huggingface import (
    GeneratorTrainFn,
    HuggingFaceRAGTrainer,
    RetrieverTrainFn,
)
from fed_rag.types.rag_system import RAGSystem


# fixtures
@pytest.fixture
def train_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "query": ["first query", "second query"],
            "response": ["first response", "second response"],
        }
    )


@pytest.fixture
def retriever_training_fn() -> RetrieverTrainFn:
    def fn(
        rag_system: RAGSystem,
        train_dataset: Dataset,
        trainer_args: TrainingArguments,
    ) -> Any:
        return {"retriever_loss": 0.42}

    return fn  # type: ignore


@pytest.fixture
def generator_training_fn() -> GeneratorTrainFn:
    def fn(
        rag_system: RAGSystem,
        train_dataset: Dataset,
        trainer_args: TrainingArguments,
    ) -> Any:
        return {"generator_loss": 0.42}

    return fn  # type: ignore


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [b.__name__ for b in HuggingFaceRAGTrainer.__mro__]
    assert BaseRAGTrainer.__name__ in names_of_base_classes


def test_init(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")
    retriever_training_args = TrainingArguments()
    generator_training_args = TrainingArguments()

    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataset=train_dataset,
        retriever_training_args=retriever_training_args,
        generator_training_args=generator_training_args,
        retriever_train_fn=retriever_trainer_fn,
        generator_train_fn=generator_trainer_fn,
    )

    assert trainer.rag_system == mock_rag_system
    assert trainer.generator_training_args == generator_training_args
    assert trainer.retriever_training_args == retriever_training_args
    assert trainer.train_dataset == train_dataset
    assert trainer.generator_train_fn == generator_trainer_fn
    assert trainer.retriever_train_fn == retriever_trainer_fn
    assert trainer.mode == RAGTrainMode.RETRIEVER


def test_huggingface_extra_missing(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataset: Dataset,
) -> None:
    modules = {
        "transformers": None,
    }
    module_to_import = "fed_rag.rag_trainers.huggingface"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceRAGTrainer` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.rag_trainers.huggingface import HuggingFaceRAGTrainer

            HuggingFaceRAGTrainer(
                rag_system=mock_rag_system,
                mode="retriever",
                train_dataset=train_dataset,
                retriever_train_fn=retriever_trainer_fn,
                generator_train_fn=generator_trainer_fn,
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


def test_invalid_rag_system_due_to_generators(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataset: Dataset,
) -> None:
    retriever_training_args = TrainingArguments()
    generator_training_args = TrainingArguments()

    with pytest.raises(
        FedRAGError,
        match="Generator must be HFPretrainedModelGenerator or HFPeftModelGenerator",
    ):
        HuggingFaceRAGTrainer(
            rag_system=mock_rag_system,
            mode="retriever",
            train_dataset=train_dataset,
            retriever_training_args=retriever_training_args,
            generator_training_args=generator_training_args,
            retriever_train_fn=retriever_trainer_fn,
            generator_train_fn=generator_trainer_fn,
        )


def test_invalid_rag_system_due_to_retriever(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataset: Dataset,
) -> None:
    retriever_training_args = TrainingArguments()
    generator_training_args = TrainingArguments()

    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_model_at_init=False,
    )
    mock_rag_system.generator = generator

    with pytest.raises(
        FedRAGError,
        match="Retriever must be a HFSentenceTransformerRetriever",
    ):
        HuggingFaceRAGTrainer(
            rag_system=mock_rag_system,
            mode="retriever",
            train_dataset=train_dataset,
            retriever_training_args=retriever_training_args,
            generator_training_args=generator_training_args,
            retriever_train_fn=retriever_trainer_fn,
            generator_train_fn=generator_trainer_fn,
        )


@patch.object(HuggingFaceRAGTrainer, "_prepare_retriever_for_training")
def test_train_retriever(
    mock_prepare_retriever_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    mock_retriever_trainer_fn = MagicMock()
    retriever_trainer_args = TrainingArguments()
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataset=train_dataset,
        retriever_train_fn=mock_retriever_trainer_fn,
        retriever_training_args=retriever_trainer_args,
    )

    trainer.train()

    mock_prepare_retriever_for_training.assert_called_once()
    mock_retriever_trainer_fn.assert_called_once_with(
        mock_rag_system, train_dataset, retriever_trainer_args
    )


@patch.object(HuggingFaceRAGTrainer, "_prepare_retriever_for_training")
def test_train_retriever_raises_unspecified_retriever_trainer_error(
    mock_prepare_retriever_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    retriever_trainer_args = TrainingArguments()
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataset=train_dataset,
        retriever_training_args=retriever_trainer_args,
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer function.",
    ):
        trainer.train()
        mock_prepare_retriever_for_training.assert_called_once()


@patch.object(HuggingFaceRAGTrainer, "_prepare_generator_for_training")
def test_train_generator(
    mock_prepare_generator_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    mock_generator_trainer_fn = MagicMock()
    generator_trainer_args = TrainingArguments()
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataset=train_dataset,
        generator_train_fn=mock_generator_trainer_fn,
        generator_training_args=generator_trainer_args,
    )

    trainer.train()

    mock_prepare_generator_for_training.assert_called_once()
    mock_generator_trainer_fn.assert_called_once_with(
        mock_rag_system, train_dataset, generator_trainer_args
    )


@patch.object(HuggingFaceRAGTrainer, "_prepare_generator_for_training")
def test_train_generator_raises_unspecified_generator_trainer_error(
    mock_prepare_generator_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    generator_trainer_args = TrainingArguments()
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataset=train_dataset,
        generator_training_args=generator_trainer_args,
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Attempted to perform generator trainer with an unspecified trainer function.",
    ):
        trainer.train()
        mock_prepare_generator_for_training.assert_called_once()


def test_get_federated_task_retriever(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataset=train_dataset,
        retriever_train_fn=retriever_trainer_fn,
    )

    # act
    retriever_trainer, _ = trainer._get_federated_trainer()
    out = retriever_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == retriever_trainer.__fl_task_trainer_config


def test_get_federated_task_retriever_query_encoder(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    train_dataset: Dataset,
    mock_dual_retriever: BaseRetriever,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataset=train_dataset,
        retriever_train_fn=retriever_trainer_fn,
    )
    mock_rag_system.retriever = mock_dual_retriever
    trainer.rag_system = mock_rag_system

    # act
    fl_task = trainer.get_federated_task()
    _, module = trainer._get_federated_trainer()

    # assert
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert module == mock_dual_retriever.query_encoder


def test_get_federated_task_generator(
    mock_rag_system: RAGSystem,
    generator_trainer_fn: RetrieverTrainFn,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataset=train_dataset,
        generator_train_fn=generator_trainer_fn,
    )

    # act
    generator_trainer, _ = trainer._get_federated_trainer()
    out = generator_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == generator_trainer.__fl_task_trainer_config


def test_invalid_mode_raises_error(
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        HuggingFaceRAGTrainer(
            rag_system=mock_rag_system,
            mode="both",
            train_dataloader=train_dataset,
        )
