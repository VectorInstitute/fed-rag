import re
import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch

from fed_rag.base.trainer import (
    BaseGeneratorTrainer,
    BaseRetrieverTrainer,
    BaseTrainer,
)
from fed_rag.base.trainer_manager import BaseRAGTrainerManager, RAGTrainMode
from fed_rag.exceptions import (
    MissingExtraError,
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)
from fed_rag.fl_tasks.huggingface import HuggingFaceFLTask
from fed_rag.trainer_managers.huggingface import HuggingFaceRAGTrainerManager


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HuggingFaceRAGTrainerManager.__mro__
    ]
    assert BaseRAGTrainerManager.__name__ in names_of_base_classes


def test_init(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        train_dataset=train_dataset,
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )

    assert manager.generator_trainer == generator_trainer
    assert manager.retriever_trainer == retriever_trainer
    assert manager.mode == RAGTrainMode.RETRIEVER


def test_huggingface_extra_missing(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    modules = {
        "datasets": None,
    }
    module_to_import = "fed_rag.trainer_managers.huggingface"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceRAGTrainerManager` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainer_managers.huggingface import (
                HuggingFaceRAGTrainerManager,
            )

            HuggingFaceRAGTrainerManager(
                mode="retriever",
                retriever_trainer=retriever_trainer,
                generator_trainer=generator_trainer,
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_retriever_for_training")
def test_train_retriever(
    mock_prepare_retriever_for_training: MagicMock,
    retriever_trainer: BaseRetrieverTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    mock_retriever_trainer = MagicMock()
    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )
    manager.retriever_trainer = mock_retriever_trainer

    manager.train()

    mock_prepare_retriever_for_training.assert_called_once()
    mock_retriever_trainer.train.assert_called_once_with()


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_retriever_for_training")
def test_train_retriever_raises_unspecified_retriever_trainer_error(
    mock_prepare_retriever_for_training: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="retriever",
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer function.",
    ):
        manager.train()
        mock_prepare_retriever_for_training.assert_called_once()


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_generator_for_training")
def test_train_generator(
    mock_prepare_generator_for_training: MagicMock,
    generator_trainer: BaseGeneratorTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )
    mock_generator_trainer = MagicMock()
    manager.generator_trainer = mock_generator_trainer

    manager.train()

    mock_prepare_generator_for_training.assert_called_once()
    mock_generator_trainer.train.assert_called_once_with()


@patch.object(HuggingFaceRAGTrainerManager, "_prepare_generator_for_training")
def test_train_generator_raises_unspecified_generator_trainer_error(
    mock_prepare_generator_for_training: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Attempted to perform generator trainer with an unspecified trainer function.",
    ):
        manager.train()
        mock_prepare_generator_for_training.assert_called_once()


def test_get_federated_task_retriever(
    retriever_trainer: BaseTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )

    # act
    retriever_trainer, _ = trainer._get_federated_trainer()
    out = retriever_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, HuggingFaceFLTask)
    assert fl_task._trainer_spec == retriever_trainer.__fl_task_trainer_config


def test_get_federated_task_generator(
    generator_trainer: BaseTrainer,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    trainer = HuggingFaceRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
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
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        HuggingFaceRAGTrainerManager(
            mode="both",
        )


def test_get_federated_task_raises_unspecified_trainers(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    manager = HuggingFaceRAGTrainerManager(
        mode="generator",
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Cannot federate an unspecified generator trainer function.",
    ):
        manager.get_federated_task()

    # change mode to retriever
    manager.mode = "retriever"
    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Cannot federate an unspecified retriever trainer function.",
    ):
        manager.get_federated_task()


def test_prepare_generator_for_training(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceRAGTrainerManager(
        mode="generator",
    )

    with does_not_raise():
        trainer._prepare_generator_for_training()


def test_prepare_retriever_for_training(
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceRAGTrainerManager(
        mode="retriever",
    )

    with does_not_raise():
        trainer._prepare_retriever_for_training()
