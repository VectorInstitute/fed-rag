from unittest.mock import MagicMock, patch

import pytest

from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.base.trainer_manager import BaseRAGTrainerManager, RAGTrainMode
from fed_rag.exceptions import (
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
from fed_rag.trainer_managers.pytorch import PyTorchRAGTrainerManager


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in PyTorchRAGTrainerManager.__mro__
    ]
    assert BaseRAGTrainerManager.__name__ in names_of_base_classes


def test_init(
    retriever_trainer: BaseRetrieverTrainer,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )

    assert manager.generator_trainer == generator_trainer
    assert manager.retriever_trainer == retriever_trainer
    assert manager.mode == "retriever"


@patch.object(PyTorchRAGTrainerManager, "_prepare_retriever_for_training")
def test_train_retriever(
    mock_prepare_retriever_for_training: MagicMock,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )
    # mock it
    mock_retriever_trainer = MagicMock()
    manager.retriever_trainer = mock_retriever_trainer

    manager.train()

    mock_prepare_retriever_for_training.assert_called_once()
    mock_retriever_trainer.train.assert_called_once_with()


@patch.object(PyTorchRAGTrainerManager, "_prepare_retriever_for_training")
def test_train_retriever_raises_unspecified_retriever_trainer_error(
    mock_prepare_retriever_for_training: MagicMock,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="retriever",
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer function.",
    ):
        manager.train()
        mock_prepare_retriever_for_training.assert_called_once()


@patch.object(PyTorchRAGTrainerManager, "_prepare_generator_for_training")
def test_train_generator(
    mock_prepare_generator_for_training: MagicMock,
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="generator",
    )
    # mock it
    mock_generator_trainer = MagicMock()
    manager.generator_trainer = mock_generator_trainer

    manager.train()

    mock_prepare_generator_for_training.assert_called_once()
    mock_generator_trainer.train.assert_called_once_with()


@patch.object(PyTorchRAGTrainerManager, "_prepare_generator_for_training")
def test_train_generator_raises_unspecified_generator_trainer_error(
    mock_prepare_generator_for_training: MagicMock,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="generator",
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Attempted to perform generator trainer with an unspecified trainer function.",
    ):
        manager.train()
        mock_prepare_generator_for_training.assert_called_once()


def test_get_federated_task_retriever(
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    # arrange
    manager = PyTorchRAGTrainerManager(
        mode="retriever",
        retriever_trainer=retriever_trainer,
    )

    # act
    retriever_trainer, _ = manager._get_federated_trainer()
    out = retriever_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = manager.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, PyTorchFLTask)
    assert fl_task._trainer_spec == retriever_trainer.__fl_task_trainer_config


def test_get_federated_task_generator(
    generator_trainer: BaseGeneratorTrainer,
) -> None:
    # arrange
    manager = PyTorchRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
    )

    # act
    generator_trainer, _ = manager._get_federated_trainer()
    out = generator_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = manager.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, PyTorchFLTask)
    assert fl_task._trainer_spec == generator_trainer.__fl_task_trainer_config


def test_get_federated_task_raises_unspecified_trainer_retriever() -> None:
    # arrange
    manager = PyTorchRAGTrainerManager(
        mode="retriever",
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Cannot federate an unspecified retriever trainer function.",
    ):
        manager.get_federated_task()


def test_get_federated_task_raises_unspecified_trainer_generator() -> None:
    # arrange
    trainer = PyTorchRAGTrainerManager(
        mode="generator",
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Cannot federate an unspecified generator trainer function.",
    ):
        trainer.get_federated_task()


def test_invalid_mode_raises_error() -> None:
    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        PyTorchRAGTrainerManager(
            mode="both",
        )


def test_prepare_generator_for_training(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
    )

    # add mocks
    mock_generator_model = MagicMock()
    mock_retriever_model = MagicMock()
    manager.generator_trainer.model = mock_generator_model
    manager.retriever_trainer.model = mock_retriever_model

    # act
    manager._prepare_generator_for_training()

    mock_generator_model.train.assert_called_once()
    mock_retriever_model.eval.assert_called_once()


def test_prepare_retriever_for_training(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    manager = PyTorchRAGTrainerManager(
        mode="generator",
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
    )

    # add mocks
    mock_generator_model = MagicMock()
    mock_retriever_model = MagicMock()
    manager.generator_trainer.model = mock_generator_model
    manager.retriever_trainer.model = mock_retriever_model

    # act
    manager._prepare_retriever_for_training()

    mock_generator_model.eval.assert_called_once()
    mock_retriever_model.train.assert_called_once()
