import pytest
from torch.utils.data import DataLoader, Dataset

from fed_rag.base.trainer import BaseTrainer
from fed_rag.exceptions import InconsistentDatasetError
from fed_rag.trainers.pytorch import PyTorchTrainerProtocol, TrainingArgs
from fed_rag.types.rag_system import RAGSystem

from .conftest import TestGeneratorTrainer, TestRetrieverTrainer


def test_pt_trainer_init(
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    training_args = TrainingArgs()
    trainer = TestGeneratorTrainer(
        rag_system=mock_rag_system,
        train_dataloader=train_dataloader,
        training_arguments=training_args,
    )

    assert trainer.rag_system == mock_rag_system
    assert trainer.model == mock_rag_system.generator.model
    assert trainer.train_dataset == train_dataset
    assert trainer.train_dataloader == train_dataloader
    assert trainer.training_arguments == training_args
    assert isinstance(trainer, PyTorchTrainerProtocol)
    assert isinstance(trainer, BaseTrainer)


def test_pt_retriever_trainer(
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = TestRetrieverTrainer(
        rag_system=mock_rag_system,
        train_dataloader=train_dataloader,
    )

    assert trainer.rag_system == mock_rag_system
    assert trainer.model == mock_rag_system.retriever.encoder
    assert trainer.train_dataset == train_dataset
    assert trainer.train_dataloader == train_dataloader
    assert trainer.training_arguments is None
    assert isinstance(trainer, PyTorchTrainerProtocol)
    assert isinstance(trainer, BaseTrainer)


def test_pt_trainer_init_raises_inconsistent_dataset_error(
    mock_rag_system: RAGSystem,
    another_train_dataset: Dataset,
    train_dataloader: DataLoader,
) -> None:
    msg = (
        "Inconsistent datasets detected between supplied `train_dataset` and that "
        "associated with the `train_dataloader`. These two datasets must be the same."
    )

    with pytest.raises(InconsistentDatasetError, match=msg):
        # arrange
        TestGeneratorTrainer(
            rag_system=mock_rag_system,
            train_dataset=another_train_dataset,
            train_dataloader=train_dataloader,
        )
