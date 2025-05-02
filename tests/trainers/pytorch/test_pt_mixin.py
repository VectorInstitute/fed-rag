from pytest import MonkeyPatch
from torch.utils.data import DataLoader, Dataset

from fed_rag.base.trainer import BaseTrainer
from fed_rag.trainers.pytorch import PyTorchTrainerProtocol, TrainingArgs
from fed_rag.types.rag_system import RAGSystem

from .conftest import TestGeneratorTrainer


def test_hf_trainer_init(
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    train_dataloader: DataLoader,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

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
