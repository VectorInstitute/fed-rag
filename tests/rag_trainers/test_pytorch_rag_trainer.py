from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader

from fed_rag.base.rag_trainer import BaseRAGTrainer
from fed_rag.exceptions import UnspecifiedRetrieverTrainer
from fed_rag.rag_trainers.pytorch import (
    GeneratorTrainFn,
    PyTorchRAGTrainer,
    RetrieverTrainFn,
    TrainingArgs,
)
from fed_rag.types.rag_system import RAGSystem


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [b.__name__ for b in PyTorchRAGTrainer.__mro__]
    assert BaseRAGTrainer.__name__ in names_of_base_classes


def test_init(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    retriever_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    generator_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": False}
    )

    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_training_args=retriever_trainer_args,
        generator_training_args=generator_trainer_args,
        retriever_train_fn=retriever_trainer_fn,
        generator_train_fn=generator_trainer_fn,
    )

    assert trainer.rag_system == mock_rag_system
    assert trainer.generator_train_fn == generator_trainer_fn
    assert trainer.retriever_train_fn == retriever_trainer_fn
    assert trainer.generator_training_args == generator_trainer_args
    assert trainer.retriever_training_args == retriever_trainer_args
    assert trainer.train_dataloader == train_dataloader


def test_init_with_trainer_args_dict(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    retriever_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    generator_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": False}
    )

    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_training_args=retriever_trainer_args.model_dump(),
        generator_training_args=generator_trainer_args.model_dump(),
        retriever_train_fn=retriever_trainer_fn,
        generator_train_fn=generator_trainer_fn,
    )

    assert trainer.generator_training_args == generator_trainer_args
    assert trainer.retriever_training_args == retriever_trainer_args


def test_init_with_no_trainer_args(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_train_fn=retriever_trainer_fn,
        generator_train_fn=generator_trainer_fn,
    )

    assert trainer.generator_training_args == TrainingArgs()
    assert trainer.retriever_training_args == TrainingArgs()


@patch.object(PyTorchRAGTrainer, "_prepare_retriever_for_training")
def test_train_retriever(
    mock_prepare_retriever_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    mock_retriever_trainer_fn = MagicMock()
    retriever_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_train_fn=mock_retriever_trainer_fn,
        generator_train_fn=generator_trainer_fn,
        retriever_training_args=retriever_trainer_args,
    )

    trainer.train()

    mock_prepare_retriever_for_training.assert_called_once()
    mock_retriever_trainer_fn.assert_called_once_with(
        mock_rag_system, train_dataloader, retriever_trainer_args
    )


@patch.object(PyTorchRAGTrainer, "_prepare_retriever_for_training")
def test_train_retriever_raises_unspecified_retriever_trainer_error(
    mock_prepare_retriever_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    retriever_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        generator_train_fn=generator_trainer_fn,
        retriever_training_args=retriever_trainer_args,
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer function.",
    ):
        trainer.train()
        mock_prepare_retriever_for_training.assert_called_once()
