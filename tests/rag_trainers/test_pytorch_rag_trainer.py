from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader

from fed_rag.base.rag_trainer import BaseRAGTrainer
from fed_rag.exceptions import (
    UnspecifiedGeneratorTrainer,
    UnspecifiedRetrieverTrainer,
    UnsupportedTrainerMode,
)
from fed_rag.fl_tasks.pytorch import PyTorchFLTask
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
    train_dataloader: DataLoader,
) -> None:
    retriever_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_training_args=retriever_trainer_args,
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Attempted to perform retriever trainer with an unspecified trainer function.",
    ):
        trainer.train()
        mock_prepare_retriever_for_training.assert_called_once()


@patch.object(PyTorchRAGTrainer, "_prepare_generator_for_training")
def test_train_generator(
    mock_prepare_generator_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    train_dataloader: DataLoader,
) -> None:
    mock_generator_trainer_fn = MagicMock()
    generator_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataloader=train_dataloader,
        generator_train_fn=mock_generator_trainer_fn,
        generator_training_args=generator_trainer_args,
    )

    trainer.train()

    mock_prepare_generator_for_training.assert_called_once()
    mock_generator_trainer_fn.assert_called_once_with(
        mock_rag_system, train_dataloader, generator_trainer_args
    )


@patch.object(PyTorchRAGTrainer, "_prepare_generator_for_training")
def test_train_generator_raises_unspecified_generator_trainer_error(
    mock_prepare_generator_for_training: MagicMock,
    mock_rag_system: RAGSystem,
    train_dataloader: DataLoader,
) -> None:
    generator_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataloader=train_dataloader,
        generator_training_args=generator_trainer_args,
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Attempted to perform generator trainer with an unspecified trainer function.",
    ):
        trainer.train()
        mock_prepare_generator_for_training.assert_called_once()


def test_train_with_invalid_mode_raises_error(
    mock_rag_system: RAGSystem,
    train_dataloader: DataLoader,
) -> None:
    generator_trainer_args = TrainingArgs(
        learning_rate=0.42, custom_kwargs={"param": True}
    )
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="both",
        train_dataloader=train_dataloader,
        generator_training_args=generator_trainer_args,
    )

    with pytest.raises(
        UnsupportedTrainerMode, match="Unsupported trainer mode: 'both'"
    ):
        trainer.train()


def test_prepare_generator_for_training_mono_encoder(
    mock_rag_system: RAGSystem,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataloader=train_dataloader,
        generator_train_fn=generator_trainer_fn,
    )
    mock_rag_system = MagicMock()
    mock_rag_system.retriever.encoder.return_value = True
    trainer.rag_system = mock_rag_system

    trainer._prepare_generator_for_training()

    mock_rag_system.generator.model.train.assert_called_once()
    mock_rag_system.retriever.encoder.eval.assert_called_once()


def test_prepare_generator_for_training_dual_encoder(
    mock_rag_system: RAGSystem,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataloader=train_dataloader,
        generator_train_fn=generator_trainer_fn,
    )
    mock_rag_system = MagicMock()
    mock_rag_system.retriever.encoder.return_value = False
    trainer.rag_system = mock_rag_system

    trainer._prepare_generator_for_training()

    mock_rag_system.generator.model.train.assert_called_once()
    mock_rag_system.retriever.query_encoder.eval.assert_called_once()
    mock_rag_system.retriever.context_encoder.eval.assert_called_once()


def test_prepare_retriever_for_training_mono_encoder(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    train_dataloader: DataLoader,
) -> None:
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_train_fn=retriever_trainer_fn,
    )
    mock_rag_system = MagicMock()
    mock_rag_system.retriever.encoder.return_value = True
    trainer.rag_system = mock_rag_system

    trainer._prepare_retriever_for_training()

    mock_rag_system.generator.model.eval.assert_called_once()
    mock_rag_system.retriever.encoder.train.assert_called_once()


@pytest.mark.parametrize(
    (
        "context_encoder_frozen",
        "context_encoder_train_count",
        "context_encoder_eval_count",
    ),
    [(True, 0, 1), (False, 1, 0)],
)
def test_prepare_retriever_for_training_dual_encoder(
    context_encoder_frozen: bool,
    context_encoder_train_count: int,
    context_encoder_eval_count: int,
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_train_fn=retriever_trainer_fn,
    )
    mock_rag_system = MagicMock()
    mock_rag_system.retriever.encoder.return_value = False
    trainer.rag_system = mock_rag_system

    # act
    trainer._prepare_retriever_for_training(
        freeze_context_encoder=context_encoder_frozen
    )

    # assert
    mock_rag_system.generator.model.eval.assert_called_once()
    mock_rag_system.retriever.query_encoder.train.assert_called_once()
    mock_rag_system.retriever.context_encoder.eval.call_count == context_encoder_eval_count
    mock_rag_system.retriever.context_encoder.train.call_count == context_encoder_train_count


def test_get_federated_task_retriever(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_train_fn=retriever_trainer_fn,
    )

    # act
    retriever_trainer, _ = trainer._get_federated_trainer()
    out = retriever_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, PyTorchFLTask)
    assert fl_task._trainer_spec == retriever_trainer.__fl_task_trainer_config


def test_get_federated_task_retriever_query_encoder(
    mock_rag_system: RAGSystem,
    retriever_trainer_fn: RetrieverTrainFn,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
        retriever_train_fn=retriever_trainer_fn,
    )
    mock_rag_system = MagicMock()
    mock_rag_system.retriever.encoder.return_value = False
    trainer.rag_system = mock_rag_system

    # act
    fl_task = trainer.get_federated_task()

    # assert
    assert isinstance(fl_task, PyTorchFLTask)


def test_get_federated_task_generator(
    mock_rag_system: RAGSystem,
    generator_trainer_fn: GeneratorTrainFn,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataloader=train_dataloader,
        generator_train_fn=generator_trainer_fn,
    )

    # act
    generator_trainer, _ = trainer._get_federated_trainer()
    out = generator_trainer(MagicMock(), MagicMock(), MagicMock())
    fl_task = trainer.get_federated_task()

    # assert
    assert out.loss == 0
    assert isinstance(fl_task, PyTorchFLTask)
    assert fl_task._trainer_spec == generator_trainer.__fl_task_trainer_config


def test_get_federated_task_raises_unspecified_trainer_retriever(
    mock_rag_system: RAGSystem,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataloader=train_dataloader,
    )

    with pytest.raises(
        UnspecifiedRetrieverTrainer,
        match="Cannot federate an unspecified retriever trainer function.",
    ):
        trainer.get_federated_task()


def test_get_federated_task_raises_unspecified_trainer_generator(
    mock_rag_system: RAGSystem,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="generator",
        train_dataloader=train_dataloader,
    )

    with pytest.raises(
        UnspecifiedGeneratorTrainer,
        match="Cannot federate an unspecified generator trainer function.",
    ):
        trainer.get_federated_task()


def test_get_federated_task_raises_unsupported_trainer_mode(
    mock_rag_system: RAGSystem,
    train_dataloader: DataLoader,
) -> None:
    # arrange
    trainer = PyTorchRAGTrainer(
        rag_system=mock_rag_system,
        mode="both",
        train_dataloader=train_dataloader,
    )

    with pytest.raises(
        UnsupportedTrainerMode, match="Unsupported trainer mode: 'both'"
    ):
        trainer.get_federated_task()
