from torch.utils.data import DataLoader

from fed_rag.base.rag_trainer import BaseRAGTrainer
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
