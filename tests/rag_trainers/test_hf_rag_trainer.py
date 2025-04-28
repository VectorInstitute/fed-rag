from datasets import Dataset
from transformers import Trainer, TrainingArguments

from fed_rag.base.rag_trainer import BaseRAGTrainer, RAGTrainMode
from fed_rag.rag_trainers.huggingface import HFRAGTrainer
from fed_rag.types.rag_system import RAGSystem


def test_pt_rag_trainer_class() -> None:
    names_of_base_classes = [b.__name__ for b in HFRAGTrainer.__mro__]
    assert BaseRAGTrainer.__name__ in names_of_base_classes


def test_init(
    mock_rag_system: RAGSystem,
    retriever_trainer: Trainer,
    retriever_training_args: TrainingArguments,
    generator_trainer: Trainer,
    generator_training_args: TrainingArguments,
    train_dataset: Dataset,
) -> None:
    trainer = HFRAGTrainer(
        rag_system=mock_rag_system,
        mode="retriever",
        train_dataset=train_dataset,
        retriever_training_args=retriever_training_args,
        generator_training_args=generator_training_args,
        retriever_trainer=retriever_trainer,
        generator_trainer=generator_trainer,
    )

    assert trainer.rag_system == mock_rag_system
    assert trainer.generator_trainer == generator_trainer
    assert trainer.retriever_trainer == retriever_trainer
    assert trainer.generator_training_args == generator_training_args
    assert trainer.retriever_training_args == retriever_training_args
    assert trainer.train_dataset == train_dataset
    assert trainer.mode == RAGTrainMode.RETRIEVER
