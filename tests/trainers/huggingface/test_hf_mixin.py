from datasets import Dataset
from transformers import TrainingArguments

from fed_rag.base.trainer import BaseTrainer
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerProtocol
from fed_rag.types.rag_system import RAGSystem

from .conftest import TestHFTrainer


def test_hf_trainer_init(
    train_dataset: Dataset, hf_rag_system: RAGSystem
) -> None:
    # arrange
    training_args = TrainingArguments()
    trainer = TestHFTrainer(
        rag_system=hf_rag_system,
        model=hf_rag_system.retriever.encoder,
        train_dataset=train_dataset,
        training_arguments=training_args,
    )

    assert trainer.rag_system == hf_rag_system
    assert trainer.model == hf_rag_system.retriever.encoder
    assert trainer.train_dataset == train_dataset
    assert trainer.training_arguments == training_args
    assert isinstance(trainer, HuggingFaceTrainerProtocol)
    assert isinstance(trainer, BaseTrainer)
