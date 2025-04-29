from typing import Any

import pytest
from datasets import Dataset
from pytest import MonkeyPatch
from transformers import TrainingArguments

from fed_rag.base.rag_trainer import BaseRAGTrainer, RAGTrainMode
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
