import pytest
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import Trainer

from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult


class TestHFRetrieverTrainer(HuggingFaceTrainerMixin, BaseRetrieverTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)

    def hf_trainer_obj(self) -> Trainer:
        return Trainer()


class TestHFGeneratorTrainer(HuggingFaceTrainerMixin, BaseGeneratorTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)

    def hf_trainer_obj(self) -> Trainer:
        return Trainer()


@pytest.fixture()
def train_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "query": ["first query", "second query"],
            "response": ["first response", "second response"],
        }
    )


@pytest.fixture()
def hf_rag_system(mock_rag_system: RAGSystem) -> RAGSystem:
    encoder = SentenceTransformer(modules=[torch.nn.Linear(5, 5)])
    encoder.tokenizer = None
    mock_rag_system.retriever.encoder = encoder
    return mock_rag_system
