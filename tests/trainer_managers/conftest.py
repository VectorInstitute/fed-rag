from typing import Any

import numpy as np
import pytest
import torch
from pydantic import PrivateAttr
from torch.utils.data import DataLoader, Dataset

from fed_rag.base.fl_task import BaseFLTask
from fed_rag.base.generator import BaseGenerator
from fed_rag.base.retriever import BaseRetriever
from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.base.trainer_manager import BaseRAGTrainerManager
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.types.rag_system import RAGConfig, RAGSystem
from fed_rag.types.results import TestResult, TrainResult


class _TestDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.features = np.random.rand(size, 2)
        self.labels = np.random.choice(2, size=size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[np.ndarray, Any]:
        return self.features[index], self.labels[index]


@pytest.fixture()
def train_dataloader() -> DataLoader:
    dataset = _TestDataset(size=10)
    return DataLoader(dataset, batch_size=2, shuffle=True)


@pytest.fixture()
def val_dataloader() -> DataLoader:
    dataset = _TestDataset(size=4)
    return DataLoader(dataset, batch_size=2, shuffle=True)


class MockRetriever(BaseRetriever):
    _encoder: torch.nn.Module = PrivateAttr(default=torch.nn.Linear(3, 3))

    def encode_context(self, context: str, **kwargs: Any) -> torch.Tensor:
        return self._encoder.forward(torch.ones(3))

    def encode_query(self, query: str, **kwargs: Any) -> torch.Tensor:
        return self._encoder.forward(torch.zeros(3))

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @encoder.setter
    def encoder(self, value: torch.nn.Module) -> None:
        self._encoder = value

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return None


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()


class MockDualRetriever(BaseRetriever):
    _query_encoder: torch.nn.Module = PrivateAttr(
        default=torch.nn.Linear(2, 1)
    )
    _context_encoder: torch.nn.Module = PrivateAttr(
        default=torch.nn.Linear(2, 1)
    )

    def encode_context(self, context: str, **kwargs: Any) -> torch.Tensor:
        return self._encoder.forward(torch.ones(2))

    def encode_query(self, query: str, **kwargs: Any) -> torch.Tensor:
        return self._encoder.forward(torch.zeros(2))

    @property
    def encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return self._query_encoder

    @query_encoder.setter
    def query_encoder(self, value: torch.nn.Module) -> None:
        self._query_encoder = value

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return self._context_encoder

    @context_encoder.setter
    def context_encoder(self, value: torch.nn.Module) -> None:
        self._context_encoder = value


@pytest.fixture
def mock_dual_retriever() -> MockDualRetriever:
    return MockDualRetriever()


class MockTokenizer(BaseTokenizer):
    def encode(self, input: str, **kwargs: Any) -> list[int]:
        return [0, 1, 2]

    def decode(self, input_ids: list[int], **kwargs: Any) -> str:
        return "mock decoded sentence"

    @property
    def unwrapped(self) -> None:
        return None


@pytest.fixture()
def mock_tokenizer() -> BaseTokenizer:
    return MockTokenizer()


class MockGenerator(BaseGenerator):
    _model = torch.nn.Linear(2, 1)
    _tokenizer = MockTokenizer()

    def generate(self, input: str) -> str:
        return f"mock output from '{input}'."

    def compute_target_sequence_proba(self, prompt: str, target: str) -> float:
        return 0.42

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @model.setter
    def model(self, value: torch.nn.Module) -> None:
        self._model = value

    @property
    def tokenizer(self) -> MockTokenizer:
        return self._tokenizer


@pytest.fixture
def mock_generator() -> BaseGenerator:
    return MockGenerator()


@pytest.fixture
def mock_rag_system(
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
) -> RAGSystem:
    knowledge_store = InMemoryKnowledgeStore()
    rag_config = RAGConfig(
        top_k=2,
    )
    return RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )


@pytest.fixture
def another_mock_rag_system(
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
) -> RAGSystem:
    knowledge_store = InMemoryKnowledgeStore()
    rag_config = RAGConfig(
        top_k=1,
    )
    return RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )


class MockRAGTrainerManager(BaseRAGTrainerManager):
    def _prepare_generator_for_training(self, **kwargs: Any) -> None:
        return None

    def _prepare_retriever_for_training(
        self, decoder: bool = False, **kwargs: Any
    ) -> None:
        return None

    def _train_retriever(self, **kwargs: Any) -> str:
        return "retriever trained"

    def _train_generator(self, **kwargs: Any) -> str:
        return "generator trained"

    def train(self, **kwargs: Any) -> str:
        return "rag system trained"

    def get_federated_task(self) -> BaseFLTask:
        raise NotImplementedError()


class TestRetrieverTrainer(BaseRetrieverTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)


class TestGeneratorTrainer(BaseGeneratorTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)


@pytest.fixture
def retriever_trainer(mock_rag_system: RAGSystem) -> BaseRetrieverTrainer:
    return TestRetrieverTrainer(
        rag_system=mock_rag_system,
        train_dataset=[{"query": "mock query", "response": "mock response"}],
    )


@pytest.fixture
def generator_trainer(mock_rag_system: RAGSystem) -> BaseGeneratorTrainer:
    return TestGeneratorTrainer(
        rag_system=mock_rag_system,
        train_dataset=[{"query": "mock query", "response": "mock response"}],
    )
