from typing import Any

import pytest
import torch
from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.retriever import BaseRetriever
from fed_rag.types.knowledge_node import KnowledgeNode


class MockRetriever(BaseRetriever):
    _encoder: torch.nn.Module = PrivateAttr(default=torch.nn.Linear(3, 3))

    def encode_context(self, context: str, **kwargs: Any) -> torch.Tensor:
        return self._encoder.forward(torch.ones(3))

    def encode_query(self, query: str, **kwargs: Any) -> torch.Tensor:
        return self._encoder.forward(torch.zeros(3))

    @property
    def encoder(self) -> torch.nn.Module:
        return self._encoder

    @property
    def query_encoder(self) -> torch.nn.Module | None:
        return None

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return None


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()


@pytest.fixture
def dummy_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer(modules=[torch.nn.Linear(5, 5)])


@pytest.fixture
def knowledge_nodes() -> list[KnowledgeNode]:
    return [
        KnowledgeNode(
            embedding=[1.0, 0.0, 1.0], node_type="text", text_content="node 1"
        ),
        KnowledgeNode(
            embedding=[1.0, 0.0, 0.0],
            node_type="multimodal",
            text_content="node 2",
            image_content=b"node 2",
        ),
        KnowledgeNode(
            embedding=[
                1.0,
                1.0,
                0.0,
            ],
            node_type="multimodal",
            text_content="node 3",
            image_content=b"node 3",
        ),
    ]


class MockGenerator(BaseGenerator):
    def generate(self, query: str, context: str, **kwargs: dict) -> str:
        return f"mock output from '{query}' with '{context}'."

    @property
    def model(self) -> torch.nn.Module:
        return torch.nn.Linear(2, 1)


@pytest.fixture
def mock_generator() -> BaseGenerator:
    return MockGenerator()
