from typing import Any

import pytest
import torch
from pydantic import PrivateAttr
from sentence_transformers import SentenceTransformer

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.knowledge_store import BaseAsyncKnowledgeStore
from fed_rag.base.retriever import BaseRetriever
from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.data_structures import KnowledgeNode


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

    @property
    def context_encoder(self) -> torch.nn.Module | None:
        return self._context_encoder


class DummyAsyncKnowledgeStore(BaseAsyncKnowledgeStore):
    nodes: list[KnowledgeNode] = []

    async def load_node(self, node: KnowledgeNode) -> None:
        self.nodes.append(node)

    async def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        return []

    async def delete_node(self, node_id: str) -> bool:
        return True

    async def clear(self) -> None:
        self.nodes.clear()

    async def count(self) -> int:
        return len(self.nodes)

    async def persist(self) -> None:
        pass

    async def load(self) -> None:
        pass


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()


@pytest.fixture
def mock_dual_retriever() -> MockDualRetriever:
    return MockDualRetriever()


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
    _prompt_template = "{query} and {context}"

    def generate(self, query: str, context: str, **kwargs: Any) -> str:
        return f"mock output from '{query}' and '{context}'."

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

    @tokenizer.setter
    def tokenizer(self, value: BaseTokenizer) -> None:
        self._tokenizer = value

    @property
    def prompt_template(self) -> str:
        return self._prompt_template

    @prompt_template.setter
    def prompt_template(self, v: str) -> None:
        self._prompt_template = v


@pytest.fixture
def mock_generator() -> BaseGenerator:
    return MockGenerator()
