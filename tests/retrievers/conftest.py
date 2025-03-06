from typing import Any

import pytest
import torch
from pydantic import PrivateAttr

from fed_rag.base.retriever import BaseRetriever


class MockRetriever(BaseRetriever):
    _model: torch.nn.Module = PrivateAttr(default=torch.nn.Linear(2, 1))

    def encode_context(self, context: str, **kwargs: Any) -> torch.Tensor:
        return self._model.forward(torch.ones(2))

    def encode_query(self, query: str, **kwargs: Any) -> torch.Tensor:
        return self._model.forward(torch.zeros(2))

    @property
    def model(self) -> torch.nn.Module:
        return self._model


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()
