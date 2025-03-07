"""Base Retriever"""

from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class BaseRetriever(BaseModel, ABC):
    """Base Retriever Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def encode_query(
        self, query: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        """Encode query."""

    @abstractmethod
    def encode_context(
        self, context: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        """Encode context."""

    @property
    @abstractmethod
    def encoder(self) -> torch.nn.Module | None:
        """PyTorch model associated with the encoder associated with retriever."""

    @property
    @abstractmethod
    def query_encoder(self) -> torch.nn.Module | None:
        """PyTorch model associated with the query encoder associated with retriever."""

    @property
    @abstractmethod
    def context_encoder(self) -> torch.nn.Module | None:
        """PyTorch model associated with the context encoder associated with retriever."""
