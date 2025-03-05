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
        ...

    @abstractmethod
    def encode_context(
        self, context: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        """Encode context."""
        ...

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        ...
