"""Base Retriever."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class BaseRetriever(BaseModel, ABC):
    """Base Retriever Class.

    This abstract class provides the interface for creating Retriever objects that
    encode strings into numerical vector representations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def encode_query(
        self, query: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        """Encode a string query into a torch.Tensor.

        Args:
            query (str | list[str]): The query or list of queries to encode.

        Returns:
            torch.Tensor: The vector representation(s) of the encoded query/queries.
        """

    @abstractmethod
    def encode_context(
        self, context: str | list[str], **kwargs: Any
    ) -> torch.Tensor:
        """Encode a string context into a torch.Tensor.

        Args:
            context (str | list[str]): The context or list of contexts to encode.

        Returns:
            torch.Tensor: The vector representation(s) of the encoded context(s).
        """

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
