"""Base Retriever."""

from abc import ABC, abstractmethod
from typing import Any, TypedDict
from fed_rag.data_structures.rag import Context, Query

import torch
from pydantic import BaseModel, ConfigDict

class EncoderType(TypedDict):
    text: torch.nn.Module | None
    image: torch.nn.Module | None

class BaseRetriever(BaseModel, ABC):
    """Base Retriever Class.

    This abstract class provides the interface for creating Retriever objects that
    encode strings into numerical vector representations.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def encode_query(
        self, query: str | list[str] | Query | list[Query], **kwargs: Any
    ) -> torch.Tensor:
        """Encode a string query into a torch.Tensor.

        Args:
            query (str | list[str] | Query | list[Query]): The query or list of queries to encode.

        Returns:
            torch.Tensor: The vector representation(s) of the encoded query/queries.
        """

    @abstractmethod
    def encode_context(
        self, context: str | list[str] | Context | list[Context], **kwargs: Any
    ) -> torch.Tensor:
        """Encode a string context into a torch.Tensor.

        Args:
            context (str | list[str] | Context | list[Context]): The context or list of contexts to encode.

        Returns:
            torch.Tensor: The vector representation(s) of the encoded context(s).
        """

    @property
    @abstractmethod
    def encoder(self) -> torch.nn.Module | EncoderType | None:
        """PyTorch model associated with the encoder associated with retriever."""

    @property
    @abstractmethod
    def query_encoder(self) -> torch.nn.Module | EncoderType | None:
        """PyTorch model associated with the query encoder associated with retriever."""

    @property
    @abstractmethod
    def context_encoder(self) -> torch.nn.Module | EncoderType | None:
        """PyTorch model associated with the context encoder associated with retriever."""
