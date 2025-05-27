"""Base Knowledge Store"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover
    from fed_rag.data_structures.knowledge_node import KnowledgeNode

DEFAULT_KNOWLEDGE_STORE_NAME = "default"


class BaseKnowledgeStore(BaseModel, ABC):
    """Base Knowledge Store Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        description="Name of Knowledge Store used for caching and loading.",
        default=DEFAULT_KNOWLEDGE_STORE_NAME,
    )

    @abstractmethod
    def load_node(self, node: "KnowledgeNode") -> None:
        """Load a "KnowledgeNode" into the KnowledgeStore."""

    @abstractmethod
    def load_nodes(self, nodes: list["KnowledgeNode"]) -> None:
        """Load multiple "KnowledgeNode"s in batch."""

    @abstractmethod
    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, "KnowledgeNode"]]:
        """Retrieve top-k nodes from KnowledgeStore against a provided user query.

        Returns:
            A list of tuples where the first element represents the similarity score
            of the node to the query, and the second element is the node itself.
        """

    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Remove a node from the KnowledgeStore by ID, returning success status."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all nodes from the KnowledgeStore."""

    @property
    @abstractmethod
    def count(self) -> int:
        """Return the number of nodes in the store."""

    @abstractmethod
    def persist(self) -> None:
        """Save the KnowledgeStore nodes to a permanent storage."""

    @abstractmethod
    def load(self) -> None:
        """Load the KnowledgeStore nodes from a permanent storage using `name`."""


class BaseAsyncKnowledgeStore(BaseModel, ABC):
    """Base Asynchronous Knowledge Store Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        description="Name of Knowledge Store used for caching and loading.",
        default=DEFAULT_KNOWLEDGE_STORE_NAME,
    )

    @abstractmethod
    async def load_node(self, node: "KnowledgeNode") -> None:
        """Asynchronously load a "KnowledgeNode" into the KnowledgeStore."""

    async def load_nodes(self, nodes: list["KnowledgeNode"]) -> None:
        """Default batch loader via concurrent load_node calls."""
        await asyncio.gather(*(self.load_node(n) for n in nodes))

    @abstractmethod
    async def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, "KnowledgeNode"]]:
        """Asynchronously retrieve top-k nodes from KnowledgeStore against a provided user query.

        Returns:
            A list of tuples where the first element represents the similarity score
            of the node to the query, and the second element is the node itself.
        """

    @abstractmethod
    async def delete_node(self, node_id: str) -> bool:
        """Asynchronously remove a node from the KnowledgeStore by ID, returning success status."""

    @abstractmethod
    async def clear(self) -> None:
        """Asynchronously clear all nodes from the KnowledgeStore."""

    @abstractmethod
    async def count(self) -> int:
        """Asynchronously return the number of nodes in the store."""

    @abstractmethod
    async def persist(self) -> None:
        """Asynchronously save the KnowledgeStore nodes to a permanent storage."""

    @abstractmethod
    async def load(self) -> None:
        """Asynchronously load the KnowledgeStore nodes from a permanent storage using `name`."""
