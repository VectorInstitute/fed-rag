"""Base Knowledge Store"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fed_rag.utils.asyncio import asyncio_run

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

    class _SyncConvertedKnowledgeStore(BaseKnowledgeStore):
        """A nested class for converting this store to a sync version."""

        _async_ks: "BaseAsyncKnowledgeStore" = PrivateAttr()

        def __init__(self, async_ks: "BaseAsyncKnowledgeStore"):
            super().__init__(name=async_ks.name)
            self._async_ks = async_ks

            # Copy all fields from async store
            self._copy_async_ks_fields()

        def _copy_async_ks_fields(self) -> None:
            """Copy field definitions and values from async store."""
            for field_name, field_info in type(
                self._async_ks
            ).model_fields.items():
                # add field definition to model fields
                self.__class__.model_fields[field_name] = field_info

                # set fields
                if hasattr(self._async_ks, field_name):
                    value = getattr(self._async_ks, field_name)
                    setattr(self, field_name, value)

        def load_node(self, node: "KnowledgeNode") -> None:
            asyncio_run(self._async_ks.load_node(node))

        def load_nodes(self, nodes: list["KnowledgeNode"]) -> None:
            asyncio_run(self._async_ks.load_nodes(nodes))

        def retrieve(
            self, query_emb: list[float], top_k: int
        ) -> list[tuple[float, "KnowledgeNode"]]:
            return asyncio_run(self._async_ks.retrieve(query_emb=query_emb, top_k=top_k))  # type: ignore [no-any-return]

        def delete_node(self, node_id: str) -> bool:
            return asyncio_run(self._async_ks.delete_node(node_id))  # type: ignore [no-any-return]

        def clear(self) -> None:
            asyncio_run(self._async_ks.clear())

        @property
        def count(self) -> int:
            return self._async_ks.count

        def persist(self) -> None:
            self._async_ks.persist()

        def load(self) -> None:
            self._async_ks.load()

    def to_sync(self) -> BaseKnowledgeStore:
        """Convert this async knowledge store to a sync version."""

        return BaseAsyncKnowledgeStore._SyncConvertedKnowledgeStore(self)
