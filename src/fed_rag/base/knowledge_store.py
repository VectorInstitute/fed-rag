"""Base Knowledge Store"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from fed_rag.types.knowledge_node import KnowledgeNode


class BaseKnowledgeStore(BaseModel, ABC):
    """Base Knowledge Store Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def load_node(self, node: KnowledgeNode) -> None:
        """Load a KnowledgeNode into the KnowledgeStore."""

    @abstractmethod
    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        """Load multiple KnowledgeNodes in batch."""

    @abstractmethod
    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[KnowledgeNode]:
        """Retrieve top-k nodes from KnowledgeStore against a provided user query."""

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
