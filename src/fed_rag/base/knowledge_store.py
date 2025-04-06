"""Base Knowledge Store"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from fed_rag.types.knowledge_node import KnowledgeNode

DEFAULT_KNOWLEDGE_STORE_NAME = "default"


class BaseKnowledgeStore(BaseModel, ABC):
    """Base Knowledge Store Class."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        description="Name of Knowledge Store used for caching and loading.",
        default=DEFAULT_KNOWLEDGE_STORE_NAME,
    )

    @abstractmethod
    def load_node(self, node: KnowledgeNode) -> None:
        """Load a KnowledgeNode into the KnowledgeStore."""

    @abstractmethod
    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        """Load multiple KnowledgeNodes in batch."""

    @abstractmethod
    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
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
        """
        Load the KnowledgeStore nodes from a permanent storage using `name`.

        Args:
            ks_id: The id of the knowledge store to load.
        """
