"""Base Knowledge Store"""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from fed_rag.types.knowledge_node import KnowledgeNode, NodeType


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
    def persist(
        self,
        embedding: list[float],
        node_type: NodeType,
        text_content: str | None,
        image_content: bytes | None,
    ) -> None:
        """Persist an embedding into the KnowledgeStore.

        If note_type == NodeType.TEXT, then text_content is required.
        If note_type == NodeType.IMAGE, then image_content is required.
        If note_type == NodeType.MULTIMODAL, then text_content and image_content are required.

        Args:
            embedding (list[float]): The embedding to persist.
            node_type (NodeType): The type of node to persist.
            text_content (str | None): The text content to persist.
            image_content (bytes | None): The image content to persist.
        """

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
