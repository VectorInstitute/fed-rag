"""In Memory Knowledge Store"""

import numpy as np
from pydantic import PrivateAttr
from typing_extensions import Self

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode, NodeType

DEFAULT_TOP_K = 2


def _get_top_k_nodes(
    nodes: list[KnowledgeNode],
    query_emb: list[float],
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, float]]:
    """Retrieves the top-k similar nodes against query.

    Returns:
        list[tuple[float, str]] — the node_ids and similarity scores of top-k nodes
    """

    def cosine_sim(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        np_a = np.array(a)
        np_b = np.array(b)
        cosine_sim: float = np.dot(np_a, np_b) / (
            np.linalg.norm(np_a) * np.linalg.norm(np_b)
        )
        return cosine_sim

    scores = [
        (node.node_id, cosine_sim(node.embedding, query_emb)) for node in nodes
    ]
    scores.sort(key=lambda tup: tup[1], reverse=True)
    return scores[:top_k]


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    """InMemoryKnowledgeStore Class."""

    _data: dict[str, KnowledgeNode] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_nodes(cls, nodes: list[KnowledgeNode]) -> Self:
        instance = cls()
        instance.load_nodes(nodes)
        return instance

    def load_node(self, node: KnowledgeNode) -> None:
        if node.node_id not in self._data:
            self._data[node.node_id] = node

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for node in nodes:
            self.load_node(node)

    def persist(
        self,
        embedding: list[float],
        node_type: NodeType,
        text_content: str | None,
        image_content: bytes | None,
    ) -> None:
        if node_type == NodeType.TEXT and text_content is None:
            raise ValueError("text_content is required for NodeType.TEXT")
        if node_type == NodeType.IMAGE and image_content is None:
            raise ValueError("image_content is required for NodeType.IMAGE")
        if node_type == NodeType.MULTIMODAL and (
            text_content is None or image_content is None
        ):
            raise ValueError(
                "text_content and image_content are required for NodeType.MULTIMODAL"
            )

        node = KnowledgeNode(embedding=embedding, node_type=node_type)
        if node_type == NodeType.TEXT:
            node.text_content = text_content
        elif node_type == NodeType.IMAGE:
            node.image_content = image_content
        elif node_type == NodeType.MULTIMODAL:
            node.text_content = text_content
            node.image_content = image_content

        self.load_node(node)

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        all_nodes = list(self._data.values())
        node_ids_and_scores = _get_top_k_nodes(
            nodes=all_nodes, query_emb=query_emb, top_k=top_k
        )
        return [(el[1], self._data[el[0]]) for el in node_ids_and_scores]

    def delete_node(self, node_id: str) -> bool:
        if node_id in self._data:
            del self._data[node_id]
            return True
        else:
            return False

    def clear(self) -> None:
        self._data = {}

    @property
    def count(self) -> int:
        return len(self._data)
