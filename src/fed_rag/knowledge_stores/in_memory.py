"""In Memory Knowledge Store"""

from pydantic import PrivateAttr

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode

DEFAULT_TOP_K = 2


def _get_top_k_nodes(
    nodes: list[KnowledgeNode],
    query_emb: list[float],
    top_k: int = DEFAULT_TOP_K,
) -> list[KnowledgeNode]:
    return []


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    """InMemoryKnowledgeStore Class."""

    _data: dict[str, KnowledgeNode] = PrivateAttr(default_factory=dict)

    def load_node(self, node: KnowledgeNode) -> None:
        if node.node_id not in self._data:
            self._data[node.node_id] = node

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for node in nodes:
            self.load_node(node)

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[KnowledgeNode]:
        all_nodes = list(self._data.values())
        return _get_top_k_nodes(
            nodes=all_nodes, query_emb=query_emb, top_k=top_k
        )

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
