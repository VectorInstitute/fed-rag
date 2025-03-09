"""In Memory Knowledge Store"""

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    """InMemoryKnowledgeStore Class."""

    def load_node(self, node: KnowledgeNode) -> None:
        raise NotImplementedError

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        raise NotImplementedError

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[KnowledgeNode]:
        raise NotImplementedError

    def delete_node(self, node_id: str) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError
