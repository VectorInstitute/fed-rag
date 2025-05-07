"""Qdrant Knowledge Store"""

from typing import Any

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.knowledge_stores.qdrant.utils import check_qdrant_installed
from fed_rag.types.knowledge_node import KnowledgeNode


class QdrantKnowledgeStore(BaseKnowledgeStore):
    """Qdrant Knowledge Store Class"""

    def __init__(self, *args: Any, **kwargs: Any):
        check_qdrant_installed()
        super().__init__(*args, **kwargs)

    def load_node(self, node: KnowledgeNode) -> None:
        raise NotImplementedError

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        raise NotImplementedError

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        raise NotImplementedError

    def delete_node(self, node_id: str) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    @property
    def count(self) -> int:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError
