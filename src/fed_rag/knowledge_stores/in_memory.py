"""In Memory Knowledge Store"""

import json
import uuid
from pathlib import Path
from typing import ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, PrivateAttr
from typing_extensions import Self

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode

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

    default_save_path: ClassVar[str] = ".fed_rag/data_cache/{0}.parquet"

    ks_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    _data: dict[str, KnowledgeNode] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_nodes(
        cls, nodes: list[KnowledgeNode], ks_id: str | None = None
    ) -> Self:
        instance = cls()
        instance.load_nodes(nodes)
        if ks_id:
            instance.ks_id = ks_id
        return instance

    def load_node(self, node: KnowledgeNode) -> None:
        if node.node_id not in self._data:
            self._data[node.node_id] = node

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for node in nodes:
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

    def persist(self) -> None:
        node_data = []
        for node in self._data.values():
            data = node.model_dump()
            data["metadata"] = json.dumps(data["metadata"])
            node_data.append(data)

        table = pa.Table.from_pylist(node_data)

        filename = self.__class__.default_save_path.format(self.ks_id)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, filename)

    @classmethod
    def load(cls, ks_id: str) -> Self:
        filename = cls.default_save_path.format(ks_id)
        parquet_data = pq.read_table(filename).to_pylist()

        nodes = []
        for data in parquet_data:
            data["metadata"] = json.loads(data["metadata"])
            nodes.append(KnowledgeNode(**data))

        return cls.from_nodes(nodes, ks_id)
