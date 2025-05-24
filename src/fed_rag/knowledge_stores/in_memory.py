"""In Memory Knowledge Store"""

from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, PrivateAttr, model_serializer
from typing_extensions import Self

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.data_structures.knowledge_node import KnowledgeNode
from fed_rag.exceptions.knowledge_stores import KnowledgeStoreNotFoundError
from fed_rag.knowledge_stores.mixins import ManagedMixin

DEFAULT_CACHE_DIR = ".fed_rag/data_cache/"
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

    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor::
        """Compute cosine similarity between two embeddings."""
        norm_a = torch.nn.functional.normalize(a, p=2, dim=1)
        norm_b = torch.nn.functional.normalize(b, p=2, dim=1)
        
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_tensor = torch.tensor(query_emb).to(device)
    if not torch.is_tensor(nodes["embeddings"]):
        nodes["embeddings"] = torch.tensor(nodes["embeddings"]).to(device)
    similarities = cos_sim(query_tensor,nodes["embeddings"])
     if similarities.device==device:
        similarities = similarities.to("cpu")
    similarities = similarities.tolist()[0]
    zipped = list(zip(nodes["node_ids"], similarities))
    #scores.sort(key=lambda tup: tup[1], reverse=True)
    sorted_similarities = sorted(zipped, key=lambda row: row[1],reverse=True)
    return sorted_similarities[:top_k]


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    """InMemoryKnowledgeStore Class."""

    cache_dir: str = Field(default=DEFAULT_CACHE_DIR)
    _data_storage: torch.Tensor
    _data: dict[str, KnowledgeNode] = PrivateAttr(default_factory=dict)
    _data_list: Dict[str, list] = PrivateAttr(default_factory=lambda: {
        "node_ids": [],  # Empty KnowledgeNode for key1
        "embeddings": []   # Empty KnowledgeNode for key2
    })

    @classmethod
    def from_nodes(cls, nodes: list[KnowledgeNode], **kwargs: Any) -> Self:
        instance = cls(**kwargs)
        instance.load_nodes(nodes)
        return instance

    def load_node(self, node: KnowledgeNode) -> None:
        if node.node_id not in self._data:
            self._data[node.node_id] = node
            self._data_list["node_ids"].append(node.node_id)
            self._data_list["embeddings"].append(node.embedding)

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for node in nodes:
            self.load_node(node)

    def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        all_nodes = list(self._data.values())
        node_ids_and_scores = _get_top_k_nodes(
            nodes=self._data_list, query_emb=query_emb, top_k=top_k
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

    @model_serializer(mode="wrap")
    def custom_model_dump(self, handler: Any) -> Dict[str, Any]:
        data = handler(self)
        data = cast(Dict[str, Any], data)
        # include _data in serialization
        if self._data:
            data["_data"] = self._data
        return data  # type: ignore[no-any-return]

    def persist(self) -> None:
        serialized_model = self.model_dump()
        data_values = list(serialized_model["_data"].values())

        parquet_table = pa.Table.from_pylist(data_values)

        filename = Path(self.cache_dir) / f"{self.name}.parquet"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(parquet_table, filename)

    def load(self) -> None:
        filename = Path(self.cache_dir) / f"{self.name}.parquet"
        if not filename.exists():
            msg = f"Knowledge store '{self.name}' not found at expected location: {filename}"
            raise KnowledgeStoreNotFoundError(msg)

        parquet_data = pq.read_table(filename).to_pylist()
        nodes = [KnowledgeNode(**data) for data in parquet_data]
        self.load_nodes(nodes)


class ManagedInMemoryKnowledgeStore(ManagedMixin, InMemoryKnowledgeStore):
    def persist(self) -> None:
        serialized_model = self.model_dump()
        data_values = list(serialized_model["_data"].values())

        parquet_table = pa.Table.from_pylist(data_values)

        filename = Path(self.cache_dir) / self.name / f"{self.ks_id}.parquet"
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(parquet_table, filename)

    @classmethod
    def from_name_and_id(
        cls, name: str, ks_id: str, cache_dir: str | None = None
    ) -> Self:
        cache_dir = cache_dir if cache_dir else DEFAULT_CACHE_DIR
        filename = Path(cache_dir) / name / f"{ks_id}.parquet"
        if not filename.exists():
            msg = f"Knowledge store '{name}/{ks_id}' not found at expected location: {filename}"
            raise KnowledgeStoreNotFoundError(msg)

        parquet_data = pq.read_table(filename).to_pylist()
        nodes = [KnowledgeNode(**data) for data in parquet_data]
        knowledge_store = ManagedInMemoryKnowledgeStore.from_nodes(
            nodes, name=name, cache_dir=cache_dir
        )
        # set id
        knowledge_store.ks_id = ks_id
        return knowledge_store
