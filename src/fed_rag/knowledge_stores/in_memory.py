"""In Memory Knowledge Store"""

import gc
from pathlib import Path
from typing import Any, Dict, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import Field, PrivateAttr, model_serializer
from typing_extensions import Self

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.data_structures.knowledge_node import KnowledgeNode
from fed_rag.exceptions.knowledge_stores import KnowledgeStoreNotFoundError
from fed_rag.knowledge_stores.mixins import ManagedMixin

DEFAULT_CACHE_DIR = ".fed_rag/data_cache/"
DEFAULT_TOP_K = 2


def _get_top_k_nodes(
    nodes: list[str],
    embeddings: torch.Tensor,
    query_emb: torch.Tensor,
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, float]]:
    """Retrieves the top-k similar nodes against query.

    Returns:
        list[tuple[float, str]] — the node_ids and similarity scores of top-k nodes
    """

    def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two embeddings."""
        a = a.unsqueeze(0) if a.dim() == 1 else a
        b = b.unsqueeze(0) if b.dim() == 1 else b
        norm_a = torch.nn.functional.normalize(a, p=2, dim=1)
        norm_b = torch.nn.functional.normalize(b, p=2, dim=1)

        return torch.mm(norm_a, norm_b.transpose(0, 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    similarities = cosine_sim(query_emb, embeddings)
    if similarities.device == device:
        similarities = similarities.to("cpu")
    similarities = similarities.tolist()[0]
    zipped = list(zip(nodes, similarities))
    sorted_similarities = sorted(zipped, key=lambda row: row[1], reverse=True)
    return sorted_similarities[:top_k]


class InMemoryKnowledgeStore(BaseKnowledgeStore):
    """InMemoryKnowledgeStore Class."""

    cache_dir: str = Field(default=DEFAULT_CACHE_DIR)
    _data: dict[str, KnowledgeNode] = PrivateAttr(default_factory=dict)
    _data_storage: list[float] = PrivateAttr(default_factory=list)
    _node_list: list[str] = PrivateAttr(default_factory=list)

    @classmethod
    def from_nodes(cls, nodes: list[KnowledgeNode], **kwargs: Any) -> Self:
        instance = cls(**kwargs)
        instance.load_nodes(nodes)
        return instance

    def load_node(self, node: KnowledgeNode) -> None:
        if isinstance(self._data_storage, torch.Tensor):
            device = torch.device("cpu")
            self._data_storage = self._data_storage.to(device).tolist()
            gc.collect()  # Clean up Python garbage
            torch.cuda.empty_cache()
        if node.node_id not in self._data:
            self._data[node.node_id] = node
            self._node_list.append(node.node_id)
            self._data_storage.append(node.embedding)

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for node in nodes:
            self.load_node(node)

    def retrieve(
        self, query_emb: list[float], top_k: int = DEFAULT_TOP_K
    ) -> list[tuple[float, KnowledgeNode]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_emb = torch.tensor(query_emb).to(device)
        if not torch.is_tensor(self._data_storage):
            self._data_storage = torch.tensor(self._data_storage).to(device)
        node_ids_and_scores = _get_top_k_nodes(
            nodes=self._node_list,
            embeddings=self._data_storage,
            query_emb=query_emb,
            top_k=top_k,
        )
        return [(el[1], self._data[el[0]]) for el in node_ids_and_scores]

    def batch_retrieve(
        self, query_embs: list[list[float]], top_k: int = DEFAULT_TOP_K
    ) -> list[list[tuple[float, "KnowledgeNode"]]]:
        raise NotImplementedError(
            f"batch_retrieve is not implemented for {self.__class__.__name__}."
        )

    def delete_node(self, node_id: str) -> bool:
        if isinstance(self._data_storage, torch.Tensor):
            device = torch.device("cpu")
            self._data_storage = self._data_storage.to(device).tolist()
            gc.collect()  # Clean up Python garbage
            torch.cuda.empty_cache()
        if node_id in self._data:
            del self._data[node_id]
            for i in range(len(self._node_list)):
                print(len(self._node_list))
                if node_id == self._node_list[i]:
                    del self._data_storage[i]
                    del self._node_list[i]
                    break
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
