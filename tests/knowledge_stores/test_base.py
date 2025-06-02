import inspect

import pytest

from fed_rag.base.knowledge_store import (
    BaseAsyncKnowledgeStore,
    BaseKnowledgeStore,
)
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType


def test_base_abstract_attr() -> None:
    abstract_methods = BaseKnowledgeStore.__abstractmethods__

    assert inspect.isabstract(BaseKnowledgeStore)
    assert "load_node" in abstract_methods
    assert "load_nodes" in abstract_methods
    assert "retrieve" in abstract_methods
    assert "delete_node" in abstract_methods
    assert "clear" in abstract_methods
    assert "count" in abstract_methods
    assert "persist" in abstract_methods
    assert "load" in abstract_methods


def test_base_async_abstract_attr() -> None:
    abstract_methods = BaseAsyncKnowledgeStore.__abstractmethods__

    assert inspect.isabstract(BaseAsyncKnowledgeStore)
    assert "load_node" in abstract_methods
    assert "retrieve" in abstract_methods
    assert "delete_node" in abstract_methods
    assert "clear" in abstract_methods
    assert "count" in abstract_methods
    assert "persist" in abstract_methods
    assert "load" in abstract_methods


# create a dummy store
class DummyAsyncKnowledgeStore(BaseAsyncKnowledgeStore):
    nodes: list[KnowledgeNode] = []

    async def load_node(self, node: KnowledgeNode) -> None:
        self.nodes.append(node)

    async def retrieve(
        self, query_emb: list[float], top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        return []

    async def delete_node(self, node_id: str) -> bool:
        return True

    async def clear(self) -> None:
        self.nodes.clear()

    async def count(self) -> int:
        return len(self.nodes)

    async def persist(self) -> None:
        pass

    async def load(self) -> None:
        pass


@pytest.mark.asyncio
async def test_base_async_load_nodes() -> None:
    dummy_store = DummyAsyncKnowledgeStore()
    nodes = [
        KnowledgeNode(node_type=NodeType.TEXT, text_content="Dummy text")
        for _ in range(5)
    ]

    await dummy_store.load_nodes(nodes)
    assert dummy_store.nodes == nodes


def test_to_sync_methods() -> None:
    dummy_store = DummyAsyncKnowledgeStore()
    sync_store = dummy_store.to_sync()

    nodes = [
        KnowledgeNode(node_type=NodeType.TEXT, text_content="Dummy text")
        for _ in range(5)
    ]

    sync_store.load_nodes(nodes)
    assert sync_store.nodes == nodes
