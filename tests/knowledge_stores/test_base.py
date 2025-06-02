import inspect
from contextlib import nullcontext as does_not_raise

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

    @property
    def count(self) -> int:
        return len(self.nodes)

    def persist(self) -> None:
        pass

    def load(self) -> None:
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

    with does_not_raise():
        sync_store.load_nodes(nodes[1:])
        assert sync_store.nodes == nodes[1:]

        sync_store.retrieve([1, 2, 3], 1)
        sync_store.delete_node("fake id")  # doesn't actually delete
        sync_store.load_node(nodes[0])

        # no-ops
        sync_store.load()
        sync_store.persist()

        assert sync_store.count == len(nodes)

        sync_store.clear()
        assert sync_store.count == 0
