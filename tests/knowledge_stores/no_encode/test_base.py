import inspect

import pytest

from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
    BaseNoEncodeKnowledgeStore,
)
from fed_rag.data_structures.knowledge_node import KnowledgeNode, NodeType


def test_base_abstract_attr() -> None:
    abstract_methods = BaseNoEncodeKnowledgeStore.__abstractmethods__

    assert inspect.isabstract(BaseNoEncodeKnowledgeStore)
    assert "load_node" in abstract_methods
    assert "load_nodes" in abstract_methods
    assert "retrieve" in abstract_methods
    assert "delete_node" in abstract_methods
    assert "clear" in abstract_methods
    assert "count" in abstract_methods
    assert "persist" in abstract_methods
    assert "load" in abstract_methods


def test_base_async_abstract_attr() -> None:
    abstract_methods = BaseAsyncNoEncodeKnowledgeStore.__abstractmethods__

    assert inspect.isabstract(BaseAsyncNoEncodeKnowledgeStore)
    assert "load_node" in abstract_methods
    assert "retrieve" in abstract_methods
    assert "delete_node" in abstract_methods
    assert "clear" in abstract_methods
    assert "count" in abstract_methods
    assert "persist" in abstract_methods
    assert "load" in abstract_methods


@pytest.mark.asyncio
async def test_base_async_load_nodes() -> None:
    # create a dummy store
    class DummyAsyncKnowledgeStore(BaseAsyncNoEncodeKnowledgeStore):
        nodes: list[KnowledgeNode] = []

        async def load_node(self, node: KnowledgeNode) -> None:
            self.nodes.append(node)

        async def retrieve(
            self, query: str, top_k: int
        ) -> list[tuple[float, KnowledgeNode]]:
            return [(ix, n) for ix, n in enumerate(self.nodes[:top_k])]

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

    dummy_store = DummyAsyncKnowledgeStore()
    nodes = [
        KnowledgeNode(node_type=NodeType.TEXT, text_content="Dummy text")
        for _ in range(5)
    ]

    await dummy_store.load_nodes(nodes)
    res = await dummy_store.retrieve("mock query", top_k=2)

    assert dummy_store.nodes == nodes
    assert len(res) == 2
