import pytest

from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode


@pytest.fixture
def text_nodes() -> list[KnowledgeNode]:
    return [
        KnowledgeNode(
            embedding=[1.0, 0.0, 1.0], node_type="text", text_content="node 1"
        ),
        KnowledgeNode(
            embedding=[1.0, 0.0, 0.0], node_type="text", text_content="node 2"
        ),
        KnowledgeNode(
            embedding=[
                1.0,
                1.0,
                0.0,
            ],
            node_type="text",
            text_content="node 3",
        ),
    ]


def test_in_memory_knowledge_store_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in InMemoryKnowledgeStore.__mro__
    ]
    assert BaseKnowledgeStore.__name__ in names_of_base_classes


def test_in_memory_knowledge_store_init() -> None:
    knowledge_store = InMemoryKnowledgeStore()

    assert knowledge_store.count == 0


def test_from_nodes(text_nodes: list[KnowledgeNode]) -> None:
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=text_nodes)

    assert knowledge_store.count == 3
    assert all(n.node_id in knowledge_store._data for n in text_nodes)


def test_delete_node(text_nodes: list[KnowledgeNode]) -> None:
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=text_nodes)

    assert knowledge_store.count == 3

    res = knowledge_store.delete_node(text_nodes[0].node_id)

    assert res is True
    assert knowledge_store.count == 2
    assert text_nodes[0].node_id not in knowledge_store._data


def test_delete_node_returns_false(text_nodes: list[KnowledgeNode]) -> None:
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=text_nodes)

    assert knowledge_store.count == 3

    res = knowledge_store.delete_node("non_included_id")

    assert res is False
    assert knowledge_store.count == 3
    assert all(n.node_id in knowledge_store._data for n in text_nodes)


def test_load_node(text_nodes: list[KnowledgeNode]) -> None:
    knowledge_store = InMemoryKnowledgeStore()
    assert knowledge_store.count == 0

    knowledge_store.load_node(text_nodes[-1])

    assert knowledge_store.count == 1
    assert text_nodes[-1].node_id in knowledge_store._data


def test_load_nodes(text_nodes: list[KnowledgeNode]) -> None:
    knowledge_store = InMemoryKnowledgeStore()
    assert knowledge_store.count == 0

    knowledge_store.load_nodes(text_nodes)

    assert knowledge_store.count == 3
    assert all(n.node_id in knowledge_store._data for n in text_nodes)


def test_clear(text_nodes: list[KnowledgeNode]) -> None:
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=text_nodes)
    assert knowledge_store.count == 3

    knowledge_store.clear()

    assert knowledge_store.count == 0
    assert all(n.node_id not in knowledge_store._data for n in text_nodes)


@pytest.mark.parametrize(
    ("query_emb", "top_k", "expected_node_ix"),
    [([1.0, 1.0, 1.0], 2, [0, 2]), ([0.5, 0.0, 0.0], 1, [1])],
    ids=[str([1.0, 1.0, 1.0]), str([0.5, 0.0, 0.0])],
)
def test_retrieve(
    query_emb: list[float],
    top_k: int,
    expected_node_ix: list[int],
    text_nodes: list[KnowledgeNode],
) -> None:
    # arrange
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=text_nodes)

    # act
    res = knowledge_store.retrieve(query_emb, top_k=top_k)

    # assert
    assert [el[1] for el in res] == [text_nodes[ix] for ix in expected_node_ix]
