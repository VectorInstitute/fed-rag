import inspect

from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
    BaseNoEncodeKnowledgeStore,
)


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
