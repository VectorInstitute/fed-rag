import inspect

from fed_rag.base.knowledge_store import BaseKnowledgeStore


def test_base_abstract_attr() -> None:
    abstract_methods = BaseKnowledgeStore.__abstractmethods__

    assert inspect.isabstract(BaseKnowledgeStore)
    assert "load_node" in abstract_methods
    assert "load_nodes" in abstract_methods
    assert "retrieve" in abstract_methods
    assert "delete_node" in abstract_methods
    assert "clear" in abstract_methods
    assert "count" in abstract_methods
