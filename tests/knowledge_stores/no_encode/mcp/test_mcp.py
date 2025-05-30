from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
)
from fed_rag.knowledge_stores.no_encode import MCPKnowledgeStore


def test_mcp_knowledge_store_class() -> None:
    names_of_base_classes = [b.__name__ for b in MCPKnowledgeStore.__mro__]
    assert BaseAsyncNoEncodeKnowledgeStore.__name__ in names_of_base_classes
