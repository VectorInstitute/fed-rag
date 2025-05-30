import inspect

from fed_rag.base.mcp_knowledge_source import BaseMCPKnowledgeSource


def test_base_abstract_attr() -> None:
    abstract_methods = BaseMCPKnowledgeSource.__abstractmethods__

    assert inspect.isabstract(BaseMCPKnowledgeSource)
    assert (
        "read_resource_result_to_knowledge_store_retrieval_result"
        in abstract_methods
    )
    assert (
        "call_tool_result_to_knowledge_store_retrieval_result"
        in abstract_methods
    )
