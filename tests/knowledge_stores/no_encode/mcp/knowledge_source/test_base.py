import inspect

from fed_rag.base.mcp_knowledge_source import BaseMCPKnowledgeSource


def test_base_abstract_attr() -> None:
    abstract_methods = BaseMCPKnowledgeSource.__abstractmethods__

    assert inspect.isabstract(BaseMCPKnowledgeSource)
    assert "call_tool_result_to_knowledge_node" in abstract_methods
