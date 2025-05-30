from unittest.mock import MagicMock, patch

from fed_rag.knowledge_stores.no_encode import MCPKnowledgeSource


@patch("fed_rag.knowledge_stores.no_encode.mcp.source.uuid")
def test_source_init(mock_uuid: MagicMock) -> None:
    mock_uuid.uuid4.return_value = "mock_uuid"
    mcp_source = MCPKnowledgeSource(
        url="https://fake_url", tool_name="fake_tool"
    )

    assert mcp_source.name == "source-mock_uuid"
    assert mcp_source.url == "https://fake_url"
    assert mcp_source.tool_name == "fake_tool"
    assert mcp_source._converter_fn is None
