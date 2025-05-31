from unittest.mock import MagicMock, patch

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import CallToolResultConversionError
from fed_rag.knowledge_stores.no_encode import MCPStreamableHttpKnowledgeSource
from fed_rag.knowledge_stores.no_encode.mcp.sources.utils import (
    default_converter,
)


@patch("fed_rag.knowledge_stores.no_encode.mcp.sources.streamable_http.uuid")
def test_source_init(mock_uuid: MagicMock) -> None:
    mock_uuid.uuid4.return_value = "mock_uuid"
    mcp_source = MCPStreamableHttpKnowledgeSource(
        url="https://fake_url", tool_name="fake_tool"
    )

    assert mcp_source.name == "source-mock_uuid"
    assert mcp_source.url == "https://fake_url"
    assert mcp_source.tool_name == "fake_tool"
    assert mcp_source._converter_fn == default_converter


def test_source_init_with_fluent_style() -> None:
    mcp_source = (
        MCPStreamableHttpKnowledgeSource(
            url="https://fake_url", tool_name="fake_tool"
        )
        .with_name("fake-name")
        .with_converter(
            lambda result: KnowledgeNode(
                text_content="fake text", node_type="text"
            )
        )
    )

    assert mcp_source.name == "fake-name"
    assert mcp_source.url == "https://fake_url"
    assert mcp_source.tool_name == "fake_tool"
    assert mcp_source._converter_fn is not None


def test_source_custom_convert() -> None:
    mcp_source = MCPStreamableHttpKnowledgeSource(
        url="https://fake_url", tool_name="fake_tool"
    )
    mcp_source.with_converter(
        lambda result, metadata: KnowledgeNode(
            text_content="fake text", node_type="text"
        )
    )

    # act
    content = [
        TextContent(text="text 1", type="text"),
        TextContent(text="text 2", type="text"),
        ImageContent(data="fakeimage", mimeType="image/png", type="image"),
    ]
    result = CallToolResult(content=content)
    node = mcp_source.call_tool_result_to_knowledge_node(result=result)

    assert node.text_content == "fake text"


def test_source_default_convert() -> None:
    mcp_source = MCPStreamableHttpKnowledgeSource(
        url="https://fake_url", tool_name="fake_tool"
    )

    # act
    content = [
        TextContent(text="text 1", type="text"),
        TextContent(text="text 2", type="text"),
        ImageContent(data="fakeimage", mimeType="image/png", type="image"),
    ]
    result = CallToolResult(content=content)
    node = mcp_source.call_tool_result_to_knowledge_node(result=result)

    node_from_default = default_converter(
        result, metadata=mcp_source.model_dump()
    )

    assert node.text_content == node_from_default.text_content
    assert node.metadata == node_from_default.metadata


def test_source_default_convert_raises_error() -> None:
    mcp_source = MCPStreamableHttpKnowledgeSource(
        url="https://fake_url", tool_name="fake_tool"
    )

    # act
    result = CallToolResult(content=[], isError=True)

    with pytest.raises(
        CallToolResultConversionError,
        match="Cannot convert a `CallToolResult` with `isError` set to `True`.",
    ):
        _ = mcp_source.call_tool_result_to_knowledge_node(result=result)
