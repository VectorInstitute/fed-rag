from unittest.mock import MagicMock, patch

import pytest
from mcp import StdioServerParameters
from mcp.types import CallToolResult, ImageContent, TextContent

from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import CallToolResultConversionError
from fed_rag.knowledge_stores.no_encode import MCPStdioKnowledgeSource
from fed_rag.knowledge_stores.no_encode.mcp.sources.utils import (
    default_converter,
)


@patch("fed_rag.knowledge_stores.no_encode.mcp.sources.stdio.uuid")
def test_source_init(mock_uuid: MagicMock) -> None:
    mock_uuid.uuid4.return_value = "mock_uuid"
    mcp_source = MCPStdioKnowledgeSource(
        server_params=StdioServerParameters(
            command="uv run",
            args=["fake.py"],
        ),
        tool_name="fake_tool",
        query_param_name="query",
        tool_call_kwargs={"param_1": 1, "param_2": "value 2"},
    )

    assert mcp_source.name == "source-stdio-mock_uuid"
    assert mcp_source.query_param_name == "query"
    assert mcp_source.server_params.command == "uv run"
    assert mcp_source.server_params.args == ["fake.py"]
    assert mcp_source.server_params.env is None
    assert mcp_source.tool_name == "fake_tool"
    assert mcp_source.tool_call_kwargs == {"param_1": 1, "param_2": "value 2"}
    assert mcp_source._converter_fn == default_converter


def test_source_init_with_fluent_style() -> None:
    mcp_source = (
        MCPStdioKnowledgeSource(
            server_params=StdioServerParameters(
                command="uv run",
                args=["fake.py"],
            ),
            tool_name="fake_tool",
            query_param_name="query",
        )
        .with_name("fake-name")
        .with_query_param_name("question")
        .with_tool_call_kwargs({"param": 1})
        .with_converter(
            lambda result: KnowledgeNode(
                text_content="fake text", node_type="text"
            )
        )
    )

    assert mcp_source.name == "fake-name"
    assert mcp_source.query_param_name == "question"
    assert mcp_source.server_params.command == "uv run"
    assert mcp_source.server_params.args == ["fake.py"]
    assert mcp_source.server_params.env is None
    assert mcp_source.tool_name == "fake_tool"
    assert mcp_source.tool_call_kwargs == {"param": 1}
    assert mcp_source._converter_fn != default_converter


def test_source_custom_convert() -> None:
    mcp_source = MCPStdioKnowledgeSource(
        server_params=StdioServerParameters(
            command="uv run",
            args=["fake.py"],
        ),
        tool_name="fake_tool",
        query_param_name="query",
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
    node = mcp_source.call_tool_result_to_knowledge_nodes_list(result=result)

    assert node.text_content == "fake text"


def test_source_default_convert() -> None:
    mcp_source = MCPStdioKnowledgeSource(
        server_params=StdioServerParameters(
            command="uv run",
            args=["fake.py"],
        ),
        tool_name="fake_tool",
        query_param_name="query",
    )

    # act
    content = [
        TextContent(text="text 1", type="text"),
        TextContent(text="text 2", type="text"),
        ImageContent(data="fakeimage", mimeType="image/png", type="image"),
    ]
    result = CallToolResult(content=content)
    nodes = mcp_source.call_tool_result_to_knowledge_nodes_list(result=result)

    nodes_from_default = default_converter(
        result, metadata=mcp_source.model_dump()
    )

    assert len(nodes) == len(nodes_from_default)
    assert all(
        x.text_content == y.text_content
        for x, y in zip(nodes, nodes_from_default)
    )


def test_source_default_convert_raises_error() -> None:
    mcp_source = MCPStdioKnowledgeSource(
        server_params=StdioServerParameters(
            command="uv run",
            args=["fake.py"],
        ),
        tool_name="fake_tool",
        query_param_name="query",
    )

    # act
    result = CallToolResult(content=[], isError=True)

    with pytest.raises(
        CallToolResultConversionError,
        match="Cannot convert a `CallToolResult` with `isError` set to `True`.",
    ):
        _ = mcp_source.call_tool_result_to_knowledge_nodes_list(result=result)


def test_mutate_server_params() -> None:
    mcp_source = MCPStdioKnowledgeSource(
        server_params=StdioServerParameters(
            command="uv run",
            args=["fake.py"],
        ),
        tool_name="fake_tool",
        query_param_name="query",
    )

    mcp_source.with_server_params(
        StdioServerParameters(
            command="poetry run",
            args=["fake.py"],
        )
    )

    assert mcp_source.server_params.command == "poetry run"
