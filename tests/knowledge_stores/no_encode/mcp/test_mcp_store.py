from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, ImageContent, TextContent

from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
)
from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import KnowledgeStoreError
from fed_rag.knowledge_stores.no_encode import (
    MCPKnowledgeSource,
    MCPKnowledgeStore,
)


@pytest.fixture
def mcp_source() -> MCPKnowledgeSource:
    return MCPKnowledgeSource(
        url="https://fake-mcp-url.io", tool_name="fake_tool"
    )


@pytest.fixture
def call_tool_result() -> CallToolResult:
    content = [
        TextContent(text="text 1", type="text"),
        TextContent(text="text 2", type="text"),
        ImageContent(data="fakeimage", mimeType="image/png", type="image"),
    ]
    return CallToolResult(content=content)


def test_mcp_knowledge_store_class() -> None:
    names_of_base_classes = [b.__name__ for b in MCPKnowledgeStore.__mro__]
    assert BaseAsyncNoEncodeKnowledgeStore.__name__ in names_of_base_classes


def test_mcp_knowledge_store_init(mcp_source: MCPKnowledgeSource) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    assert store.name == "default-mcp"
    assert store.sources == {mcp_source.name: mcp_source}


@pytest.mark.asyncio
@patch("fed_rag.knowledge_stores.no_encode.mcp.store.ClientSession")
@patch("fed_rag.knowledge_stores.no_encode.mcp.store.streamablehttp_client")
async def test_mcp_knowledge_store_retrieve(
    mock_streamable_client: AsyncMock,
    mock_session_class: AsyncMock,
    mcp_source: MCPKnowledgeSource,
    call_tool_result: CallToolResult,
) -> None:
    # arrange mocks
    mock_session_instance = AsyncMock()
    mock_session_instance.initialize = AsyncMock()
    mock_session_instance.call_tool = AsyncMock(return_value=call_tool_result)

    # mock context managers
    mock_session_class.return_value.__aenter__ = AsyncMock(
        return_value=mock_session_instance
    )
    mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

    mock_streamable_client.return_value.__aenter__ = AsyncMock(
        return_value=(None, None, None)
    )
    mock_streamable_client.return_value.__aexit__ = AsyncMock(
        return_value=None
    )

    mcp_source.with_converter(
        lambda result: KnowledgeNode(
            text_content="fake text", node_type="text"
        )
    )
    store = MCPKnowledgeStore().add_source(mcp_source)
    result = await store.retrieve(query="mock_query", top_k=10)

    assert result[0][0] == 1.0  # Default similiarity score
    assert result[0][1].text_content == "fake text"
    mock_session_instance.call_tool.assert_called_once_with(
        mcp_source.tool_name, {"message": "mock_query"}
    )


def test_add_source_raises_error_with_existing_name(
    mcp_source: MCPKnowledgeSource,
) -> None:
    with pytest.raises(
        KnowledgeStoreError,
        match=f"A source with the same name, {mcp_source.name}, already exists.",
    ):
        _ = MCPKnowledgeStore().add_source(mcp_source).add_source(mcp_source)


# test not implemented
@pytest.mark.asyncio
async def test_load_node_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        await store.load_node(
            KnowledgeNode(node_type="text", text_content="text 1")
        )


@pytest.mark.asyncio
async def test_load_nodes_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        await store.load_nodes(
            [KnowledgeNode(node_type="text", text_content="text 1")]
        )


@pytest.mark.asyncio
async def test_delete_node_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        await store.delete_node("")


@pytest.mark.asyncio
async def test_clear_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        await store.clear()


def test_count_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        store.count


def test_persist_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        store.persist()


def test_load_raises_not_implemented_error(
    mcp_source: MCPKnowledgeSource,
) -> None:
    store = MCPKnowledgeStore().add_source(mcp_source)

    with pytest.raises(NotImplementedError):
        store.load()
