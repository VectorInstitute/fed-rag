import pytest

from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
)
from fed_rag.data_structures import KnowledgeNode
from fed_rag.knowledge_stores.no_encode import (
    MCPKnowledgeSource,
    MCPKnowledgeStore,
)


def test_mcp_knowledge_store_class() -> None:
    names_of_base_classes = [b.__name__ for b in MCPKnowledgeStore.__mro__]
    assert BaseAsyncNoEncodeKnowledgeStore.__name__ in names_of_base_classes


@pytest.fixture
def mcp_source() -> MCPKnowledgeSource:
    return MCPKnowledgeSource(
        url="https://fake-mcp-url.io", tool_name="fake_tool"
    )


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
