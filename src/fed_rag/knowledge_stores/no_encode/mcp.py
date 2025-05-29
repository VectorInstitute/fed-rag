"""MCP Knowledge Store"""


from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
)
from fed_rag.data_structures import MCPServerMetadata


class MCPKnowledgeStore(BaseAsyncNoEncodeKnowledgeStore):
    """MCP Knowledge Store.

    Retrieve knowledge from attached MCP servers.
    """

    servers: list[MCPServerMetadata]
