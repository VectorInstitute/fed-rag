"""MCP Knowledge Store"""

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from typing_extensions import Self

from fed_rag.base.no_encode_knowledge_store import (
    BaseAsyncNoEncodeKnowledgeStore,
)
from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import KnowledgeStoreError

from .source import MCPStreamableHttpKnowledgeSource

DEFAULT_SCORE = 1.0
DEFAULT_KNOWLEDGE_STORE_NAME = "default-mcp"


class MCPKnowledgeStore(BaseAsyncNoEncodeKnowledgeStore):
    """MCP Knowledge Store.

    Retrieve knowledge from attached MCP servers.
    """

    name: str = DEFAULT_KNOWLEDGE_STORE_NAME
    sources: dict[str, MCPStreamableHttpKnowledgeSource]

    def __init__(self, sources: list[MCPStreamableHttpKnowledgeSource] = []):
        sources_dict = {s.name: s for s in sources}
        super().__init__(sources=sources_dict)

    def add_source(self, source: MCPStreamableHttpKnowledgeSource) -> Self:
        """Add a source to knowledge store.

        Support fluent chaining.
        """

        if source.name in self.sources:
            raise KnowledgeStoreError(
                f"A source with the same name, {source.name}, already exists."
            )

        self.sources[source.name] = source
        return self

    async def _retrieve_from_source(
        self, query: str, source_id: str
    ) -> KnowledgeNode:
        source = self.sources[source_id]

        # Connect to a streamable HTTP server
        async with streamablehttp_client(source.url) as (
            read_stream,
            write_stream,
            _,
        ):
            # Create a session using the client streams
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()
                # Call a tool
                tool_result = await session.call_tool(
                    source.tool_name, {"message": query}
                )

        return source.call_tool_result_to_knowledge_node(tool_result)

    async def retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        """Retrieve from all MCP knowledge sources.

        Args:
            query (str): query to send to each MCP source
            top_k (int): number of nodes to retrieve

        Returns:
            list[tuple[float, KnowledgeNode]]: _description_
        """
        knowledge_nodes: list[KnowledgeNode] = []
        for source_id in self.sources.keys():
            knowledge_node = await self._retrieve_from_source(query, source_id)
            knowledge_nodes.append(knowledge_node)

        # TODO: apply smarter logic here to only share top k nodes
        # perhaps use re-rankers?
        return [(DEFAULT_SCORE, node) for node in knowledge_nodes[:top_k]]

    # Not implemented methods
    async def load_node(self, node: KnowledgeNode) -> None:
        raise NotImplementedError(
            "load_node is not implemented for MCPKnowledgeStore."
        )

    async def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        raise NotImplementedError(
            "load_nodes is not implemented for MCPKnowledgeStore."
        )

    async def delete_node(self, node_id: str) -> None:
        raise NotImplementedError(
            "delete_node is not implemented for MCPKnowledgeStore."
        )

    async def clear(self) -> None:
        raise NotImplementedError(
            "clear is not implemented for MCPKnowledgeStore."
        )

    @property
    def count(self) -> int:
        raise NotImplementedError(
            "count is not implemented for MCPKnowledgeStore."
        )

    def persist(self) -> None:
        raise NotImplementedError(
            "persist is not implemented for MCPKnowledgeStore."
        )

    def load(self) -> None:
        raise NotImplementedError(
            "load is not implemented for MCPKnowledgeStore."
        )
