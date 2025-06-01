import uuid
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult
from pydantic import Field

from .base import BaseMCPKnowledgeSource
from .utils import CallToolResultConverter, default_converter


class MCPStreamableHttpKnowledgeSource(BaseMCPKnowledgeSource):
    """The MCPStreamableHttpKnowledgeSource class.

    Users can easily connect MCP tools as their source of knowledge in RAG systems
    via the Streamable Http transport.
    """

    url: str = Field("Url for streamable http MCP server.")

    def __init__(
        self,
        url: str,
        tool_name: str,
        query_param_name: str,
        tool_call_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        converter_fn: CallToolResultConverter | None = None,
    ):
        name = name or f"source-{str(uuid.uuid4())}"
        tool_call_kwargs = tool_call_kwargs or {}
        super().__init__(
            name=name,
            url=url,
            tool_name=tool_name,
            query_param_name=query_param_name,
            tool_call_kwargs=tool_call_kwargs,
        )
        self._converter_fn = converter_fn or default_converter

    async def retrieve(self, query: str) -> CallToolResult:
        # Connect to a streamable HTTP server
        async with streamablehttp_client(self.url) as (
            read_stream,
            write_stream,
            _,
        ):
            # Create a session using the client streams
            async with ClientSession(read_stream, write_stream) as session:
                # Initialize the connection
                await session.initialize()
                # Call a tool
                tool_arguments = {
                    self.query_param_name: query,
                    **self.tool_call_kwargs,
                }
                tool_result = await session.call_tool(
                    self.tool_name, arguments=tool_arguments
                )
        return tool_result
