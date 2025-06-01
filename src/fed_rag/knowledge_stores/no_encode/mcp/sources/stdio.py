"""MCP Knowledge Source with stdio Transport"""

import uuid
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from typing_extensions import Self

from .base import BaseMCPKnowledgeSource
from .utils import CallToolResultConverter, default_converter


class MCPStdioKnowledgeSource(BaseMCPKnowledgeSource):
    """The MCPStdioKnowledgeSource class.

    Users can easily connect MCP tools as their source of knowledge in RAG systems
    via the stdio transport.
    """

    server_params: StdioServerParameters

    def __init__(
        self,
        server_params: StdioServerParameters,
        tool_name: str,
        query_param_name: str,
        tool_call_kwargs: dict[str, Any] | None = None,
        name: str | None = None,
        converter_fn: CallToolResultConverter | None = None,
    ):
        name = name or f"source-stdio-{str(uuid.uuid4())}"
        tool_call_kwargs = tool_call_kwargs or {}
        super().__init__(
            name=name,
            server_params=server_params,
            tool_name=tool_name,
            query_param_name=query_param_name,
            tool_call_kwargs=tool_call_kwargs,
        )
        self._converter_fn = converter_fn or default_converter

    def with_server_params(self, server_params: StdioServerParameters) -> Self:
        """Setter for server params.

        For convenience and users who prefer the fluent style.
        """

        self.server_params = server_params
        return self

    async def retrieve(self, query: str) -> CallToolResult:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
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
