"""MCP Knowledge Source with stdio Transport"""

import uuid

from mcp import StdioServerParameters
from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing_extensions import Self

from fed_rag.data_structures import KnowledgeNode

from .utils import CallToolResultConverter, default_converter


class MCPStdioKnowledgeSource(BaseModel):
    """The MCPStdioKnowledgeSource class.

    Users can easily connect MCP tools as their source of knowledge in RAG systems
    via the stdio transport.
    """

    server_params: StdioServerParameters
    name: str
    tool_name: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _converter_fn: CallToolResultConverter = PrivateAttr()

    def __init__(
        self,
        server_params: StdioServerParameters,
        tool_name: str,
        name: str | None = None,
        converter_fn: CallToolResultConverter | None = None,
    ):
        name = name or f"source-stdio-{str(uuid.uuid4())}"
        super().__init__(
            name=name, server_params=server_params, tool_name=tool_name
        )
        self._converter_fn = converter_fn or default_converter

    def with_converter(self, converter_fn: CallToolResultConverter) -> Self:
        """Setter for converter_fn.

        Supports fluent pattern: `source = MCPStdioKnowledgeSource(...).with_converter()`
        """
        self._converter_fn = converter_fn
        return self

    def with_name(self, name: str) -> Self:
        """Setter for name.

        For convenience and users who prefer the fluent style.
        """

        self.name = name
        return self

    def with_server_params(self, server_params: StdioServerParameters) -> Self:
        """Setter for server params.

        For convenience and users who prefer the fluent style.
        """

        self.server_params = server_params
        return self

    def call_tool_result_to_knowledge_node(
        self,
        result: CallToolResult,
    ) -> KnowledgeNode:
        """Convert a call tool result to a knowledge node."""
        return self._converter_fn(result=result, metadata=self.model_dump())
