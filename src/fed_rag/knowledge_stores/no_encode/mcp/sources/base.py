"""Base MCP Knowledge Source Class"""

from abc import ABC, abstractmethod
from typing import Any

from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from typing_extensions import Self

from fed_rag.data_structures import KnowledgeNode

from .utils import CallToolResultConverter


class BaseMCPKnowledgeSource(BaseModel, ABC):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True
    )
    name: str
    tool_name: str | None = None
    query_param_name: str
    tool_call_kwargs: dict[str, Any] = Field(default_factory=dict)
    _converter_fn: CallToolResultConverter = PrivateAttr()

    @abstractmethod
    async def retrieve(self, query: str) -> CallToolResult:
        """Asynchronously retrieve knowledge from this source."""

    # Common methods all sources share
    def call_tool_result_to_knowledge_nodes_list(
        self, result: CallToolResult
    ) -> list[KnowledgeNode]:
        return self._converter_fn(result=result, metadata=self.model_dump())

    def with_converter(self, converter_fn: CallToolResultConverter) -> Self:
        """Setter for converter_fn.

        Supports fluent pattern: `source = MCPStreamableHttpKnowledgeSource(...).with_converter()`
        """
        self._converter_fn = converter_fn
        return self

    def with_name(self, name: str) -> Self:
        """Setter for name.

        For convenience and users who prefer the fluent style.
        """

        self.name = name
        return self

    def with_query_param_name(self, v: str) -> Self:
        """Setter for query param name.

        For convenience and users who prefer the fluent style.
        """

        self.query_param_name = v
        return self

    def with_tool_call_kwargs(self, v: dict[str, Any]) -> Self:
        """Setter for tool call kwargs.

        For convenience and users who prefer the fluent style.
        """

        self.tool_call_kwargs = v
        return self
