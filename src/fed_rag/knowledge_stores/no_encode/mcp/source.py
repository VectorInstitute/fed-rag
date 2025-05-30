import uuid
from typing import Protocol

from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing_extensions import Self

from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import KnowledgeStoreError


class CallToolResultConverter(Protocol):
    def __call__(self, result: CallToolResult) -> KnowledgeNode:
        pass  # pragma: no cover


class MCPKnowledgeSource(BaseModel):
    """The MCPKnowledgeSource class.

    Users can easily connect MCP tools as their source of knowledge in RAG systems.
    """

    name: str
    url: str
    tool_name: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _converter_fn: CallToolResultConverter | None = PrivateAttr(default=None)

    def __init__(
        self,
        url: str,
        tool_name: str,
        name: str | None = None,
        converter_fn: CallToolResultConverter | None = None,
    ):
        name = name or f"source-{str(uuid.uuid4())}"
        super().__init__(name=name, url=url, tool_name=tool_name)
        self._converter_fn = converter_fn

    def with_converter(self, converter_fn: CallToolResultConverter) -> Self:
        """Setter for converter_fn.

        Supports fluent pattern: `source = MCPKnowledgeSource(...).with_converter()`
        """
        self._converter_fn = converter_fn
        return self

    def with_name(self, name: str) -> Self:
        """Setter for name.

        For convenience and users who prefer the fluent style.
        """

        self.name = name
        return self

    def call_tool_result_to_knowledge_node(
        self,
        result: CallToolResult,
    ) -> KnowledgeNode:
        """Convert a call tool result to a knowledge node."""
        converter: CallToolResultConverter = (
            self._converter_fn or self.default_converter
        )
        return converter(result=result)

    def default_converter(self, result: CallToolResult) -> KnowledgeNode:
        if result.isError:
            raise KnowledgeStoreError(
                "Failed to convert `CallToolResult` to a `KnowledgeNode`: result has error status."
            )

        text_content = "\n".join(
            c.text for c in result.content if c.type == "text"
        )
        return KnowledgeNode(
            node_type="text",
            text_content=text_content,
            metadata=self.model_dump(),
        )
