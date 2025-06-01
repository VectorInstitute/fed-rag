from typing import Any, Protocol

from mcp.types import CallToolResult

from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import CallToolResultConversionError


class CallToolResultConverter(Protocol):
    def __call__(
        self, result: CallToolResult, metadata: dict[str, Any] | None = None
    ) -> list[KnowledgeNode]:
        pass  # pragma: no cover


def default_converter(
    result: CallToolResult, metadata: dict[str, Any] | None = None
) -> list[KnowledgeNode]:
    if result.isError:
        raise CallToolResultConversionError(
            "Cannot convert a `CallToolResult` with `isError` set to `True`."
        )

    return [
        KnowledgeNode(
            node_type="text",
            text_content=c.text,
            metadata=metadata,
        )
        for c in result.content
        if c.type == "text"
    ]
