from typing import Any, Protocol

from mcp.types import CallToolResult

from fed_rag.data_structures import KnowledgeNode
from fed_rag.exceptions import CallToolResultConversionError


class CallToolResultConverter(Protocol):
    def __call__(
        self, result: CallToolResult, metadata: dict[str, Any] | None = None
    ) -> KnowledgeNode:
        pass  # pragma: no cover


def default_converter(
    result: CallToolResult, metadata: dict[str, Any] | None = None
) -> KnowledgeNode:
    if result.isError:
        raise CallToolResultConversionError(
            "Cannot convert a `CallToolResult` with `isError` set to `True`."
        )

    text_content = "<<<CONTENT_SEPARATOR>>>".join(
        c.text for c in result.content if c.type == "text"
    )
    metadata = metadata or {}
    return KnowledgeNode(
        node_type="text",
        text_content=text_content,
        metadata=metadata,
    )
