"""Base MCP Knowledge Source"""

from abc import ABC, abstractmethod

from mcp.types import CallToolResult
from pydantic import BaseModel, ConfigDict

from fed_rag.data_structures import KnowledgeNode


class BaseMCPKnowledgeSource(BaseModel, ABC):
    """A base model for MCP knowledge sources (i.e., server) for MCPKnowledgeStore."""

    name: str
    url: str
    tool_name: str | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def call_tool_result_to_knowledge_node(
        resource_result: CallToolResult,
    ) -> KnowledgeNode:
        """Convert a call tool result to a knowledge node."""
