"""Base MCP Knowledge Source"""

from abc import ABC, abstractmethod

from mcp.types import CallToolResult, ReadResourceResult
from pydantic import BaseModel, ConfigDict

from fed_rag.data_structures import KnowledgeNode

type KnowledgeStoreRetrievalResult = list[tuple[float, KnowledgeNode]]


class BaseMCPKnowledgeSource(BaseModel, ABC):
    """A base model for MCP knowledge sources (i.e., server) for MCPKnowledgeStore."""

    url: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def read_resource_result_to_knowledge_store_retrieval_result(
        resource_result: ReadResourceResult,
    ) -> KnowledgeStoreRetrievalResult:
        """Convert a resource result to a knowledge store retrieval result."""

    @abstractmethod
    def call_tool_result_to_knowledge_store_retrieval_result(
        resource_result: CallToolResult,
    ) -> KnowledgeStoreRetrievalResult:
        """Convert a call tool result to a knowledge store retrieval result."""
