"""Exceptions for Knowledge Stores."""

from .core import FedRAGError, FedRAGWarning


class KnowledgeStoreError(FedRAGError):
    """Base knowledge store error for all knowledge-store-related exceptions."""

    pass


class KnowledgeStoreWarning(FedRAGWarning):
    """Base knowledge store error for all knowledge-store-related warnings."""

    pass


class KnowledgeStoreNotFoundError(KnowledgeStoreError, FileNotFoundError):
    pass


class InvalidDistanceError(KnowledgeStoreError):
    pass


class LoadNodeError(KnowledgeStoreError):
    pass


class MCPKnowledgeStoreError(KnowledgeStoreError):
    """Base knowledge store error for all knowledge-store-related exceptions."""

    pass


class CallToolResultConversionError(MCPKnowledgeStoreError):
    """Raised when trying to convert a ~mcp.CallToolResult that has error status."""

    pass
