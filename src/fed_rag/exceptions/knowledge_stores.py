"""Exceptions for Knowledge Stores."""

from .core import FedRAGError, FedRAGWarning


class KnowledgeStoreError(FedRAGError):
    """Base knowledge store error for all knowledge-store-related exceptions."""

    pass


class KnowledgeStoreWarning(FedRAGWarning):
    """Base knowledge store error for all knowledge-store-related warnings."""

    pass


class KnowledgeStoreNotFoundError(KnowledgeStoreError, FileNotFoundError):
    """Raised if the knowledge store can not be found or loaded from file."""

    pass


class InvalidDistanceError(KnowledgeStoreError):
    """Raised if provided an invalid similarity distance."""

    pass


class LoadNodeError(KnowledgeStoreError):
    """Raised if an error occurs when loading a node."""

    pass


class MCPKnowledgeStoreError(KnowledgeStoreError):
    """Base knowledge store error for all knowledge-store-related exceptions."""

    pass


class CallToolResultConversionError(MCPKnowledgeStoreError):
    """Raised when trying to convert a ~mcp.CallToolResult that has error status."""

    pass
