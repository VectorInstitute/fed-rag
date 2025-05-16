"""
RAG System type definitions and implementation.

Note: The RAGSystem implementation has moved to fed_rag.core.rag_system.
This module is maintained for backward compatibility.
"""

import warnings

from ..core.rag_system import RAGSystem
from .rag import RAGConfig, RAGResponse, SourceNode

warnings.warn(
    "Importing RAGSystem from fed_rag.types.rag_system is deprecated and will be"
    "removed in a future release. Use fed_rag.core.rag_system or fed_rag instead.",
    DeprecationWarning,
    stacklevel=2,  # point to users import statement
)


# Export all symbols for backward compatibility
__all__ = ["RAGSystem", "RAGConfig", "RAGResponse", "SourceNode"]
