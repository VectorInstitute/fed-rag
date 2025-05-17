"""Data structures for RAG.

Note: The correct module has moved to fed_rag.data_structures.rag. This module is
maintained for backward compatibility.
"""

import warnings

from ..data_structures.rag import RAGConfig, RAGResponse, SourceNode

warnings.warn(
    "Importing RAGConfig, RAGResponse, SourceNode from fed_rag.types.rag"
    "is deprecated and will be removed in a future release. Use "
    "fed_rag.data_structures.rag or fed_rag.data_structures instead.",
    DeprecationWarning,
    stacklevel=2,  # point to users import statement
)

__all__ = ["RAGConfig", "RAGResponse", "SourceNode"]
