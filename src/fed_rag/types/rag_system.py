"""Public RAG System API.

This module contains the public RAGSystem class that users interact with.
It combines the base implementation with available bridge mixins.
"""

from fed_rag._bridges.llamaindex.bridge import LlamaIndexBridgeMixin
from fed_rag.types._rag_system import (
    RAGConfig,
    RAGResponse,
    SourceNode,
    _RAGSystem,
)


# Define the public RAGSystem with all available bridges
class RAGSystem(LlamaIndexBridgeMixin, _RAGSystem):
    """RAG System with all available bridge functionality.

    The RAGSystem is the main entry point for creating and managing
    retrieval-augmented generation systems. It integrates retrievers,
    generators, and knowledge stores into a unified workflow.

    In addition to core RAG functionality, it provides bridge methods
    to popular frameworks:

    - to_llamaindex(): Convert to a LlamaIndex object (requires llama-index)
    - to_langchain(): Convert to a LangChain object (requires langchain)
    """

    pass


# Export public classes
__all__ = ["RAGSystem", "RAGConfig", "RAGResponse", "SourceNode"]
