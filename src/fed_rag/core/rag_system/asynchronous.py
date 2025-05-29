"""Async RAG System Module"""

from fed_rag._bridges.llamaindex.bridge import LlamaIndexBridgeMixin
from fed_rag.core.rag_system._asynchronous import _AsyncRAGSystem


# Define the public RAGSystem with all available bridges
class AsyncRAGSystem(LlamaIndexBridgeMixin, _AsyncRAGSystem):
    """Async RAG System with all available bridge functionality.

    The RAGSystem is the main entry point for creating and managing
    retrieval-augmented generation systems.
    """

    pass
