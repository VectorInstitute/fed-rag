"""RAG System Module"""

from fed_rag._bridges.langchain.bridge import LangChainBridgeMixin
from fed_rag._bridges.llamaindex.bridge import LlamaIndexBridgeMixin
from fed_rag.core.rag_system._synchronous import _RAGSystem


# Define the public RAGSystem with all available bridges
class RAGSystem(LlamaIndexBridgeMixin, LangChainBridgeMixin, _RAGSystem):
    """RAG System with all available bridge functionality.

    The RAGSystem is the main entry point for creating and managing
    retrieval-augmented generation systems.
    """

    pass
