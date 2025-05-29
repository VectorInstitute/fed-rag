"""No Encode RAG System Module"""

from fed_rag.core.no_encode_rag_system._synchronous import _NoEncodeRAGSystem


# Define the public NoEncodeRAGSystem with all available bridges
class NoEncodeRAGSystem(_NoEncodeRAGSystem):
    """NoEncode RAG System with all available bridge functionality.

    The NoEncodeRAGSystem is the main entry point for creating and managing
    retrieval-augmented generation systems that skip encoding altogether,
    enabling direct natural language queries to knowledge sources like MCP
    servers, APIs, and databases.

    Unlike traditional RAG systems that require separate retriever components
    and pre-computed embeddings, NoEncode RAG systems perform direct queries
    against NoEncode knowledge sources.
    """

    pass
