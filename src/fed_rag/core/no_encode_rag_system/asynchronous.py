"""Async No Encode RAG System Module"""

from fed_rag.core.no_encode_rag_system._asynchronous import (
    _AsyncNoEncodeRAGSystem,
)

from .synchronous import NoEncodeRAGSystem


# Define the public NoEncodeRAGSystem with all available bridges
class AsyncNoEncodeRAGSystem(_AsyncNoEncodeRAGSystem):
    """Async NoEncode RAG System with all available bridge functionality.

    The AsyncNoEncodeRAGSystem is the main entry point for creating and managing
    retrieval-augmented generation systems that skip encoding altogether,
    enabling direct natural language queries to knowledge sources like MCP
    servers, APIs, and databases.

    Unlike traditional RAG systems that require separate retriever components
    and pre-computed embeddings, NoEncode RAG systems perform direct queries
    against NoEncode knowledge sources.
    """

    def to_sync(
        self,
    ) -> NoEncodeRAGSystem:
        return NoEncodeRAGSystem(
            knowledge_store=self.knowledge_store.to_sync(),
            generator=self.generator,  # NOTE: this should actually be sync!
            rag_config=self.rag_config,
        )
