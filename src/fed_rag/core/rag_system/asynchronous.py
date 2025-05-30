"""Async RAG System Module"""

from fed_rag._bridges.langchain.bridge import LangChainBridgeMixin
from fed_rag._bridges.llamaindex.bridge import LlamaIndexBridgeMixin
from fed_rag.core.rag_system._asynchronous import _AsyncRAGSystem

from .synchronous import RAGSystem


# Define the public RAGSystem with all available bridges
class AsyncRAGSystem(
    LlamaIndexBridgeMixin, LangChainBridgeMixin, _AsyncRAGSystem
):
    """Async RAG System with all available bridge functionality.

    The RAGSystem is the main entry point for creating and managing
    retrieval-augmented generation systems.
    """

    def to_sync(
        self,
    ) -> RAGSystem:
        return RAGSystem(
            knowledge_store=self.knowledge_store.to_sync(),
            generator=self.generator,  # NOTE: this should actually be sync!
            retriever=self.retriever,  # NOTE: this should actually be sync!
            rag_config=self.rag_config,
        )
