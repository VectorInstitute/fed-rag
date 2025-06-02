"""Internal Async RAG System Module"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from fed_rag.base.bridge import BridgeRegistryMixin
from fed_rag.data_structures import RAGConfig, RAGResponse, SourceNode

if TYPE_CHECKING:  # pragma: no cover
    # to avoid circular imports, using forward refs
    from fed_rag.base.generator import BaseGenerator
    from fed_rag.base.no_encode_knowledge_store import (
        BaseAsyncNoEncodeKnowledgeStore,
    )


class _AsyncNoEncodeRAGSystem(BridgeRegistryMixin, BaseModel):
    """Unbridged implementation of NoEncodeRAGSystem.

    IMPORTANT: This is an internal implementation class.
    It should only be used by bridge mixins and never referenced directly
    by user code or other parts of the library.

    All interaction with RAG systems should be through the public AsyncNoEncodeRAGSystem
    class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: "BaseGenerator"
    knowledge_store: "BaseAsyncNoEncodeKnowledgeStore"
    rag_config: RAGConfig

    async def query(self, query: str) -> RAGResponse:
        """Asynchronously query the RAG system."""
        source_nodes = await self.retrieve(query)
        context = self._format_context(source_nodes)
        response = await self.generate(query=query, context=context)
        return RAGResponse(source_nodes=source_nodes, response=response)

    async def retrieve(self, query: str) -> list[SourceNode]:
        """Asynchronously retrieve from AsyncNoEncodeKnowledgeStore."""
        raw_retrieval_result = await self.knowledge_store.retrieve(
            query=query, top_k=self.rag_config.top_k
        )
        return [
            SourceNode(score=el[0], node=el[1]) for el in raw_retrieval_result
        ]

    async def generate(self, query: str, context: str) -> str:
        """Asynchronously generate response to query with context."""
        return self.generator.generate(query=query, context=context)  # type: ignore

    def _format_context(self, source_nodes: list[SourceNode]) -> str:
        """Format the context from the source nodes."""
        # TODO: how to format image context
        return str(
            self.rag_config.context_separator.join(
                [node.get_content()["text_content"] for node in source_nodes]
            )
        )


def _resolve_forward_refs() -> None:
    """Resolve forward references in _RAGSystem."""

    # These imports are needed for Pydantic to resolve forward references
    # ruff: noqa: F401
    from fed_rag.base.generator import BaseGenerator
    from fed_rag.base.no_encode_knowledge_store import (
        BaseAsyncNoEncodeKnowledgeStore,
    )

    # Update forward references
    _AsyncNoEncodeRAGSystem.model_rebuild()


_resolve_forward_refs()
