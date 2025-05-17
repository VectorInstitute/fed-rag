"""Internal RAG System Module"""

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict

from fed_rag.base.bridge import BridgeMetadata
from fed_rag.data_structures import (
    KnowledgeNode,
    RAGConfig,
    RAGResponse,
    SourceNode,
)

if TYPE_CHECKING:  # pragma: no cover
    # to avoid circular imports, using forward refs
    from fed_rag.base.generator import BaseGenerator
    from fed_rag.base.knowledge_store import BaseKnowledgeStore
    from fed_rag.base.retriever import BaseRetriever


class _RAGSystem(BaseModel):
    """Unbridged implementation of RAGSystem.

    IMPORTANT: This is an internal implementation class.
    It should only be used by bridge mixins and never referenced directly
    by user code or other parts of the library.

    All interaction with RAG systems should be through the public RAGSystem class.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: "BaseGenerator"
    retriever: "BaseRetriever"
    knowledge_store: "BaseKnowledgeStore"
    rag_config: RAGConfig
    bridges: ClassVar[dict[str, BridgeMetadata]] = {}

    @classmethod
    def _register_bridge(cls, metadata: BridgeMetadata) -> None:
        """To be used only by `BaseBridgeMixin`."""
        if metadata["framework"] not in cls.bridges:
            cls.bridges[metadata["framework"]] = metadata

    def query(self, query: str) -> RAGResponse:
        """Query the RAG system."""
        source_nodes = self.retrieve(query)
        context = self._format_context(source_nodes)
        response = self.generate(query=query, context=context)
        return RAGResponse(source_nodes=source_nodes, response=response)

    def retrieve(self, query: str) -> list[SourceNode]:
        """Retrieve from KnowledgeStore."""
        query_emb: list[float] = self.retriever.encode_query(query).tolist()
        raw_retrieval_result = self.knowledge_store.retrieve(
            query_emb=query_emb, top_k=self.rag_config.top_k
        )
        return [
            SourceNode(score=el[0], node=el[1]) for el in raw_retrieval_result
        ]

    def generate(self, query: str, context: str) -> str:
        """Generate response to query with context."""
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
    from fed_rag.base.knowledge_store import BaseKnowledgeStore
    from fed_rag.base.retriever import BaseRetriever

    # Update forward references
    _RAGSystem.model_rebuild()


_resolve_forward_refs()
