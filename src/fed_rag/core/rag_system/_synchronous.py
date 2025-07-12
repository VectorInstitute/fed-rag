"""Internal RAG System Module"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from fed_rag.base.bridge import BridgeRegistryMixin
from fed_rag.data_structures import RAGConfig, RAGResponse, SourceNode
from fed_rag.exceptions import RAGSystemError

if TYPE_CHECKING:  # pragma: no cover
    # to avoid circular imports, using forward refs
    from fed_rag.base.generator import BaseGenerator
    from fed_rag.base.knowledge_store import BaseKnowledgeStore
    from fed_rag.base.retriever import BaseRetriever


class _RAGSystem(BridgeRegistryMixin, BaseModel):
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

    def query(self, query: str) -> RAGResponse:
        """Query the RAG system."""
        source_nodes = self.retrieve(query)
        context = self._format_context(source_nodes)
        response = self.generate(query=query, context=context)
        return RAGResponse(source_nodes=source_nodes, response=response)

    def batch_query(self, queries: list[str]) -> list[RAGResponse]:
        """Batch query the RAG system."""
        source_nodes_list = self.batch_retrieve(queries)
        contexts = [
            self._format_context(source_nodes)
            for source_nodes in source_nodes_list
        ]
        responses = self.batch_generate(queries, contexts)
        return [
            RAGResponse(source_nodes=source_nodes, response=response)
            for source_nodes, response in zip(source_nodes_list, responses)
        ]

    def retrieve(self, query: str) -> list[SourceNode]:
        """Retrieve from KnowledgeStore."""
        from torch import Tensor

        encode_result = self.retriever.encode_query(query)
        if isinstance(encode_result, Tensor):
            query_emb: list[float] = encode_result.tolist()
        else:
            try:
                query_emb = encode_result["text"].tolist()
            except AttributeError:
                raise RAGSystemError(
                    "Encode result does not have a text embedding."
                )

        raw_retrieval_result = self.knowledge_store.retrieve(
            query_emb=query_emb, top_k=self.rag_config.top_k
        )
        return [
            SourceNode(score=el[0], node=el[1]) for el in raw_retrieval_result
        ]

    def batch_retrieve(self, queries: list[str]) -> list[list[SourceNode]]:
        """Batch retrieve from KnowledgeStore."""
        from torch import Tensor

        encode_result = self.retriever.encode_query(queries)
        if isinstance(encode_result, Tensor):
            query_embs: list[list[float]] = encode_result.tolist()
        else:
            try:
                query_embs = encode_result["text"].tolist()
            except AttributeError:
                raise RAGSystemError(
                    "Encode result does not have a text embedding."
                )
        try:
            raw_retrieval_results = self.knowledge_store.batch_retrieve(
                query_embs=query_embs, top_k=self.rag_config.top_k
            )
        except NotImplementedError:
            raw_retrieval_results = [
                self.knowledge_store.retrieve(
                    query_emb=query_emb, top_k=self.rag_config.top_k
                )
                for query_emb in query_embs
            ]

        return [
            [SourceNode(score=el[0], node=el[1]) for el in raw_result]
            for raw_result in raw_retrieval_results
        ]

    def generate(self, query: str, context: str) -> str:
        """Generate response to query with context."""
        return self.generator.generate(query=query, context=context)  # type: ignore

    def batch_generate(
        self, queries: list[str], contexts: list[str]
    ) -> list[str]:
        """Batch generate responses to queries with contexts."""
        if len(queries) != len(contexts):
            raise RAGSystemError(
                "Queries and contexts must have the same length for batch generation."
            )
        return self.generator.generate(query=queries, context=contexts)  # type: ignore

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
