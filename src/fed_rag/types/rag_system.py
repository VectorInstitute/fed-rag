"""RAG System"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.knowledge_store import BaseKnowledgeStore
from fed_rag.base.retriever import BaseRetriever
from fed_rag.types.knowledge_node import KnowledgeNode


class SourceNode(BaseModel):
    score: float
    node: KnowledgeNode

    def __getattr__(self, __name: str) -> Any:
        """Convenient wrapper on getattr of associated node."""
        return getattr(self.node, __name)


class RAGResponse(BaseModel):
    response: str
    source_nodes: list[SourceNode]

    def __str__(self) -> str:
        return self.response


class RAGConfig(BaseModel):
    top_k: int
    context_separator: str = "\n"


class RAGSystem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    generator: BaseGenerator
    retriever: BaseRetriever
    knowledge_store: BaseKnowledgeStore
    rag_config: RAGConfig

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
        return self.rag_config.context_separator.join(
            [node.get_content()["text_content"] for node in source_nodes]
        )
