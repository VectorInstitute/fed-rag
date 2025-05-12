from typing import Any, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeWithScore,
    QueryBundle,
)

from fed_rag.types.knowledge_node import KnowledgeNode
from fed_rag.types.rag_system import RAGSystem, SourceNode


def convert_source_node_to_llama_index_node_with_score(
    nodes: Sequence[SourceNode],
) -> list[NodeWithScore]:
    """Convert ~fed_rag.SourceNode to ~llama_index.NodeWithScore."""
    return []


def convert_llama_index_docs_to_knowledge_node(
    llama_nodes: Sequence[BaseNode],
) -> list[KnowledgeNode]:
    """Convert ~llama_index.BaseNodes to ~fed_rag.KnowledgeNodes."""
    return []


class FedRAGManagedIndex(BaseManagedIndex):
    def __init__(self, rag_system: RAGSystem, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._rag_system = rag_system

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        knowledge_nodes = convert_llama_index_docs_to_knowledge_node(
            llama_nodes=nodes
        )
        self._rag_system.knowledge_store.load_nodes(knowledge_nodes)

    def delete_ref_doc(
        self,
        ref_doc_id: str,
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "_delete_ref_doc not implemented for `FedRAGManagedIndex`."
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        # node id's are presereved after conversion
        self._rag_system.knowledge_store.delete_node(node_id=node_id)

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        raise NotImplementedError(
            "update_ref_doc not implemented for `FedRAGManagedIndex`."
        )

    class FedRAGRetriever(BaseRetriever):
        def __init__(self, rag_system: RAGSystem, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._rag_system = rag_system

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            """Retrieve specialization for FedRAG.

            Currently only supports text-based queries.
            """
            source_nodes = self._rag_system.retrieve(
                query=query_bundle.query_str
            )
            return convert_source_node_to_llama_index_node_with_score(
                source_nodes
            )

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return self.FedRAGRetriever(rag_system=self._rag_system)
