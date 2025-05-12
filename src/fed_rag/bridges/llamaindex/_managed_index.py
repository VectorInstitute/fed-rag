from typing import Any, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.schema import BaseNode, Document

from fed_rag.types.knowledge_node import KnowledgeNode


def convert_knowledge_node_to_llama_index_docs(
    nodes: Sequence[KnowledgeNode],
) -> Sequence[BaseNode]:
    return []


class FedRAGManagedIndex(BaseManagedIndex):
    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        pass

    def delete_ref_doc(
        self,
        ref_doc_id: str,
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        pass

    def update_ref_doc(self, document: Document, **update_kwargs: Any) -> None:
        pass

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        pass
