from typing import Any, Optional, Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.data_structs.struct_type import IndexStructType
from llama_index.core.indices.managed.base import BaseManagedIndex
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.llms.utils import LLMType
from llama_index.core.schema import Document, MediaResource
from llama_index.core.schema import Node as LlamaNode
from llama_index.core.schema import NodeWithScore, QueryBundle

from fed_rag.exceptions import BridgeError
from fed_rag.types.knowledge_node import KnowledgeNode
from fed_rag.types.rag_system import RAGSystem, SourceNode


def convert_source_node_to_llama_index_node_with_score(
    node: SourceNode,
) -> NodeWithScore:
    """Convert ~fed_rag.SourceNode to ~llama_index.NodeWithScore.

    NOTE: Currently only text nodes are supported.
    """
    text_resource = MediaResource(text=node.node.text_content)
    llama_index_node = LlamaNode(
        text_resource=text_resource, id_=node.node.node_id
    )
    return NodeWithScore(score=node.score, node=llama_index_node)


def convert_llama_index_node_to_knowledge_node(
    llama_node: LlamaNode,
) -> KnowledgeNode:
    """Convert ~llama_index.BaseNodes to ~fed_rag.KnowledgeNodes.

    NOTE: Currently only text nodes are supported.
    """
    if llama_node.embedding is None:
        raise BridgeError(
            "Failed to convert ~llama_index.Node: embedding attribute is None."
        )

    if llama_node.text_resource is None:
        raise BridgeError(
            "Failed to convert ~llama_index.Node: text_resource attribute is None."
        )
    return KnowledgeNode(
        node_id=llama_node.id_,
        embedding=llama_node.embedding,
        node_type="text",
        text_content=llama_node.text_resource.text,
        metadata=llama_node.metadata,
    )


class FedRAGManagedIndex(BaseManagedIndex):
    # Inner Clases
    class FedRAGRetriever(BaseRetriever):
        """A ~llama_index.BaseRetriever adapter for fed_rag.RAGSystem."""

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
            return [
                convert_source_node_to_llama_index_node_with_score(sn)
                for sn in source_nodes
            ]

    class FedRAGLLM(CustomLLM):
        """A ~llama_index.LLM adapter for fed_rag.RAGSystem.

        NOTE: this is a very basic LLM adapter.
        """

        def __init__(self, rag_system: RAGSystem, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._rag_system = rag_system

        @property
        def metadata(self) -> LLMMetadata:
            """Get LLM metadata."""
            return LLMMetadata(
                model_name="fedrag.generator",
            )

        @llm_completion_callback()
        def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
            res = self._rag_system.generator.generate(query=prompt, context="")
            return CompletionResponse(text=res)

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, **kwargs: Any
        ) -> CompletionResponseGen:
            raise NotImplementedError(
                "stream_complete is not implemented for FedRAGLLM."
            )

    class FedRAGIndexStruct(IndexStruct):
        @classmethod
        def get_type(cls) -> IndexStructType:
            return IndexStructType.VECTOR_STORE

    # methods and attributes
    def __init__(self, rag_system: RAGSystem, *args: Any, **kwargs: Any):
        nodes = kwargs.get("nodes", [])
        if len(list(nodes)) > 0:
            raise BridgeError(
                "FedRAGManagedIndex does not support nodes on initialization."
            )
        super().__init__(nodes=[], *args, **kwargs)
        self._rag_system = rag_system

    def _insert(
        self, nodes: Sequence[LlamaNode], **insert_kwargs: Any
    ) -> None:
        knowledge_nodes = [
            convert_llama_index_node_to_knowledge_node(llama_node=n)
            for n in nodes
        ]
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

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        return self.FedRAGRetriever(rag_system=self._rag_system)

    def as_query_engine(
        self, llm: Optional[LLMType] = None, **kwargs: Any
    ) -> BaseQueryEngine:
        # set llm
        llm = self.FedRAGLLM(rag_system=self._rag_system)
        return super().as_query_engine(llm=llm, **kwargs)

    def _build_index_from_nodes(
        self, nodes: Sequence[LlamaNode], **build_kwargs: Any
    ) -> IndexStruct:
        return self.FedRAGIndexStruct(
            summary="~fed_rag.FedRAGManagedIndex wrapper of RAGSystem."
        )
