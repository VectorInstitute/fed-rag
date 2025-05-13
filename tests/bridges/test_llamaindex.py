from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.schema import MediaResource
from llama_index.core.schema import Node as LlamaNode
from llama_index.core.schema import QueryBundle

from fed_rag.bridges.llamaindex._managed_index import (
    FedRAGManagedIndex,
    convert_llama_index_node_to_knowledge_node,
    convert_source_node_to_llama_index_node_with_score,
)
from fed_rag.bridges.llamaindex.bridge import LlamaIndexBridgeMixin
from fed_rag.exceptions import BridgeError
from fed_rag.types.rag_system import KnowledgeNode, RAGSystem, SourceNode


def test_rag_system_bridges(mock_rag_system: RAGSystem) -> None:
    metadata = LlamaIndexBridgeMixin.get_bridge_metadata()

    assert "llama-index" in mock_rag_system.bridges
    assert mock_rag_system.bridges[metadata["framework"]] == metadata
    assert LlamaIndexBridgeMixin._bridge_extra == "llama-index"


@patch("fed_rag.bridges.llamaindex._managed_index.FedRAGManagedIndex")
def test_rag_system_conversion_method(
    mock_managed_index_class: MagicMock, mock_rag_system: RAGSystem
) -> None:
    metadata = LlamaIndexBridgeMixin.get_bridge_metadata()

    conversion_method = getattr(mock_rag_system, metadata["method_name"])
    conversion_method()

    mock_managed_index_class.assert_called_once_with(
        rag_system=mock_rag_system
    )


# test node converters
def test_convert_llama_node_to_knowledge_node() -> None:
    llama_node = LlamaNode(
        id_="1",
        embedding=[1, 1, 1],
        text_resource=MediaResource(text="mock text"),
    )
    fed_rag_node = convert_llama_index_node_to_knowledge_node(llama_node)

    assert fed_rag_node.node_id == "1"
    assert fed_rag_node.embedding == [1, 1, 1]
    assert fed_rag_node.node_type == "text"
    assert fed_rag_node.text_content == "mock text"


def test_convert_llama_node_to_knowledge_node_raises_error_missing_embedding() -> (
    None
):
    llama_node = LlamaNode(
        id_="1", text_resource=MediaResource(text="mock text")
    )

    with pytest.raises(
        BridgeError,
        match="Failed to convert ~llama_index.Node: embedding attribute is None.",
    ):
        convert_llama_index_node_to_knowledge_node(llama_node)


def test_convert_llama_node_to_knowledge_node_raises_error_text_resource_is_none() -> (
    None
):
    llama_node = LlamaNode(id_="1", embedding=[1, 1, 1])

    with pytest.raises(
        BridgeError,
        match="Failed to convert ~llama_index.Node: text_resource attribute is None.",
    ):
        convert_llama_index_node_to_knowledge_node(llama_node)


def test_convert_source_node_to_llama_node_with_score() -> None:
    source_node = SourceNode(
        score=0.42,
        node=KnowledgeNode(
            node_id="1",
            embedding=[1, 1, 1],
            node_type="text",
            text_content="mock text",
        ),
    )

    llama_node_with_score = convert_source_node_to_llama_index_node_with_score(
        source_node
    )

    assert llama_node_with_score.score == 0.42
    assert isinstance(llama_node_with_score.node, LlamaNode)
    assert llama_node_with_score.node.id_ == "1"
    assert llama_node_with_score.node.text_resource.text == "mock text"


# test FedRAGManagedIndex
def test_fedrag_managed_index_init(mock_rag_system: RAGSystem) -> None:
    index = FedRAGManagedIndex(rag_system=mock_rag_system)

    assert index._rag_system == mock_rag_system


def test_fedrag_managed_index_init_raises_error_nodes_passed(
    mock_rag_system: RAGSystem,
) -> None:
    with pytest.raises(
        BridgeError,
        match="FedRAGManagedIndex does not support nodes on initialization.",
    ):
        FedRAGManagedIndex(rag_system=mock_rag_system, nodes=[LlamaNode()])


def test_fedrag_managed_index_as_retriever(mock_rag_system: RAGSystem) -> None:
    index = FedRAGManagedIndex(rag_system=mock_rag_system)
    retriever = index.as_retriever()

    assert isinstance(retriever, FedRAGManagedIndex.FedRAGRetriever)
    assert retriever._rag_system == mock_rag_system


def test_fedrag_managed_index_as_retriever_retrieve_method(
    mock_rag_system: RAGSystem,
) -> None:
    mock_rag_system.knowledge_store.load_node(
        KnowledgeNode(
            node_id="1",
            node_type="text",
            text_content="mock node",
            embedding=[1, 1, 1],
        )
    )
    index = FedRAGManagedIndex(rag_system=mock_rag_system)
    retriever = index.as_retriever()

    # act
    res = retriever._retrieve(query_bundle=QueryBundle(query_str="mock query"))

    assert len(res) == 1
    assert res[0].node_id == "1"
    assert res[0].node.text_resource.text == "mock node"


@patch("llama_index.core.indices.base.resolve_llm")
@patch(
    "fed_rag.bridges.llamaindex._managed_index.FedRAGManagedIndex.FedRAGLLM"
)
def test_fedrag_managed_index_as_query_engine_mocked(
    mock_fedrag_llm_class: MagicMock,
    mock_resolve_llm: MagicMock,
    mock_rag_system: RAGSystem,
) -> None:
    index = FedRAGManagedIndex(rag_system=mock_rag_system)
    query_engine = index.as_query_engine()

    assert isinstance(query_engine, BaseQueryEngine)
    mock_fedrag_llm_class.assert_called_once_with(rag_system=mock_rag_system)
    mock_resolve_llm.assert_called_once()


def test_fedrag_managed_index_as_query_engine(
    mock_rag_system: RAGSystem,
) -> None:
    index = FedRAGManagedIndex(rag_system=mock_rag_system)
    query_engine = index.as_query_engine()

    assert isinstance(query_engine, BaseQueryEngine)


def test_fedrag_managed_index_delete_node(
    mock_rag_system: RAGSystem,
) -> None:
    mock_rag_system.knowledge_store.load_node(
        KnowledgeNode(
            node_id="1",
            node_type="text",
            text_content="mock node",
            embedding=[1, 1, 1],
        )
    )
    index = FedRAGManagedIndex(rag_system=mock_rag_system)

    # act
    index._delete_node(node_id="1")

    # assert
    assert index._rag_system.knowledge_store.count == 0


def test_fedrag_managed_index_insert(
    mock_rag_system: RAGSystem,
) -> None:
    llama_nodes = [
        LlamaNode(
            id_="1",
            embedding=[1, 1, 1],
            text_resource=MediaResource(text="node 1"),
        ),
        LlamaNode(
            id_="2",
            embedding=[2, 2, 2],
            text_resource=MediaResource(text="node 2"),
        ),
    ]
    index = FedRAGManagedIndex(rag_system=mock_rag_system)

    # act
    index._insert(nodes=llama_nodes)

    # assert
    assert index._rag_system.knowledge_store.count == 2


def test_fedrag_llm_complete(mock_rag_system: RAGSystem) -> None:
    llm = FedRAGManagedIndex.FedRAGLLM(mock_rag_system)

    response = llm.complete("mock prompt")

    assert response.text == "mock output from 'mock prompt' and ''."


def test_fedrag_llm_stream_complete_raises_error(
    mock_rag_system: RAGSystem,
) -> None:
    llm = FedRAGManagedIndex.FedRAGLLM(mock_rag_system)

    with pytest.raises(
        NotImplementedError,
        match="stream_complete is not implemented for FedRAGLLM.",
    ):
        llm.stream_complete(prompt="mock prompt")


## test methods with no implementation
def test_fedrag_managed_index_raises_not_implemented_error(
    mock_rag_system: RAGSystem,
) -> None:
    from llama_index.core.schema import Document

    index = FedRAGManagedIndex(rag_system=mock_rag_system)

    with pytest.raises(
        NotImplementedError,
        match="update_ref_doc not implemented for `FedRAGManagedIndex`.",
    ):
        index.update_ref_doc(document=Document())

    with pytest.raises(
        NotImplementedError,
        match="delete_ref_doc not implemented for `FedRAGManagedIndex`.",
    ):
        index.delete_ref_doc(ref_doc_id="1")
