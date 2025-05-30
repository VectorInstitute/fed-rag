import re
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.outputs import Generation, LLMResult
from langchain_core.vectorstores import VectorStoreRetriever

from fed_rag._bridges.langchain import LangChainBridgeMixin
from fed_rag._bridges.langchain._bridge_classes import (
    FedRAGLLM,
    FedRAGVectorStore,
)
from fed_rag.core.rag_system._synchronous import _RAGSystem
from fed_rag.data_structures import KnowledgeNode, SourceNode
from fed_rag.exceptions import BridgeError


def test_rag_system_bridges(mock_rag_system: _RAGSystem) -> None:
    metadata = LangChainBridgeMixin.get_bridge_metadata()

    assert "langchain-core" in mock_rag_system.bridges
    assert mock_rag_system.bridges[metadata["framework"]] == metadata
    assert LangChainBridgeMixin._bridge_extra == "langchain"


@patch("fed_rag._bridges.langchain._bridge_classes.FedRAGVectorStore")
@patch("fed_rag._bridges.langchain._bridge_classes.FedRAGLLM")
def test_rag_system_conversion_method(
    mock_vector_store_class: MagicMock,
    mock_llm_class: MagicMock,
    mock_rag_system: _RAGSystem,
) -> None:
    metadata = LangChainBridgeMixin.get_bridge_metadata()
    assert metadata["method_name"] == "to_langchain"
    assert hasattr(mock_rag_system, metadata["method_name"])

    conversion_method = getattr(mock_rag_system, metadata["method_name"])
    vector_store, llm = conversion_method()

    mock_vector_store_class.assert_called_once_with(mock_rag_system)
    mock_llm_class.assert_called_once_with(mock_rag_system)


def test_rag_system_conversion_results(mock_rag_system: _RAGSystem) -> None:
    vector_store, llm = mock_rag_system.to_langchain()

    assert isinstance(vector_store, FedRAGVectorStore)
    assert isinstance(llm, FedRAGLLM)


def test_rag_vector_store_embeddings(mock_rag_system: _RAGSystem) -> None:
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor([[1, 2, 3]])
    )

    vector_store = FedRAGVectorStore(mock_rag_system)
    embeddings = vector_store.embeddings

    assert isinstance(embeddings, Embeddings)

    test_query = "test query"
    test_result = [[1, 2, 3]]

    result = embeddings.embed_query(test_query)
    mock_rag_system.retriever.encode_query.assert_called_once_with(test_query)
    assert result == test_result

    mock_rag_system.retriever.encode_query.reset_mock()

    result = embeddings.embed_documents([test_query])
    mock_rag_system.retriever.encode_query.assert_called_once_with(
        [test_query]
    )
    assert result == test_result


def test_rag_vector_store_delete_none(mock_rag_system: _RAGSystem) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    vector_store = FedRAGVectorStore(mock_rag_system)

    assert not vector_store.delete(ids=None)
    mock_rag_system.knowledge_store.delete_node.assert_not_called()


def test_rag_vector_store_delete_ids(mock_rag_system: _RAGSystem) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    vector_store = FedRAGVectorStore(mock_rag_system)

    ids_to_delete = ["id1", "id2"]
    assert vector_store.delete(ids=ids_to_delete)
    mock_rag_system.knowledge_store.delete_node.assert_has_calls(
        [call("id1"), call("id2")], any_order=False
    )


def test_rag_vector_store_add_text_metadata_length_mismatch_error(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    vector_store = FedRAGVectorStore(mock_rag_system)

    with pytest.raises(
        ValueError, match="The number of texts, metadatas, and ids must match."
    ):
        vector_store.add_texts(
            texts=["text1", "text2"],
            metadatas=[{"key": "value"}],  # only one metadata for two texts
        )

    mock_rag_system.knowledge_store.load_nodes.assert_not_called()


def test_rag_vector_store_add_texts_ids_length_mismatch_error(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    vector_store = FedRAGVectorStore(mock_rag_system)

    with pytest.raises(
        ValueError, match="The number of texts, metadatas, and ids must match."
    ):
        vector_store.add_texts(
            texts=["text1", "text2"],
            ids=["id1"],  # only one id for two texts
        )

    mock_rag_system.knowledge_store.load_nodes.assert_not_called()


def test_rag_vector_store_add_texts(mock_rag_system: _RAGSystem) -> None:
    embedding = [1.0, 2.0, 3.0]
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(embedding)
    )
    vector_store = FedRAGVectorStore(mock_rag_system)

    texts = ["text1", "text2"]
    metadatas = [{"key": "value1"}, {"key": "value2"}]
    ids = ["id1", "id2"]

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    expected_nodes = [
        KnowledgeNode(
            node_id="id1",
            embedding=embedding,
            node_type="text",
            text_content="text1",
            metadata={"key": "value1"},
        ),
        KnowledgeNode(
            node_id="id2",
            embedding=embedding,
            node_type="text",
            text_content="text2",
            metadata={"key": "value2"},
        ),
    ]

    mock_rag_system.knowledge_store.load_nodes.assert_called_once_with(
        expected_nodes
    )


def test_rag_vector_store_add_documents(mock_rag_system: _RAGSystem) -> None:
    embedding = [1.0, 2.0, 3.0]
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(embedding)
    )
    vector_store = FedRAGVectorStore(mock_rag_system)

    documents = [
        Document(page_content="text1", metadata={"key": "value1"}),
        Document(page_content="text2", metadata={"key": "value2"}),
    ]
    ids = ["id1", "id2"]

    vector_store.add_documents(documents=documents, ids=ids)

    expected_nodes = [
        KnowledgeNode(
            node_id="id1",
            embedding=embedding,
            node_type="text",
            text_content="text1",
            metadata={"key": "value1"},
        ),
        KnowledgeNode(
            node_id="id2",
            embedding=embedding,
            node_type="text",
            text_content="text2",
            metadata={"key": "value2"},
        ),
    ]

    mock_rag_system.knowledge_store.load_nodes.assert_called_once_with(
        expected_nodes
    )


def test_rag_vector_store_add_documents_no_ids(
    mock_rag_system: _RAGSystem,
) -> None:
    embedding = [1.0, 2.0, 3.0]
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(embedding)
    )
    vector_store = FedRAGVectorStore(mock_rag_system)

    documents = [
        Document(page_content="text1", metadata={"key": "value1"}),
        Document(page_content="text2", metadata={"key": "value2"}),
    ]

    ids = vector_store.add_documents(documents=documents)

    assert isinstance(ids, list)
    assert len(ids) == 2

    expected_nodes = [
        KnowledgeNode(
            node_id=ids[0],
            embedding=embedding,
            node_type="text",
            text_content="text1",
            metadata={"key": "value1"},
        ),
        KnowledgeNode(
            node_id=ids[1],
            embedding=embedding,
            node_type="text",
            text_content="text2",
            metadata={"key": "value2"},
        ),
    ]

    mock_rag_system.knowledge_store.load_nodes.assert_called_once_with(
        expected_nodes
    )


query_text = "test query"
query_vector = [1.0, 2.0, 3.0]
top_k = 5
expected_nodes = [
    SourceNode(
        score=0.9,
        node=KnowledgeNode(
            node_id="1", text_content="node 1", node_type="text"
        ),
    ),
    SourceNode(
        score=0.8,
        node=KnowledgeNode(
            node_id="2", text_content="node 2", node_type="text"
        ),
    ),
]


def test_rag_vector_store_similarity_search(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(query_vector)
    )
    vector_store = FedRAGVectorStore(mock_rag_system)

    mock_rag_system.knowledge_store.retrieve.return_value = [
        (node.score, node.node) for node in expected_nodes
    ]

    results = vector_store.similarity_search(query=query_text, k=top_k)

    mock_rag_system.knowledge_store.retrieve.assert_called_once_with(
        query_emb=query_vector, top_k=top_k
    )

    assert len(results) == len(expected_nodes)
    for result, expected in zip(results, expected_nodes):
        assert result.id == expected.node.node_id
        assert result.page_content == expected.node.text_content

    mock_rag_system.retriever.encode_query.assert_called_once_with(query_text)


def test_rag_vector_store_similarity_search_by_vector(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    vector_store = FedRAGVectorStore(mock_rag_system)

    mock_rag_system.knowledge_store.retrieve.return_value = [
        (node.score, node.node) for node in expected_nodes
    ]

    results = vector_store.similarity_search_by_vector(
        embedding=query_vector, k=top_k
    )

    mock_rag_system.knowledge_store.retrieve.assert_called_once_with(
        query_emb=query_vector, top_k=top_k
    )

    assert len(results) == len(expected_nodes)
    for result, expected in zip(results, expected_nodes):
        assert result.id == expected.node.node_id
        assert result.page_content == expected.node.text_content


def test_rag_vector_store_similarity_search_with_score(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(query_vector)
    )
    vector_store = FedRAGVectorStore(mock_rag_system)

    mock_rag_system.knowledge_store.retrieve.return_value = [
        (node.score, node.node) for node in expected_nodes
    ]

    results = vector_store.similarity_search_with_score(
        query=query_text, k=top_k
    )

    mock_rag_system.knowledge_store.retrieve.assert_called_once_with(
        query_emb=query_vector, top_k=top_k
    )

    assert len(results) == len(expected_nodes)
    for result, expected in zip(results, expected_nodes):
        assert result[0].id == expected.node.node_id
        assert result[0].page_content == expected.node.text_content
        assert result[1] == expected.score

    mock_rag_system.retriever.encode_query.assert_called_once_with(query_text)


def test_rag_vector_store_similarity_search_with_relevance_scores(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(query_vector)
    )
    vector_store = FedRAGVectorStore(mock_rag_system)

    mock_rag_system.knowledge_store.retrieve.return_value = [
        (node.score, node.node) for node in expected_nodes
    ]

    results = vector_store.similarity_search_with_relevance_scores(
        query=query_text, k=top_k
    )

    mock_rag_system.knowledge_store.retrieve.assert_called_once_with(
        query_emb=query_vector, top_k=top_k
    )

    assert len(results) == len(expected_nodes)
    for result, expected in zip(results, expected_nodes):
        assert result[0].id == expected.node.node_id
        assert result[0].page_content == expected.node.text_content
        assert result[1] == 1 - expected.score

    mock_rag_system.retriever.encode_query.assert_called_once_with(query_text)


def test_rag_vector_store_from_texts_not_implemented_error(
    mock_rag_system: _RAGSystem,
) -> None:
    vector_store = FedRAGVectorStore(mock_rag_system)
    msg = (
        "FedRAGVectorStore does not support from_texts method. "
        "Use the add_texts method to add texts to the vector store."
    )
    with pytest.raises(BridgeError, match=re.escape(msg)):
        vector_store.from_texts(texts=["text1", "text2"])


def test_rag_vector_store_as_retriever(mock_rag_system: _RAGSystem) -> None:
    mock_rag_system.knowledge_store = MagicMock()
    mock_rag_system.retriever = MagicMock()
    mock_rag_system.retriever.encode_query = MagicMock(
        return_value=torch.tensor(query_vector)
    )

    vector_store = FedRAGVectorStore(mock_rag_system)
    retriever = vector_store.as_retriever()

    assert isinstance(retriever, VectorStoreRetriever)

    mock_rag_system.knowledge_store.retrieve.return_value = [
        (node.score, node.node) for node in expected_nodes
    ]

    results = retriever.get_relevant_documents(query_text, k=top_k)

    assert len(results) == len(expected_nodes)
    for result, expected in zip(results, expected_nodes):
        assert result.id == expected.node.node_id
        assert result.page_content == expected.node.text_content

    mock_rag_system.retriever.encode_query.assert_called_once_with(query_text)


def test_rag_llm_generate_stop_set_error(mock_rag_system: _RAGSystem) -> None:
    mock_rag_system.generator = MagicMock()
    fedrag_llm = FedRAGLLM(mock_rag_system)
    msg = (
        "FedRAGLLM does not support stop sequences. "
        "Please use the generator directly if you need this feature."
    )
    with pytest.raises(
        BridgeError,
        match=re.escape(msg),
    ):
        fedrag_llm.generate([query_text], stop=["stop1", "stop2"])


def test_rag_llm_generate(mock_rag_system: _RAGSystem) -> None:
    mock_rag_system.generator = MagicMock()
    mock_rag_system.generator.generate = MagicMock(
        return_value="Generated response"
    )

    fedrag_llm = FedRAGLLM(mock_rag_system)
    result = fedrag_llm.generate([query_text])

    assert isinstance(result, LLMResult)
    assert len(result.generations) == 1
    assert isinstance(result.generations[0], list)
    assert isinstance(result.generations[0][0], Generation)
    assert result.generations[0][0].text == "Generated response"

    mock_rag_system.generator.generate.assert_called_once_with(
        query=query_text, context=""
    )


def test_rag_llm_generate_multiple_queries(
    mock_rag_system: _RAGSystem,
) -> None:
    mock_rag_system.generator = MagicMock()
    mock_rag_system.generator.generate = MagicMock(
        return_value="Generated response"
    )

    fedrag_llm = FedRAGLLM(mock_rag_system)
    queries = [query_text, "another query"]
    result = fedrag_llm.generate(queries)

    assert isinstance(result, LLMResult)
    assert len(result.generations) == 2
    for generation in result.generations:
        assert isinstance(generation, list)
        assert isinstance(generation[0], Generation)
        assert generation[0].text == "Generated response"

    mock_rag_system.generator.generate.assert_has_calls(
        [
            call(query=query_text, context=""),
            call(query="another query", context=""),
        ]
    )


def test_rag_llm_stream_not_implemented_error(
    mock_rag_system: _RAGSystem,
) -> None:
    fedrag_llm = FedRAGLLM(mock_rag_system)
    msg = (
        "FedRAGLLM does not support streaming. "
        "Please use the generator directly if you need this feature."
    )
    with pytest.raises(
        BridgeError,
        match=re.escape(msg),
    ):
        for _ in fedrag_llm.stream(query_text):
            pass
