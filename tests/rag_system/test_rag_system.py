from unittest.mock import MagicMock, patch

import torch

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.retriever import BaseRetriever
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode
from fed_rag.types.rag_system import RAGConfig, RAGSystem, SourceNode

from .conftest import MockGenerator, MockRetriever


def test_rag_system_init(
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    assert rag_system.knowledge_store == knowledge_store
    assert rag_system.rag_config == rag_config
    assert rag_system.generator == mock_generator
    assert rag_system.retriever == mock_retriever


@patch.object(RAGSystem, "generate")
@patch.object(RAGSystem, "_format_context")
@patch.object(RAGSystem, "retrieve")
def test_rag_system_query(
    mock_retrieve: MagicMock,
    mock_format_context: MagicMock,
    mock_generate: MagicMock,
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    source_nodes = [
        SourceNode(score=0.99, node=knowledge_nodes[0]),
        SourceNode(score=0.85, node=knowledge_nodes[1]),
    ]
    mock_retrieve.return_value = source_nodes
    mock_format_context.return_value = "fake context"
    mock_generate.return_value = "fake generation response"

    # build rag system
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # act
    rag_response = rag_system.query(query="fake query")

    # assert
    mock_retrieve.assert_called_with("fake query")
    mock_format_context.assert_called_with(source_nodes)
    mock_generate.assert_called_with(
        query="fake query", context="fake context"
    )
    assert rag_response.source_nodes == source_nodes
    assert rag_response.response == "fake generation response"
    assert str(rag_response) == "fake generation response"


@patch.object(MockRetriever, "encode_query")
def test_rag_system_retrieve(
    mock_encode_query: MagicMock,
    mock_generator: BaseGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    mock_encode_query.return_value = torch.Tensor([1.0, 1.0, 1.0])

    # build rag system
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # expected retrieved source nodes
    expected = [
        SourceNode(score=0.8164965809277259, node=knowledge_nodes[0]),
        SourceNode(score=0.8164965809277259, node=knowledge_nodes[2]),
    ]

    # act
    result = rag_system.retrieve("fake query")

    # assert
    mock_encode_query.assert_called_with("fake query")
    assert all(res.node == exp.node for res, exp in zip(result, expected))
    assert all(
        abs(res.score - exp.score) < 1e-5 for res, exp in zip(result, expected)
    )


@patch.object(MockGenerator, "generate")
def test_rag_system_generate(
    mock_generate: MagicMock,
    mock_generator: MockGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    mock_generate.return_value = "fake generate response"

    # build rag system
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # act
    res = rag_system.generate(query="fake query", context="fake context")

    # assert
    mock_generate.assert_called_once_with(
        query="fake query", context="fake context"
    )
    assert res == "fake generate response"


def test_rag_system_format_context(
    mock_generator: MockGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # build rag system
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = RAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # act
    formatted_context = rag_system._format_context(
        source_nodes=knowledge_nodes
    )

    # assert
    assert formatted_context == "node 1\nnode 2\nnode 3"
