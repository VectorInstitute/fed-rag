from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from fed_rag import AsyncRAGSystem, RAGConfig
from fed_rag.base.generator import BaseGenerator
from fed_rag.base.retriever import BaseRetriever
from fed_rag.data_structures import KnowledgeNode, SourceNode
from fed_rag.exceptions import RAGSystemError

from .conftest import DummyAsyncKnowledgeStore, MockGenerator, MockRetriever


@pytest.mark.asyncio
async def test_rag_system_init(
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )
    assert rag_system.knowledge_store == knowledge_store
    assert rag_system.rag_config == rag_config
    assert rag_system.generator == mock_generator
    assert rag_system.retriever == mock_retriever


@pytest.mark.asyncio
async def test_rag_system_with_dual_encoder_init(
    mock_generator: BaseGenerator,
    mock_dual_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_dual_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )
    assert rag_system.knowledge_store == knowledge_store
    assert rag_system.rag_config == rag_config
    assert rag_system.generator == mock_generator
    assert rag_system.retriever == mock_dual_retriever


@pytest.mark.asyncio
@patch.object(AsyncRAGSystem, "generate")
@patch.object(AsyncRAGSystem, "_format_context")
@patch.object(AsyncRAGSystem, "retrieve")
async def test_rag_system_query(
    mock_retrieve: AsyncMock,
    mock_format_context: MagicMock,
    mock_generate: AsyncMock,
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
    knowledge_store = DummyAsyncKnowledgeStore()
    knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # act
    rag_response = await rag_system.query(query="fake query")

    # assert
    mock_retrieve.assert_called_with("fake query")
    mock_format_context.assert_called_with(source_nodes)
    mock_generate.assert_called_with(
        query="fake query", context="fake context"
    )
    assert rag_response.source_nodes == source_nodes
    assert rag_response.response == "fake generation response"
    assert str(rag_response) == "fake generation response"


@pytest.mark.asyncio
@patch.object(AsyncRAGSystem, "batch_generate")
@patch.object(AsyncRAGSystem, "_format_context")
@patch.object(AsyncRAGSystem, "batch_retrieve")
async def test_rag_system_batch_query(
    mock_batch_retrieve: MagicMock,
    mock_format_context: MagicMock,
    mock_batch_generate: MagicMock,
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    source_nodes = [
        [
            SourceNode(score=0.99, node=knowledge_nodes[0]),
            SourceNode(score=0.85, node=knowledge_nodes[1]),
        ],
        [
            SourceNode(score=0.90, node=knowledge_nodes[0]),
            SourceNode(score=0.80, node=knowledge_nodes[1]),
        ],
    ]
    mock_batch_retrieve.return_value = source_nodes
    mock_format_context.return_value = "fake context"
    mock_batch_generate.return_value = [
        "fake generation response 1",
        "fake generation response 2",
    ]

    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )
    queries = ["fake query 1", "fake query 2"]

    # act
    rag_responses = await rag_system.batch_query(queries=queries)

    # assert
    mock_batch_retrieve.assert_called_with(queries)
    mock_format_context.assert_called_with(source_nodes[1])
    mock_batch_generate.assert_called_with(
        queries, ["fake context", "fake context"]
    )
    assert all(
        res.source_nodes == sn for res, sn in zip(rag_responses, source_nodes)
    )
    assert rag_responses[0].response == "fake generation response 1"
    assert rag_responses[1].response == "fake generation response 2"


@pytest.mark.asyncio
@patch.object(MockRetriever, "encode_query")
async def test_rag_system_retrieve(
    mock_encode_query: AsyncMock,
    mock_generator: BaseGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    mock_encode_query.return_value = torch.Tensor([1.0, 1.0, 1.0])

    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
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
    result = await rag_system.retrieve("fake query")

    # assert
    mock_encode_query.assert_called_with("fake query")
    assert all(res.node == exp.node for res, exp in zip(result, expected))
    assert all(
        abs(res.score - exp.score) < 1e-5 for res, exp in zip(result, expected)
    )


@pytest.mark.asyncio
@patch.object(MockRetriever, "encode_query")
async def test_rag_system_batch_retrieve(
    mock_encode_query: MagicMock,
    mock_generator: BaseGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    mock_encode_query.return_value = torch.Tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )

    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # queries and expected retrieved source nodes
    queries = ["fake query 1", "fake query 2"]
    expected = [
        [
            SourceNode(score=0.8164965809277259, node=knowledge_nodes[0]),
            SourceNode(score=0.8164965809277259, node=knowledge_nodes[2]),
        ],
        [
            SourceNode(score=0.8164965809277259, node=knowledge_nodes[0]),
            SourceNode(score=0.8164965809277259, node=knowledge_nodes[2]),
        ],
    ]

    # act
    result = await rag_system.batch_retrieve(queries)

    # assert
    mock_encode_query.assert_called_with(queries)
    assert all(
        res.node == exp.node
        for res_sample, exp_sample in zip(result, expected)
        for res, exp in zip(res_sample, exp_sample)
    )
    assert all(
        abs(res.score - exp.score) < 1e-5
        for res_sample, exp_sample in zip(result, expected)
        for res, exp in zip(res_sample, exp_sample)
    )


@pytest.mark.asyncio
@patch.object(MockGenerator, "generate")
async def test_rag_system_generate(
    mock_generate: AsyncMock,
    mock_generator: MockGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    mock_generate.return_value = "fake generate response"

    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # act
    res = await rag_system.generate(query="fake query", context="fake context")

    # assert
    mock_generate.assert_called_once_with(
        query="fake query", context="fake context"
    )
    assert res == "fake generate response"


@pytest.mark.asyncio
@patch.object(MockGenerator, "generate")
async def test_rag_system_batch_generate(
    mock_generate: MagicMock,
    mock_generator: MockGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # arrange mocks
    mock_generate.return_value = [
        "fake generate response 1",
        "fake generate response 2",
    ]

    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # queries and contexts
    queries = ["fake query 1", "fake query 2"]
    contexts = ["fake context 1", "fake context 2"]

    # act
    res = await rag_system.batch_generate(queries=queries, contexts=contexts)

    # assert
    mock_generate.assert_called_once_with(query=queries, context=contexts)
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0] == "fake generate response 1"
    assert res[1] == "fake generate response 2"


@pytest.mark.asyncio
@patch.object(MockGenerator, "generate")
async def test_rag_system_batch_generate_list_length_mismatch_error(
    mock_generate: MagicMock,
    mock_generator: MockGenerator,
    mock_retriever: MockRetriever,
) -> None:
    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    # queries and contexts
    queries = ["fake query 1", "fake query 2"]
    contexts = ["fake context 1"]  # only one context provided

    # act + assert
    with pytest.raises(
        RAGSystemError,
        match="Queries and contexts must have the same length for batch generation.",
    ):
        await rag_system.batch_generate(queries=queries, contexts=contexts)

    mock_generate.assert_not_called()


@pytest.mark.asyncio
async def test_rag_system_format_context(
    mock_generator: MockGenerator,
    mock_retriever: MockRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    # build rag system
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
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


@pytest.mark.asyncio
async def test_rag_system_to_sync(
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    knowledge_store = DummyAsyncKnowledgeStore()
    await knowledge_store.load_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = AsyncRAGSystem(
        generator=mock_generator,
        retriever=mock_retriever,
        knowledge_store=knowledge_store,
        rag_config=rag_config,
    )

    sync_rag_system = rag_system.to_sync()

    assert sync_rag_system.generator == rag_system.generator
    assert sync_rag_system.rag_config == rag_system.rag_config
    assert sync_rag_system.knowledge_store.name == knowledge_store.name
