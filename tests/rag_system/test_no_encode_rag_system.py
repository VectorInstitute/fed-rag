from unittest.mock import MagicMock, patch

import pytest

from fed_rag import NoEncodeRAGSystem, RAGConfig
from fed_rag.base.bridge import BaseBridgeMixin
from fed_rag.base.generator import BaseGenerator
from fed_rag.base.no_encode_knowledge_store import BaseNoEncodeKnowledgeStore
from fed_rag.data_structures import KnowledgeNode, NodeType, SourceNode

from .conftest import MockGenerator


class DummyNoEncodeKnowledgeStore(BaseNoEncodeKnowledgeStore):
    nodes: list[KnowledgeNode] = []

    def load_node(self, node: KnowledgeNode) -> None:
        self.nodes.append(node)

    def load_nodes(self, nodes: list[KnowledgeNode]) -> None:
        for n in nodes:
            self.load_node(n)

    def retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[float, KnowledgeNode]]:
        return [(ix, n) for ix, n in enumerate(self.nodes[:top_k])]

    def delete_node(self, node_id: str) -> bool:
        return True

    def clear(self) -> None:
        self.nodes.clear()

    def count(self) -> int:
        return len(self.nodes)

    def persist(self) -> None:
        pass

    def load(self) -> None:
        pass


@pytest.fixture()
def dummy_store() -> BaseNoEncodeKnowledgeStore:
    dummy_store = DummyNoEncodeKnowledgeStore()
    nodes = [
        KnowledgeNode(node_type=NodeType.TEXT, text_content="Dummy text")
        for _ in range(5)
    ]
    dummy_store.load_nodes(nodes)
    return dummy_store


def test_rag_system_init(
    mock_generator: BaseGenerator,
    knowledge_nodes: list[KnowledgeNode],
    dummy_store: BaseNoEncodeKnowledgeStore,
) -> None:
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = NoEncodeRAGSystem(
        generator=mock_generator,
        knowledge_store=dummy_store,
        rag_config=rag_config,
    )
    assert rag_system.knowledge_store == dummy_store
    assert rag_system.rag_config == rag_config
    assert rag_system.generator == mock_generator


@patch.object(NoEncodeRAGSystem, "generate")
@patch.object(NoEncodeRAGSystem, "_format_context")
@patch.object(NoEncodeRAGSystem, "retrieve")
def test_rag_system_query(
    mock_retrieve: MagicMock,
    mock_format_context: MagicMock,
    mock_generate: MagicMock,
    mock_generator: BaseGenerator,
    knowledge_nodes: list[KnowledgeNode],
    dummy_store: BaseNoEncodeKnowledgeStore,
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
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = NoEncodeRAGSystem(
        generator=mock_generator,
        knowledge_store=dummy_store,
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


@patch.object(MockGenerator, "generate")
def test_rag_system_generate(
    mock_generate: MagicMock,
    mock_generator: MockGenerator,
    dummy_store: BaseNoEncodeKnowledgeStore,
) -> None:
    # arrange mocks
    mock_generate.return_value = "fake generate response"

    # build rag system
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = NoEncodeRAGSystem(
        generator=mock_generator,
        knowledge_store=dummy_store,
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
    dummy_store: BaseNoEncodeKnowledgeStore,
) -> None:
    # build rag system
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = NoEncodeRAGSystem(
        generator=mock_generator,
        knowledge_store=dummy_store,
        rag_config=rag_config,
    )

    # act
    source_nodes = rag_system.retrieve("mock query")
    formatted_context = rag_system._format_context(source_nodes)

    # assert
    assert formatted_context == "Dummy text\nDummy text"


def test_bridging_no_encode_rag_system(
    mock_generator: MockGenerator,
    dummy_store: BaseNoEncodeKnowledgeStore,
) -> None:
    # arrange

    # create test/mock bridge
    class _TestBridgeMixin(BaseBridgeMixin):
        _bridge_version = "0.1.0"
        _bridge_extra = "my-bridge"
        _framework = "my-bridge-framework"
        _compatible_versions = {"min": "0.1.2", "max": "0.2.0"}
        _method_name = "to_bridge"

        def to_bridge(self) -> None:
            self._validate_framework_installed()
            return None

    class BridgedNoEncodeRAGSystem(_TestBridgeMixin, NoEncodeRAGSystem):
        pass

    # build bridged rag system
    rag_config = RAGConfig(
        top_k=2,
    )
    rag_system = BridgedNoEncodeRAGSystem(
        generator=mock_generator,
        knowledge_store=dummy_store,
        rag_config=rag_config,
    )

    assert _TestBridgeMixin._framework in BridgedNoEncodeRAGSystem.bridges
    assert (
        rag_system.bridges["my-bridge-framework"]
        == _TestBridgeMixin.get_bridge_metadata()
    )

    # cleanup
    del BridgedNoEncodeRAGSystem.bridges[_TestBridgeMixin._framework]
