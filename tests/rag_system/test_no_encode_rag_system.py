from fed_rag import NoEncodeRAGSystem, RAGConfig
from fed_rag.base.generator import BaseGenerator
from fed_rag.base.no_encode_knowledge_store import BaseNoEncodeKnowledgeStore
from fed_rag.data_structures import KnowledgeNode, NodeType


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


def test_rag_system_init(
    mock_generator: BaseGenerator,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    dummy_store = DummyNoEncodeKnowledgeStore()
    nodes = [
        KnowledgeNode(node_type=NodeType.TEXT, text_content="Dummy text")
        for _ in range(5)
    ]
    dummy_store.load_nodes(nodes)
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
