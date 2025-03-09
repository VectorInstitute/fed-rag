from fed_rag.base.generator import BaseGenerator
from fed_rag.base.retriever import BaseRetriever
from fed_rag.knowledge_stores.in_memory import InMemoryKnowledgeStore
from fed_rag.types.knowledge_node import KnowledgeNode
from fed_rag.types.rag_system import RAGConfig, RAGSystem


def test_rag_system_init(
    mock_generator: BaseGenerator,
    mock_retriever: BaseRetriever,
    knowledge_nodes: list[KnowledgeNode],
) -> None:
    knowledge_store = InMemoryKnowledgeStore.from_nodes(nodes=knowledge_nodes)
    rag_config = RAGConfig(top_k=2, context_template="{source_nodes}")
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
