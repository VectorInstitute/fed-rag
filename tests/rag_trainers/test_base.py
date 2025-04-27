from fed_rag.types.rag_system import RAGSystem

from .conftest import MockRAGTrainer


def test_init(mock_rag_system: RAGSystem) -> None:
    trainer = MockRAGTrainer(rag_system=mock_rag_system, mode="generator")

    assert trainer.rag_system == mock_rag_system
    assert trainer.mode == "generator"
