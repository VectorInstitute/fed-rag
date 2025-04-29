from fed_rag.types.rag_system import RAGSystem

from .conftest import MockTrainer


def test_init(mock_rag_system: RAGSystem) -> None:
    trainer = MockTrainer(rag_system=mock_rag_system)

    assert trainer.rag_system == mock_rag_system
