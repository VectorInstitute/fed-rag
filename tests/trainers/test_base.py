from fed_rag.types.rag_system import RAGSystem

from .conftest import MockRetrieverTrainer, MockTrainer


def test_init(mock_rag_system: RAGSystem) -> None:
    trainer = MockTrainer(
        rag_system=mock_rag_system,
        train_dataset=[{"query": "mock example", "response": "mock response"}],
    )

    assert trainer.rag_system == mock_rag_system


def test_retriever_trainer_with_dual_encoder_retriever(
    mock_rag_system_dual_encoder: RAGSystem,
) -> None:
    trainer = MockRetrieverTrainer(
        rag_system=mock_rag_system_dual_encoder,
        train_dataset=[{"query": "mock example", "response": "mock response"}],
    )

    assert trainer.rag_system == mock_rag_system_dual_encoder
    assert (
        trainer.model == mock_rag_system_dual_encoder.retriever.query_encoder
    )
