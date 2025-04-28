import pytest

from fed_rag.base.rag_trainer import RAGTrainMode
from fed_rag.exceptions import UnsupportedTrainerMode
from fed_rag.types.rag_system import RAGSystem

from .conftest import MockRAGTrainer


def test_init(mock_rag_system: RAGSystem) -> None:
    trainer = MockRAGTrainer(rag_system=mock_rag_system, mode="generator")

    assert trainer.rag_system == mock_rag_system
    assert trainer.mode == "generator"


def test_invalid_mode_raises_error(
    mock_rag_system: RAGSystem,
) -> None:
    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        MockRAGTrainer(rag_system=mock_rag_system, mode="both")
