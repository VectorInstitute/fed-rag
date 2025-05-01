import pytest

from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.base.trainer_manager import RAGTrainMode
from fed_rag.exceptions import InconsistentRAGSystems, UnsupportedTrainerMode
from fed_rag.types.rag_system import RAGSystem

from .conftest import MockRAGTrainerManager


def test_init(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    trainer = MockRAGTrainerManager(
        generator_trainer=generator_trainer,
        retriever_trainer=retriever_trainer,
        mode="generator",
    )

    assert trainer.retriever_trainer == retriever_trainer
    assert trainer.generator_trainer == generator_trainer
    assert trainer.mode == "generator"


def test_invalid_mode_raises_error(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
) -> None:
    msg = (
        f"Unsupported RAG train mode: both. "
        f"Mode must be one of: {', '.join([m.value for m in RAGTrainMode])}"
    )
    with pytest.raises(UnsupportedTrainerMode, match=msg):
        MockRAGTrainerManager(
            generator_trainer=generator_trainer,
            retriever_trainer=retriever_trainer,
            mode="both",
        )


def test_inconsistent_rag_systems_raises_error(
    generator_trainer: BaseGeneratorTrainer,
    retriever_trainer: BaseRetrieverTrainer,
    another_mock_rag_system: RAGSystem,
) -> None:
    msg = (
        "Inconsistent RAG systems detected between retriever and generator trainers. "
        "Both trainers must use the same RAG system instance for consistent training."
    )
    with pytest.raises(InconsistentRAGSystems, match=msg):
        generator_trainer.rag_system = another_mock_rag_system
        MockRAGTrainerManager(
            generator_trainer=generator_trainer,
            retriever_trainer=retriever_trainer,
            mode="generator",
        )
