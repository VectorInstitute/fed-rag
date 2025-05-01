import pytest

from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.base.trainer_manager import RAGTrainMode
from fed_rag.exceptions import UnsupportedTrainerMode

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
