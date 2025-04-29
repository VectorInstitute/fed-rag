import pytest
from datasets import Dataset
from pytest import MonkeyPatch

from fed_rag.exceptions import TrainerError
from fed_rag.trainers.huggingface.lsr import HuggingFaceLSRTrainer
from fed_rag.types.rag_system import RAGSystem


def test_init(
    hf_rag_system: RAGSystem, train_dataset: Dataset, monkeypatch: MonkeyPatch
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceLSRTrainer(
        model=hf_rag_system.retriever.encoder,
        rag_system=hf_rag_system,
        train_dataset=train_dataset,
    )

    assert trainer.train_dataset == train_dataset
    assert trainer.model == hf_rag_system.retriever.encoder
    assert trainer.rag_system == hf_rag_system


def test_invalid_retriever_raises_error(
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    with pytest.raises(
        TrainerError,
        match=(
            "For `HuggingFaceLSRTrainer`, attribute `model` must be of type "
            "`~sentence_transformers.SentenceTransformer`."
        ),
    ):
        HuggingFaceLSRTrainer(
            model=mock_rag_system.retriever.encoder,
            rag_system=mock_rag_system,
            train_dataset=train_dataset,
        )
