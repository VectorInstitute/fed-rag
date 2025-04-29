from datasets import Dataset
from pytest import MonkeyPatch

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
