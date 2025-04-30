import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch
from transformers.trainer_utils import TrainOutput

from fed_rag.exceptions import (
    InvalidLossError,
    MissingExtraError,
    TrainerError,
)
from fed_rag.trainers.huggingface.lsr import (
    HuggingFaceLSRTrainer,
    LSRSentenceTransformerTrainer,
)
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
    assert isinstance(trainer.hf_trainer_obj, LSRSentenceTransformerTrainer)


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


def test_huggingface_extra_missing(
    train_dataset: Dataset, hf_rag_system: RAGSystem, monkeypatch: MonkeyPatch
) -> None:
    modules = {
        "transformers": None,
    }
    modules_to_import = [
        "fed_rag.trainers.huggingface.mixin",
        "fed_rag.trainers.huggingface.lsr",
    ]
    original_modules = [sys.modules.pop(m, None) for m in modules_to_import]

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceLSRTrainer` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainers.huggingface.lsr import HuggingFaceLSRTrainer

            HuggingFaceLSRTrainer(
                model=hf_rag_system.retriever.encoder,
                rag_system=hf_rag_system,
                train_dataset=train_dataset,
            )

    # restore module so to not affect other tests
    for ix, original_module in enumerate(original_modules):
        if original_module:
            sys.modules[modules_to_import[ix]] = original_module


@patch.object(HuggingFaceLSRTrainer, "hf_trainer_obj")
def test_train(
    mock_hf_trainer: MagicMock,
    hf_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceLSRTrainer(
        model=hf_rag_system.retriever.encoder,
        rag_system=hf_rag_system,
        train_dataset=train_dataset,
    )
    mock_hf_trainer.train.return_value = TrainOutput(
        global_step=42, training_loss=0.42, metrics={}
    )

    out = trainer.train()

    mock_hf_trainer.train.assert_called_once()
    assert out.loss == 0.42


def test_evaluate(
    hf_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceLSRTrainer(
        model=hf_rag_system.retriever.encoder,
        train_dataset=train_dataset,
        rag_system=hf_rag_system,
    )

    with pytest.raises(NotImplementedError):
        trainer.evaluate()


# test LSRSentenceTransformerTrainer
def test_lsr_sentence_transformer_training_missing_extra_error(
    hf_rag_system: RAGSystem,
) -> None:
    modules = {
        "sentence_transformers": None,
    }
    modules_to_import = [
        "fed_rag.trainers.huggingface.lsr",
    ]
    original_modules = [sys.modules.pop(m, None) for m in modules_to_import]

    with patch.dict("sys.modules", modules):
        msg = (
            "`LSRSentenceTransformerTrainer` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainers.huggingface.lsr import (
                LSRSentenceTransformerTrainer,
            )

            LSRSentenceTransformerTrainer(
                model=hf_rag_system.retriever.encoder,
            )

    # restore module so to not affect other tests
    for ix, original_module in enumerate(original_modules):
        if original_module:
            sys.modules[modules_to_import[ix]] = original_module


def test_lsr_sentence_transformer_raises_invalid_loss_error(
    hf_rag_system: RAGSystem,
) -> None:
    with pytest.raises(
        InvalidLossError,
        match="`LSRSentenceTransformerTrainer` must use ~fed_rag.loss.LSRLoss`.",
    ):
        LSRSentenceTransformerTrainer(
            model=hf_rag_system.retriever.encoder,
            loss="loss",
        )


@patch.object(LSRSentenceTransformerTrainer, "collect_scores")
def test_lsr_sentence_transformer_compute_loss(
    mock_collect_scores: MagicMock,
    hf_rag_system: RAGSystem,
) -> None:
    hf_trainer = LSRSentenceTransformerTrainer(
        model=hf_rag_system.retriever.encoder,
    )
    mock_loss = MagicMock()
    hf_trainer.loss = mock_loss
    mock_collect_scores.return_value = 0, 0

    # act
    hf_trainer.compute_loss(
        model=hf_trainer.model, inputs={}, return_outputs=True
    )

    # assert
    mock_collect_scores.assert_called_once()
    mock_loss.assert_called_once()
