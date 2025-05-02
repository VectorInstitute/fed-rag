import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch
from transformers import Trainer
from transformers.trainer_utils import TrainOutput

from fed_rag.exceptions import FedRAGError, MissingExtraError
from fed_rag.trainers.huggingface.ralt import HuggingFaceTrainerForRALT
from fed_rag.types.rag_system import RAGSystem


def test_init(
    hf_rag_system: RAGSystem, train_dataset: Dataset, monkeypatch: MonkeyPatch
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceTrainerForRALT(
        rag_system=hf_rag_system,
        train_dataset=train_dataset,
    )

    assert trainer.train_dataset == train_dataset
    assert trainer.model == hf_rag_system.generator.model
    assert trainer.rag_system == hf_rag_system
    assert isinstance(trainer.hf_trainer_obj, Trainer)


def test_init_raises_invalid_rag_system_error(
    mock_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    with pytest.raises(FedRAGError):
        HuggingFaceTrainerForRALT(
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
        "fed_rag.trainers.huggingface.ralt",
    ]
    original_modules = [sys.modules.pop(m, None) for m in modules_to_import]

    with patch.dict("sys.modules", modules):
        msg = (
            "`HuggingFaceTrainerForRALT` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainers.huggingface.ralt import (
                HuggingFaceTrainerForRALT,
            )

            HuggingFaceTrainerForRALT(
                rag_system=hf_rag_system,
                train_dataset=train_dataset,
            )

    # restore module so to not affect other tests
    for ix, original_module in enumerate(original_modules):
        if original_module:
            sys.modules[modules_to_import[ix]] = original_module


@patch.object(HuggingFaceTrainerForRALT, "hf_trainer_obj")
def test_train(
    mock_hf_trainer: MagicMock,
    hf_rag_system: RAGSystem,
    train_dataset: Dataset,
    monkeypatch: MonkeyPatch,
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    trainer = HuggingFaceTrainerForRALT(
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

    trainer = HuggingFaceTrainerForRALT(
        train_dataset=train_dataset,
        rag_system=hf_rag_system,
    )

    with pytest.raises(NotImplementedError):
        trainer.evaluate()
