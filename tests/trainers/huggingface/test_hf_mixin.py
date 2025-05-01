import re
import sys
from unittest.mock import patch

import pytest
from datasets import Dataset
from pytest import MonkeyPatch
from transformers import Trainer, TrainingArguments

from fed_rag.base.trainer import BaseRetrieverTrainer, BaseTrainer
from fed_rag.exceptions import MissingExtraError
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerProtocol
from fed_rag.types.rag_system import RAGSystem

from .conftest import TestHFGeneratorTrainer


def test_hf_trainer_init(
    train_dataset: Dataset, hf_rag_system: RAGSystem, monkeypatch: MonkeyPatch
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    # arrange
    training_args = TrainingArguments()
    trainer = TestHFGeneratorTrainer(
        rag_system=hf_rag_system,
        train_dataset=train_dataset,
        training_arguments=training_args,
    )

    assert trainer.rag_system == hf_rag_system
    assert trainer.model == hf_rag_system.generator.model
    assert trainer.train_dataset == train_dataset
    assert trainer.training_arguments == training_args
    assert isinstance(trainer, HuggingFaceTrainerProtocol)
    assert isinstance(trainer, BaseTrainer)


def test_huggingface_extra_missing(
    train_dataset: Dataset, hf_rag_system: RAGSystem, monkeypatch: MonkeyPatch
) -> None:
    # skip validation of rag system
    monkeypatch.setenv("FEDRAG_SKIP_VALIDATION", "1")

    modules = {
        "transformers": None,
    }
    module_to_import = "fed_rag.trainers.huggingface.mixin"
    original_module = sys.modules.pop(module_to_import, None)

    with patch.dict("sys.modules", modules):
        msg = (
            "`TestHFRetrieverTrainer` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.trainers.huggingface.mixin import (
                HuggingFaceTrainerMixin,
            )
            from fed_rag.types.results import TestResult, TrainResult

            class TestHFRetrieverTrainer(
                HuggingFaceTrainerMixin, BaseRetrieverTrainer
            ):
                __test__ = False  # needed for Pytest collision. Avoids PytestCollectionWarning

                def train(self) -> TrainResult:
                    return TrainResult(loss=0.42)

                def evaluate(self) -> TestResult:
                    return TestResult(loss=0.42)

                def hf_trainer_obj(self) -> Trainer:
                    return Trainer()

            training_args = TrainingArguments()
            TestHFRetrieverTrainer(
                rag_system=hf_rag_system,
                train_dataset=train_dataset,
                training_arguments=training_args,
            )

    # restore module so to not affect other tests
    if original_module:
        sys.modules[module_to_import] = original_module
