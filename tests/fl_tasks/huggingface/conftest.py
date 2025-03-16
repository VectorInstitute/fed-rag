"""PyTorchFLTask Unit Tests"""

from typing import Any, Callable

import pytest
import torch

# huggingface
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from transformers import PretrainedConfig, PreTrainedModel

from fed_rag.decorators import federate
from fed_rag.types import TestResult, TrainResult


@pytest.fixture
def hf_pretrained_model() -> PreTrainedModel:
    class _TestHFConfig(PretrainedConfig):
        model_type = "testmodel"

        def __init__(self, num_hidden: int = 42, **kwargs: Any):
            super().__init__(**kwargs)
            self.num_hidden = num_hidden

    class _TestHFPretrainedModel(PreTrainedModel):
        config_class = _TestHFConfig

        def __init__(self, config: _TestHFConfig):
            super().__init__(config)
            self.config = config
            self.model = torch.nn.Linear(3, self.config.num_hidden)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.model(input)

    return _TestHFPretrainedModel(_TestHFConfig())


@pytest.fixture
def hf_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer(modules=[torch.nn.Linear(5, 5)])


@pytest.fixture
def hf_peft_model() -> PeftModel:
    class MLP(torch.nn.Module):
        """Taken from Peft's documentation, specifially 'Custom Models'.

        https://huggingface.co/docs/peft/main/en/developer_guides/custom_models
        """

        def __init__(self, num_units_hidden: int = 40):
            super().__init__()
            self.seq = torch.nn.Sequential(
                torch.nn.Linear(20, num_units_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(num_units_hidden, num_units_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(num_units_hidden, 2),
                torch.nn.LogSoftmax(dim=-1),
            )

        def forward(self, X: torch.Tensor) -> torch.Tensor:
            return self.seq(X)

    config = LoraConfig(
        target_modules=["seq.0", "seq.2"],
        r=4,
    )

    base_model = MLP()
    return get_peft_model(base_model, config)


# Datasets
@pytest.fixture
def train_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "text1": ["My first sentence", "Another pair"],
            "text2": ["My second sentence", "Unrelated sentence"],
            "label": [0.8, 0.3],
        }
    )


@pytest.fixture
def val_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "text1": ["My third sentence"],
            "text2": ["My fourth sentence"],
            "label": [
                0.42,
            ],
        }
    )


# trainers and testers
@pytest.fixture()
def trainer_pretrained_model() -> Callable:
    @federate.trainer.huggingface
    def fn(
        net: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainResult:
        return TrainResult(loss=0.0)

    return fn  # type: ignore


@pytest.fixture()
def tester_pretrained_model() -> Callable:
    @federate.tester.huggingface
    def fn(
        net: PreTrainedModel,
        test_dataset: Dataset,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore


@pytest.fixture()
def mismatch_tester_pretrained_model() -> Callable:
    @federate.tester.huggingface
    def fn(
        mdl: PreTrainedModel,  # mismatch here
        test_dataset: Dataset,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore


@pytest.fixture()
def undecorated_trainer() -> Callable:
    def fn(
        net: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainResult:
        return TrainResult(loss=0.0)

    return fn  # type: ignore


@pytest.fixture()
def undecorated_tester() -> Callable:
    def fn(
        net: PreTrainedModel,
        test_dataset: Dataset,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore


@pytest.fixture()
def trainer_sentence_transformer() -> Callable:
    @federate.trainer.huggingface
    def fn(
        net: SentenceTransformer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainResult:
        return TrainResult(loss=0.0)

    return fn  # type: ignore


@pytest.fixture()
def tester_sentence_transformer() -> Callable:
    @federate.tester.huggingface
    def fn(
        net: SentenceTransformer,
        test_dataset: Dataset,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore


@pytest.fixture()
def trainer_peft_model() -> Callable:
    @federate.trainer.huggingface
    def fn(
        net: PeftModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ) -> TrainResult:
        return TrainResult(loss=0.0)

    return fn  # type: ignore


@pytest.fixture()
def tester_peft_model() -> Callable:
    @federate.tester.huggingface
    def fn(
        net: PeftModel,
        test_dataset: Dataset,
    ) -> TestResult:
        return TestResult(loss=0.0, metrics={})

    return fn  # type: ignore
