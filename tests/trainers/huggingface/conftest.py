from typing import Any

import pytest
import tokenizers
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer, models
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
)

from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.base.trainer import BaseGeneratorTrainer, BaseRetrieverTrainer
from fed_rag.trainers.huggingface.mixin import HuggingFaceTrainerMixin
from fed_rag.types.rag_system import RAGSystem
from fed_rag.types.results import TestResult, TrainResult


class MockTokenizer(BaseTokenizer):
    def encode(self, input: str, **kwargs: Any) -> list[int]:
        return [0, 1, 2]

    def decode(self, input_ids: list[int], **kwargs: Any) -> str:
        return "mock decoded sentence"

    @property
    def unwrapped(self) -> None:
        return None


@pytest.fixture()
def mock_tokenizer() -> BaseTokenizer:
    return MockTokenizer()


@pytest.fixture
def hf_tokenizer() -> PreTrainedTokenizer:
    tokenizer = Tokenizer(
        models.WordPiece({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


class TestHFRetrieverTrainer(HuggingFaceTrainerMixin, BaseRetrieverTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)

    def hf_trainer_obj(self) -> Trainer:
        return Trainer()


class TestHFGeneratorTrainer(HuggingFaceTrainerMixin, BaseGeneratorTrainer):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )

    def train(self) -> TrainResult:
        return TrainResult(loss=0.42)

    def evaluate(self) -> TestResult:
        return TestResult(loss=0.42)

    def hf_trainer_obj(self) -> Trainer:
        return Trainer()


@pytest.fixture()
def train_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "query": ["first query", "second query"],
            "response": ["first response", "second response"],
        }
    )


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


@pytest.fixture
def dummy_tokenizer() -> PreTrainedTokenizer:
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture
def dummy_pretrained_model_and_tokenizer(
    dummy_tokenizer: PreTrainedTokenizer,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    return _TestHFPretrainedModel(_TestHFConfig()), dummy_tokenizer


@pytest.fixture()
def hf_rag_system(
    mock_rag_system: RAGSystem, dummy_tokenizer: PreTrainedTokenizer
) -> RAGSystem:
    encoder = SentenceTransformer(modules=[torch.nn.Linear(5, 5)])
    # Mock the tokenize method on the first module
    encoder.tokenizer = None
    encoder._first_module().tokenize = lambda texts: {
        "input_ids": torch.ones((len(texts), 10))
    }
    encoder.encode = lambda texts, **kwargs: torch.ones(
        (len(texts) if isinstance(texts, list) else 1, 5)
    )

    mock_rag_system.retriever.encoder = encoder

    dummy_pretrained_model = _TestHFPretrainedModel(_TestHFConfig())
    mock_rag_system.generator.model = dummy_pretrained_model
    mock_rag_system.generator.tokenizer.unwrapped = dummy_tokenizer

    return mock_rag_system
