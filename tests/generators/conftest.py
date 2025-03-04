from typing import Any

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from fed_rag.base.generator import BaseGenerator


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
def dummy_pretrained_model_and_tokenizer() -> (
    tuple[PreTrainedModel, PreTrainedTokenizer]
):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    model = _TestHFPretrainedModel(_TestHFConfig())
    return model, tokenizer


class MockGenerator(BaseGenerator):
    def generate(self, input: str) -> str:
        return f"mock output from '{input}'."

    @property
    def model(self) -> torch.nn.Module:
        return torch.nn.Linear(2, 1)


@pytest.fixture
def mock_generator() -> BaseGenerator:
    return MockGenerator()
