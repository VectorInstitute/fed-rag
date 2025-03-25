from typing import Any

import pytest
import tokenizers
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from fed_rag.base.tokenizer import BaseTokenizer


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
