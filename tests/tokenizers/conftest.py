import pytest
import tokenizers
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@pytest.fixture
def dummy_tokenizer() -> PreTrainedTokenizer:
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
