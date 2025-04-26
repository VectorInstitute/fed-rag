import re
import sys
from unittest.mock import MagicMock, patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer

from fed_rag.base.tokenizer import BaseTokenizer
from fed_rag.exceptions import MissingExtraError
from fed_rag.tokenizers.hf_pretrained_tokenizer import HFPretrainedTokenizer


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [b.__name__ for b in HFPretrainedTokenizer.__mro__]
    assert BaseTokenizer.__name__ in names_of_base_classes


@patch.object(HFPretrainedTokenizer, "_load_model_from_hf")
def test_hf_pretrained_tokenizer_class_init_delayed_load(
    mock_load_from_hf: MagicMock, hf_tokenizer: PreTrainedTokenizer
) -> None:
    tokenizer = HFPretrainedTokenizer(
        model_name="fake_name", load_model_at_init=False
    )

    assert tokenizer.model_name == "fake_name"
    assert tokenizer._tokenizer is None

    # load model
    mock_load_from_hf.return_value = hf_tokenizer

    tokenizer._load_model_from_hf()
    args, kwargs = mock_load_from_hf.call_args

    mock_load_from_hf.assert_called_once()
    assert tokenizer.unwrapped == hf_tokenizer
    assert args == ()
    assert kwargs == {}


@patch.object(HFPretrainedTokenizer, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init(
    mock_load_from_hf: MagicMock, hf_tokenizer: PreTrainedTokenizer
) -> None:
    # arrange
    mock_load_from_hf.return_value = hf_tokenizer

    # act
    generator = HFPretrainedTokenizer(
        model_name="fake_name",
    )
    args, kwargs = mock_load_from_hf.call_args

    # assert
    mock_load_from_hf.assert_called_once()
    assert generator.model_name == "fake_name"
    assert generator.unwrapped == hf_tokenizer
    assert args == ()
    assert kwargs == {}


@patch.object(HFPretrainedTokenizer, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_no_load(
    mock_load_from_hf: MagicMock, hf_tokenizer: PreTrainedTokenizer
) -> None:
    tokenizer = HFPretrainedTokenizer(
        model_name="fake_name", load_model_at_init=False
    )

    mock_load_from_hf.assert_not_called()
    assert tokenizer.model_name == "fake_name"
    assert tokenizer._tokenizer is None

    # load model using setter
    tokenizer.unwrapped = hf_tokenizer

    assert tokenizer.unwrapped == hf_tokenizer


@patch.object(AutoTokenizer, "from_pretrained")
def test_hf_pretrained_load_model_from_hf(
    mock_tokenizer_from_pretrained: MagicMock,
    hf_tokenizer: PreTrainedTokenizer,
) -> None:
    # arrange
    mock_tokenizer_from_pretrained.return_value = hf_tokenizer

    # act
    tokenizer = HFPretrainedTokenizer(
        model_name="fake_name", load_model_kwargs={"device_map": "cpu"}
    )

    # assert
    assert tokenizer.model_name == "fake_name"
    mock_tokenizer_from_pretrained.assert_called_once_with(
        "fake_name", device_map="cpu"
    )
    assert tokenizer.unwrapped == hf_tokenizer


def test_encode() -> None:
    # arrange
    tokenizer = HFPretrainedTokenizer(
        model_name="fake_name", load_model_at_init=False
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
    }
    tokenizer.unwrapped = mock_tokenizer

    # act
    result = tokenizer.encode("fake input")

    assert result["input_ids"] == [1, 2]
    assert result["attention_mask"] == [1, 1]
    mock_tokenizer.assert_called_once()


def test_decode() -> None:
    # arrange
    tokenizer = HFPretrainedTokenizer(
        model_name="fake_name", load_model_at_init=False
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "fake decoded"
    tokenizer.unwrapped = mock_tokenizer

    # act
    result = tokenizer.decode([1, 2])

    assert result == "fake decoded"
    mock_tokenizer.decode.assert_called_once_with(token_ids=[1, 2])


def test_huggingface_extra_missing() -> None:
    """Test extra is not installed."""

    modules = {"transformers": None}
    module_to_import = "fed_rag.tokenizers.hf_pretrained_tokenizer"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HFPretrainedTokenizer` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.tokenizers.hf_pretrained_tokenizer import (
                HFPretrainedTokenizer,
            )

            HFPretrainedTokenizer("fake_name")

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module
