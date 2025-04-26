import re
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.testing import assert_close
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.tokenizer import EncodeResult
from fed_rag.exceptions import MissingExtraError
from fed_rag.generators.huggingface import HFPretrainedModelGenerator
from fed_rag.tokenizers.hf_pretrained_tokenizer import HFPretrainedTokenizer


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HFPretrainedModelGenerator.__mro__
    ]
    assert BaseGenerator.__name__ in names_of_base_classes


@patch.object(HFPretrainedTokenizer, "_load_model_from_hf")
@patch.object(HFPretrainedModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_delayed_load(
    mock_load_from_hf: MagicMock,
    mock_load_from_hf_tokenizer: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    generator = HFPretrainedModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )

    assert generator.model_name == "fake_name"
    assert generator._model is None
    assert generator._tokenizer is not None

    # load model
    mock_load_from_hf_tokenizer.return_value = (
        dummy_pretrained_model_and_tokenizer[1]
    )
    mock_load_from_hf.return_value = dummy_pretrained_model_and_tokenizer[0]

    generator._load_model_from_hf()
    args, kwargs = mock_load_from_hf.call_args

    mock_load_from_hf.assert_called_once()
    assert generator.model == dummy_pretrained_model_and_tokenizer[0]
    assert isinstance(generator.tokenizer, HFPretrainedTokenizer)
    assert (
        generator.tokenizer.unwrapped
        == dummy_pretrained_model_and_tokenizer[1]
    )
    assert args == ()
    assert kwargs == {}


@patch.object(HFPretrainedTokenizer, "_load_model_from_hf")
@patch.object(HFPretrainedModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init(
    mock_load_from_hf: MagicMock,
    mock_load_from_hf_tokenizer: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    # arrange
    mock_load_from_hf.return_value = dummy_pretrained_model_and_tokenizer[0]
    mock_load_from_hf_tokenizer.return_value = (
        dummy_pretrained_model_and_tokenizer[1]
    )

    # act
    generator = HFPretrainedModelGenerator(
        model_name="fake_name",
    )
    args, kwargs = mock_load_from_hf.call_args

    # assert
    mock_load_from_hf.assert_called_once()
    mock_load_from_hf_tokenizer.assert_called_once()
    assert generator.model_name == "fake_name"
    assert generator.model == dummy_pretrained_model_and_tokenizer[0]
    assert isinstance(generator.tokenizer, HFPretrainedTokenizer)
    assert (
        generator.tokenizer.unwrapped
        == dummy_pretrained_model_and_tokenizer[1]
    )
    assert args == ()
    assert kwargs == {}


@patch.object(HFPretrainedTokenizer, "_load_model_from_hf")
@patch.object(HFPretrainedModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_no_load(
    mock_load_from_hf: MagicMock,
    mock_load_from_hf_tokenizer: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    generator = HFPretrainedModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )

    mock_load_from_hf.assert_not_called()
    mock_load_from_hf_tokenizer.assert_not_called()
    assert generator.model_name == "fake_name"
    assert generator._model is None
    assert generator._tokenizer is not None
    assert isinstance(generator.tokenizer, HFPretrainedTokenizer)

    # load model using setter
    model, tokenizer = dummy_pretrained_model_and_tokenizer
    generator.model = model
    generator.tokenizer.unwrapped = tokenizer

    assert generator.model == model
    assert generator.tokenizer.unwrapped == tokenizer


@patch.object(AutoModelForCausalLM, "from_pretrained")
@patch.object(AutoTokenizer, "from_pretrained")
def test_hf_pretrained_load_model_from_hf(
    mock_tokenizer_from_pretrained: MagicMock,
    mock_model_from_pretrained: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    # arrange
    model, tokenizer = dummy_pretrained_model_and_tokenizer
    mock_model_from_pretrained.return_value = model
    mock_tokenizer_from_pretrained.return_value = tokenizer

    # act
    generator = HFPretrainedModelGenerator(
        model_name="fake_name", load_model_kwargs={"device_map": "cpu"}
    )

    # assert
    assert generator.model_name == "fake_name"
    mock_tokenizer_from_pretrained.assert_called_once()
    mock_model_from_pretrained.assert_called_once_with(
        "fake_name", device_map="cpu"
    )
    assert generator.model == model
    assert generator.tokenizer.unwrapped == tokenizer


def test_generate() -> None:
    # arrange
    generator = HFPretrainedModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = torch.Tensor([1, 2, 3])
    mock_tokenizer_result = MagicMock()
    mock_tokenizer_result.input_ids = torch.ones(2)
    mock_tokenizer.batch_decode.return_value = ["Mock output"]
    mock_tokenizer.return_value = mock_tokenizer_result
    generator.tokenizer.unwrapped = mock_tokenizer
    generator.model = mock_model

    # act
    result = generator.generate("fake input", "fake context")

    assert result == "Mock output"
    mock_tokenizer.assert_called_once()
    mock_model.generate.assert_called_once()


@patch("fed_rag.generators.huggingface.mixin.F")
def test_compute_target_sequence_proba(
    mock_torch_functional: MagicMock,
) -> None:
    # arrange
    generator = HFPretrainedModelGenerator(
        model_name="fake_name",
        load_model_at_init=False,
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = [
        EncodeResult(input_ids=[0, 1, 2, 3, 4]),
        EncodeResult(input_ids=[0, 1, 2]),
    ]
    mock_torch_functional.softmax.return_value = torch.tensor(
        [0.2, 0.2, 0.2, 0.2, 0.2]
    )
    mock_model = MagicMock()
    generator.model = mock_model
    generator.tokenizer = mock_tokenizer

    # act
    result = generator.compute_target_sequence_proba(
        prompt="fake prompt", target=" fake target"
    )

    mock_tokenizer.encode.assert_any_call("fake prompt fake target")
    mock_tokenizer.encode.assert_any_call("fake prompt")
    assert mock_torch_functional.softmax.call_count == 2
    mock_model.assert_called_once()
    assert_close(result, torch.tensor(0.2**2))


def test_huggingface_extra_missing() -> None:
    """Test extra is not installed."""

    modules = {"transformers": None}
    module_to_import = "fed_rag.generators.huggingface.hf_pretrained_model"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "`HFPretrainedModelGenerator` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.generators.huggingface.hf_pretrained_model import (
                HFPretrainedModelGenerator,
            )

            HFPretrainedModelGenerator("fake_name")

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module


def test_huggingface_extra_missing_imported_from_parent() -> None:
    """Test extra is not installed."""

    modules = {"transformers": None}
    module_to_import = "fed_rag.generators.huggingface"

    if module_to_import in sys.modules:
        original_module = sys.modules.pop(module_to_import)

    with patch.dict("sys.modules", modules):
        msg = (
            "`fed_rag.generators.huggingface` requires `huggingface` extra to be installed. "
            "To fix please run `pip install fed-rag[huggingface]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            from fed_rag.generators.huggingface import (
                HFPretrainedModelGenerator,
            )

            HFPretrainedModelGenerator("fake_name")

    # restore module so to not affect other tests
    sys.modules[module_to_import] = original_module
