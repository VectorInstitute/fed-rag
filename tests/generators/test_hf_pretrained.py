from unittest.mock import MagicMock, patch

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fed_rag.base.generator import BaseGenerator
from fed_rag.generators.hf_pretrained_model import HFPretrainedModelGenerator


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in HFPretrainedModelGenerator.__mro__
    ]
    assert BaseGenerator.__name__ in names_of_base_classes


@patch.object(HFPretrainedModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_delayed_load(
    mock_load_from_hf: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    generator = HFPretrainedModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )

    assert generator.model_name == "fake_name"
    assert generator._model is None
    assert generator._tokenizer is None

    # load model
    mock_load_from_hf.return_value = dummy_pretrained_model_and_tokenizer

    generator._load_model_from_hf()
    args, kwargs = mock_load_from_hf.call_args

    mock_load_from_hf.assert_called_once()
    assert generator.model == dummy_pretrained_model_and_tokenizer[0]
    assert generator.tokenizer == dummy_pretrained_model_and_tokenizer[1]
    assert args == ()
    assert kwargs == {}


@patch.object(HFPretrainedModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init(
    mock_load_from_hf: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    # arrange
    mock_load_from_hf.return_value = dummy_pretrained_model_and_tokenizer

    # act
    generator = HFPretrainedModelGenerator(
        model_name="fake_name",
    )
    args, kwargs = mock_load_from_hf.call_args

    # assert
    mock_load_from_hf.assert_called_once()
    assert generator.model_name == "fake_name"
    assert generator.model == dummy_pretrained_model_and_tokenizer[0]
    assert generator.tokenizer == dummy_pretrained_model_and_tokenizer[1]
    assert args == ()
    assert kwargs == {}


@patch.object(HFPretrainedModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_no_load(
    mock_load_from_hf: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    generator = HFPretrainedModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )

    mock_load_from_hf.assert_not_called()
    assert generator.model_name == "fake_name"
    assert generator._model is None
    assert generator._tokenizer is None

    # load model using setter
    model, tokenizer = dummy_pretrained_model_and_tokenizer
    generator.model = model
    generator.tokenizer = tokenizer

    assert generator.model == model
    assert generator.tokenizer == tokenizer


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
    assert generator.tokenizer == tokenizer


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
    generator.tokenizer = mock_tokenizer
    generator.model = mock_model

    # act
    result = generator.generate("fake input", "fake context")

    assert result == "Mock output"
    mock_tokenizer.assert_called_once()
    mock_model.generate.assert_called_once()
