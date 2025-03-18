from unittest.mock import MagicMock, patch

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fed_rag.base.generator import BaseGenerator
from fed_rag.generators.hf_peft_model import HFPeftModelGenerator


def test_hf_peft_generator_class() -> None:
    names_of_base_classes = [b.__name__ for b in HFPeftModelGenerator.__mro__]
    assert BaseGenerator.__name__ in names_of_base_classes


@patch.object(HFPeftModelGenerator, "_load_model_from_hf")
def test_hf_peft_generator_class_init_delayed_load(
    mock_load_from_hf: MagicMock,
    dummy_peft_model_and_tokenizer: tuple[PeftModel, PreTrainedTokenizer],
) -> None:
    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_model_at_init=False,
    )

    assert generator.model_name == "fake_name"
    assert generator.base_model_name == "fake_base_name"
    assert generator._model is None
    assert generator._tokenizer is None

    # load model
    mock_load_from_hf.return_value = dummy_peft_model_and_tokenizer

    generator._load_model_from_hf()
    args, kwargs = mock_load_from_hf.call_args

    mock_load_from_hf.assert_called_once()
    assert generator.model == dummy_peft_model_and_tokenizer[0]
    assert generator.tokenizer == dummy_peft_model_and_tokenizer[1]
    assert args == ()
    assert kwargs == {}


@patch.object(HFPeftModelGenerator, "_load_model_from_hf")
def test_hf_peft_generator_class_init(
    mock_load_from_hf: MagicMock,
    dummy_peft_model_and_tokenizer: tuple[PeftModel, PreTrainedTokenizer],
) -> None:
    # arrange
    mock_load_from_hf.return_value = dummy_peft_model_and_tokenizer

    # act
    generator = HFPeftModelGenerator(
        model_name="fake_name", base_model_name="fake_base_name"
    )
    args, kwargs = mock_load_from_hf.call_args

    # assert
    mock_load_from_hf.assert_called_once()
    assert generator.model_name == "fake_name"
    assert generator.base_model_name == "fake_base_name"
    assert generator.model == dummy_peft_model_and_tokenizer[0]
    assert generator.tokenizer == dummy_peft_model_and_tokenizer[1]
    assert args == ()
    assert kwargs == {}


@patch.object(HFPeftModelGenerator, "_load_model_from_hf")
def test_hf_pretrained_generator_class_init_no_load(
    mock_load_from_hf: MagicMock,
    dummy_peft_model_and_tokenizer: tuple[PeftModel, PreTrainedTokenizer],
) -> None:
    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_model_at_init=False,
    )

    mock_load_from_hf.assert_not_called()
    assert generator.model_name == "fake_name"
    assert generator.base_model_name == "fake_base_name"
    assert generator._model is None
    assert generator._tokenizer is None

    # load model using setter
    model, tokenizer = dummy_peft_model_and_tokenizer
    generator.model = model
    generator.tokenizer = tokenizer

    assert generator.model == model
    assert generator.tokenizer == tokenizer


@patch("fed_rag.generators.hf_peft_model.prepare_model_for_kbit_training")
@patch.object(PeftModel, "from_pretrained")
@patch.object(AutoModelForCausalLM, "from_pretrained")
@patch.object(AutoTokenizer, "from_pretrained")
def test_hf_peft_load_model_from_hf(
    mock_tokenizer_from_pretrained: MagicMock,
    mock_auto_model_from_pretrained: MagicMock,
    mock_peft_model_from_pretrained: MagicMock,
    mock_prepare_model_for_kbit_training: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
    dummy_peft_model_and_tokenizer: tuple[PeftModel, PreTrainedTokenizer],
) -> None:
    # arrange
    model, tokenizer = dummy_peft_model_and_tokenizer
    base_model, _ = dummy_pretrained_model_and_tokenizer
    mock_auto_model_from_pretrained.return_value = base_model
    mock_prepare_model_for_kbit_training.return_value = base_model
    mock_peft_model_from_pretrained.return_value = model
    mock_tokenizer_from_pretrained.return_value = tokenizer

    # act
    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_base_model_kwargs={
            "device_map": "cpu",
            "quantization_config": "fake config",
        },
        load_model_kwargs={"fake_param": "fake_value"},
    )

    # assert
    assert generator.model_name == "fake_name"
    assert generator.base_model_name == "fake_base_name"
    mock_tokenizer_from_pretrained.assert_called_once_with("fake_base_name")
    mock_auto_model_from_pretrained.assert_called_once_with(
        "fake_base_name", device_map="cpu", quantization_config="fake config"
    )
    mock_prepare_model_for_kbit_training.assert_called_once_with(base_model)
    mock_peft_model_from_pretrained.assert_called_once_with(
        base_model, "fake_name", fake_param="fake_value"
    )
    assert generator.model == model
    assert generator.tokenizer == tokenizer


def test_generate() -> None:
    # arrange
    generator = HFPeftModelGenerator(
        model_name="fake_name",
        base_model_name="fake_base_name",
        load_model_at_init=False,
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
