from unittest.mock import MagicMock, patch

from transformers import PreTrainedModel, PreTrainedTokenizer

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
