from unittest.mock import MagicMock, patch

from transformers import PreTrainedModel, PreTrainedTokenizer

from fed_rag.base.generator import BaseGenerator
from fed_rag.generators.unsloth import UnslothFastModelGenerator
from fed_rag.tokenizers.unsloth_pretrained_tokenizer import (
    UnslothPretrainedTokenizer,
)


def test_hf_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in UnslothFastModelGenerator.__mro__
    ]
    assert BaseGenerator.__name__ in names_of_base_classes


@patch.object(UnslothFastModelGenerator, "_load_model_and_tokenizer")
def test_hf_pretrained_generator_class_init_delayed_load(
    mock_load_model_and_tokenizer: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    generator = UnslothFastModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )

    assert generator.model_name == "fake_name"
    assert generator._model is None
    assert generator._tokenizer is None

    # load model
    mock_load_model_and_tokenizer.return_value = (
        dummy_pretrained_model_and_tokenizer
    )

    generator._load_model_and_tokenizer()
    args, kwargs = mock_load_model_and_tokenizer.call_args

    mock_load_model_and_tokenizer.assert_called_once()
    assert generator.model == dummy_pretrained_model_and_tokenizer[0]
    assert isinstance(generator.tokenizer, UnslothPretrainedTokenizer)
    assert (
        generator.tokenizer.unwrapped
        == dummy_pretrained_model_and_tokenizer[1]
    )
    assert args == ()
    assert kwargs == {}
