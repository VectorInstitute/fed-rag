import sys
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

from transformers import PreTrainedModel, PreTrainedTokenizer

from fed_rag.base.generator import BaseGenerator
from fed_rag.generators.unsloth import UnslothFastModelGenerator
from fed_rag.tokenizers.unsloth_pretrained_tokenizer import (
    UnslothPretrainedTokenizer,
)


def test_unsloth_pretrained_generator_class() -> None:
    names_of_base_classes = [
        b.__name__ for b in UnslothFastModelGenerator.__mro__
    ]
    assert BaseGenerator.__name__ in names_of_base_classes


@patch.object(UnslothFastModelGenerator, "_load_model_and_tokenizer")
def test_unsloth_pretrained_generator_class_init_delayed_load(
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


@patch.object(UnslothFastModelGenerator, "_load_model_and_tokenizer")
def test_unsloth_pretrained_generator_class_init(
    mock_load_model_and_tokenizer: MagicMock,
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    # arrange
    mock_load_model_and_tokenizer.return_value = (
        dummy_pretrained_model_and_tokenizer
    )

    # act
    generator = UnslothFastModelGenerator(
        model_name="fake_name",
    )
    args, kwargs = mock_load_model_and_tokenizer.call_args

    # assert
    mock_load_model_and_tokenizer.assert_called_once()
    assert generator.model_name == "fake_name"
    assert generator.model == dummy_pretrained_model_and_tokenizer[0]
    assert isinstance(generator.tokenizer, UnslothPretrainedTokenizer)
    assert (
        generator.tokenizer.unwrapped
        == dummy_pretrained_model_and_tokenizer[1]
    )
    assert args == ()
    assert kwargs == {}


def test_unsloth_load_model_and_tokenizer(
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    # mock unsloth module
    mock_fast_lm_cls = MagicMock()
    mock_fast_lm_cls.from_pretrained.return_value = (
        dummy_pretrained_model_and_tokenizer
    )

    mock_unsloth_mod = MagicMock()
    mock_unsloth_mod.__spec__ = (
        MagicMock()
    )  # needed due to Pydantic validations
    mock_unsloth_mod.FastLanguageModel = mock_fast_lm_cls

    modules = {"unsloth": mock_unsloth_mod}
    module_to_import = "unsloth"

    original_module = sys.modules.pop(module_to_import, None)

    try:
        with patch.dict("sys.modules", modules):
            generator = UnslothFastModelGenerator(
                model_name="fake_name",
                load_model_at_init=False,
                load_model_kwargs={"x": 1},
            )

            generator._load_model_and_tokenizer()

            mock_fast_lm_cls.from_pretrained.assert_called_once_with(
                "fake_name", x=1
            )
    finally:
        if original_module is not None:
            sys.modules[module_to_import] = original_module


def test_prompt_setter() -> None:
    # arrange
    generator = UnslothFastModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )

    # act
    generator.prompt_template = "query: {query} and context: {context}"

    # assert
    assert (
        generator.prompt_template.format(query="a", context="b")
        == "query: a and context: b"
    )


def test_unsloth_model_and_tokenizer_setter(
    dummy_pretrained_model_and_tokenizer: tuple[
        PreTrainedModel, PreTrainedTokenizer
    ],
) -> None:
    generator = UnslothFastModelGenerator(
        model_name="fake_name", load_model_at_init=False
    )
    tokenizer = UnslothPretrainedTokenizer(
        dummy_pretrained_model_and_tokenizer[1], "fake_name"
    )

    with does_not_raise():
        # act
        generator.model = dummy_pretrained_model_and_tokenizer[0]
        generator.tokenizer = tokenizer
