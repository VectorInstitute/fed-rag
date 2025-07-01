import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from fed_rag.base.generator import BaseGenerator
from fed_rag.generators.huggingface.gemma3n_generator import Gemma3nGenerator


def _has_hf_token():
    return bool(
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )


def test_gemma3n_generator_inherits_base() -> None:
    names_of_base_classes = [b.__name__ for b in Gemma3nGenerator.__mro__]
    assert BaseGenerator.__name__ in names_of_base_classes


@patch("fed_rag.generators.huggingface.gemma3n_generator.AutoProcessor")
@patch(
    "fed_rag.generators.huggingface.gemma3n_generator.AutoModelForImageTextToText"
)
def test_gemma3n_generator_init(
    mock_model_class: MagicMock,
    mock_processor_class: MagicMock,
) -> None:
    # Arrange
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_processor = MagicMock()
    mock_processor_class.from_pretrained.return_value = mock_processor
    mock_model_class.from_pretrained.return_value = mock_model

    # Act
    generator = Gemma3nGenerator(model_name="mock-model", device="cpu")

    # Assert
    mock_processor_class.from_pretrained.assert_called_once_with("mock-model")
    mock_model_class.from_pretrained.assert_called_once_with("mock-model")
    mock_model.to.assert_called_once_with("cpu")
    assert generator.model == mock_model
    assert generator.tokenizer == mock_processor
    assert generator._device == "cpu"


@patch("fed_rag.generators.huggingface.gemma3n_generator.AutoProcessor")
@patch(
    "fed_rag.generators.huggingface.gemma3n_generator.AutoModelForImageTextToText"
)
def test_gemma3n_generator_generate(
    mock_model_class: MagicMock,
    mock_processor_class: MagicMock,
) -> None:
    # Arrange
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.to.return_value = mock_model
    mock_model.generate.return_value = torch.tensor([[100, 200, 300, 400]])
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4]])
    }
    mock_processor.batch_decode.return_value = ["A"]
    mock_processor_class.from_pretrained.return_value = mock_processor
    mock_model_class.from_pretrained.return_value = mock_model

    generator = Gemma3nGenerator(model_name="mock-model", device="cpu")

    # Act
    result = generator.generate(query="What is the capital of France?")

    # Assert
    mock_processor.apply_chat_template.assert_called()
    mock_model.generate.assert_called()
    mock_processor.batch_decode.assert_called()
    assert result == "A"


@pytest.mark.skipif(
    not _has_hf_token(),
    reason="Hugging Face token not set. Skipping test that requires a remote call.",
)
def test_gemma3n_generator_prompt_template_property() -> None:
    generator = Gemma3nGenerator(
        model_name="google/gemma-3n-e2b-it", device="cpu"
    )
    assert hasattr(generator, "prompt_template")
    assert generator.prompt_template == ""


def test_generate_raises_on_empty_decode():
    gen = Gemma3nGenerator.__new__(Gemma3nGenerator)
    gen._processor = MagicMock()
    gen._model = MagicMock()
    gen._device = "cpu"
    gen._model.device = torch.device("cpu")
    gen._processor.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4]])
    }
    gen._model.generate.return_value = torch.tensor([[100, 200, 300, 400]])
    gen._processor.batch_decode.return_value = []  # Simulate empty decode
    with pytest.raises(
        RuntimeError,
        match="batch_decode did not return a non-empty list of strings",
    ):
        gen.generate("prompt")


def test_prompt_template_property():
    gen = Gemma3nGenerator.__new__(Gemma3nGenerator)
    # Directly test property (no remote/model needed)
    assert gen.prompt_template == ""


def test_complete_not_implemented():
    gen = Gemma3nGenerator.__new__(Gemma3nGenerator)
    with pytest.raises(NotImplementedError):
        gen.complete()


def test_compute_target_sequence_proba_not_implemented():
    gen = Gemma3nGenerator.__new__(Gemma3nGenerator)
    with pytest.raises(NotImplementedError):
        gen.compute_target_sequence_proba()


@pytest.mark.skipif(
    not _has_hf_token(),
    reason="Hugging Face token not set. Skipping test that requires a remote call.",
)
def test_gemma3n_generator_not_implemented_methods() -> None:
    generator = Gemma3nGenerator(
        model_name="google/gemma-3n-e2b-it", device="cpu"
    )
    with pytest.raises(NotImplementedError):
        generator.complete()
    with pytest.raises(NotImplementedError):
        generator.compute_target_sequence_proba()
