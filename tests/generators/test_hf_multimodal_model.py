from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from fed_rag.generators.huggingface.hf_multimodal_model import (
    HFMultimodalModelGenerator,
)


def dummy_image() -> Image.Image:
    return Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))


def dummy_audio() -> np.ndarray:
    return (np.random.rand(16000) * 2 - 1).astype("float32")


def dummy_video() -> np.ndarray:
    # 1 frame of "video"
    return (np.random.rand(1, 32, 32, 3) * 255).astype("uint8")


@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoProcessor")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoConfig")
@patch(
    "fed_rag.generators.huggingface.hf_multimodal_model.AutoModelForImageTextToText"
)
def test_hf_multimodal_generator_init(
    mock_auto_model, mock_auto_config, mock_auto_processor
):
    mock_auto_processor.from_pretrained.return_value = MagicMock()
    mock_auto_config.from_pretrained.return_value = MagicMock()
    mock_auto_model.from_pretrained.return_value = MagicMock()

    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")

    assert generator.model_name == "fake-mm-model"
    assert generator._model is not None
    assert generator._processor is not None
    assert isinstance(generator, HFMultimodalModelGenerator)


def test_pack_messages_single_and_batch():
    generator = MagicMock(spec=HFMultimodalModelGenerator)
    generator._pack_messages = (
        HFMultimodalModelGenerator._pack_messages.__get__(generator)
    )

    # Single
    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    messages = generator._pack_messages(
        "hello world",
        context="ctx",
        images=[img],
        audios=[audio],
        videos=[video],
    )
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert {"type": "text", "text": "ctx"} in content
    assert {"type": "image", "image": img} in content
    assert {"type": "audio", "audio": audio} in content
    assert {"type": "video", "video": video} in content
    assert {"type": "text", "text": "hello world"} in content

    # Batch
    messages_batch = generator._pack_messages(
        ["q1", "q2"],
        context="ctx",
        images=[img],
        audios=[audio],
        videos=[video],
    )
    assert len(messages_batch) == 2
    for msg, q in zip(messages_batch, ["q1", "q2"]):
        assert {"type": "text", "text": q} in msg["content"]


@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoProcessor")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoConfig")
@patch(
    "fed_rag.generators.huggingface.hf_multimodal_model.AutoModelForImageTextToText"
)
def test_generate_and_complete(
    mock_auto_model, mock_auto_config, mock_auto_processor
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_auto_processor.from_pretrained.return_value = mock_proc
    mock_auto_config.from_pretrained.return_value = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    # Simulate apply_chat_template
    mock_proc.apply_chat_template.return_value = {
        "input_ids": torch.ones((1, 8), dtype=torch.long)
    }
    mock_model.generate.return_value = torch.ones((1, 10), dtype=torch.long)
    mock_proc.batch_decode.return_value = ["a test response"]

    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")

    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()

    out = generator.generate(
        query="what do you see?",
        context="",
        images=[img],
        audios=[audio],
        videos=[video],
        max_new_tokens=4,
    )
    assert isinstance(out, str)
    assert out == "a test response"
    # Test complete() calls generate with correct args
    out2 = generator.complete(prompt="another prompt")
    assert out2 == "a test response"


def test_prompt_template_setter():
    generator = MagicMock(spec=HFMultimodalModelGenerator)
    # Patch the _prompt_template to ensure setter works
    generator._prompt_template = ""
    HFMultimodalModelGenerator.prompt_template.fset(generator, "abc {context}")
    assert generator._prompt_template == "abc {context}"


@patch("fed_rag.generators.huggingface.hf_multimodal_model.F")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoProcessor")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoConfig")
@patch(
    "fed_rag.generators.huggingface.hf_multimodal_model.AutoModelForImageTextToText"
)
def test_compute_target_sequence_proba(
    mock_auto_model,
    mock_auto_config,
    mock_auto_processor,
    mock_torch_functional,
):
    # Mock model, processor, logits
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_auto_processor.from_pretrained.return_value = mock_proc
    mock_auto_config.from_pretrained.return_value = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {
            "input_ids": torch.arange(10).unsqueeze(0)
        },  # input_ids for full prompt+target
        {
            "input_ids": torch.arange(5).unsqueeze(0)
        },  # input_ids for just prompt
    ]
    logits = torch.randn(1, 10, 100)
    mock_model.return_value = MagicMock(logits=logits)
    mock_torch_functional.log_softmax.return_value = torch.zeros(100)
    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")
    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    prob = generator.compute_target_sequence_proba(
        prompt="what is this?",
        target="test",
        context="context",
        images=[img],
        audios=[audio],
        videos=[video],
    )
    assert isinstance(prob, torch.Tensor)


def test_model_property_and_tokenizer():
    generator = MagicMock(spec=HFMultimodalModelGenerator)
    fake_model = MagicMock()
    fake_processor = MagicMock()
    generator._model = fake_model
    generator._processor = fake_processor
    # direct property access
    assert HFMultimodalModelGenerator.model.fget(generator) is fake_model
    assert (
        HFMultimodalModelGenerator.tokenizer.fget(generator) is fake_processor
    )
    assert (
        HFMultimodalModelGenerator.processor.fget(generator) is fake_processor
    )


def test_prompt_template_property():
    generator = MagicMock(spec=HFMultimodalModelGenerator)
    generator._prompt_template = "prompt!"
    assert (
        HFMultimodalModelGenerator.prompt_template.fget(generator) == "prompt!"
    )
    HFMultimodalModelGenerator.prompt_template.fset(generator, "new template")
    assert generator._prompt_template == "new template"
