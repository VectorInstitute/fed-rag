from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForImageTextToText

from fed_rag.data_structures.generator import Context, Prompt, Query
from fed_rag.generators.huggingface.hf_multimodal_model import (
    HFMultimodalModelGenerator,
)


def dummy_image() -> Image.Image:
    return Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))


def dummy_audio() -> np.ndarray:
    return (np.random.rand(16000) * 2 - 1).astype("float32")


def dummy_video() -> np.ndarray:
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
    generator.to_query.side_effect = (
        lambda x: Query(text=x) if isinstance(x, str) else x
    )
    generator.to_context.side_effect = (
        lambda x: Context(text=x) if isinstance(x, str) else x
    )
    generator._pack_messages = (
        HFMultimodalModelGenerator._pack_messages.__get__(generator)
    )

    # Single
    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    q = Query(text="hello world", images=[img], audios=[audio], videos=[video])
    c = Context(text="ctx", images=None, audios=None, videos=None)
    messages = generator._pack_messages(q, context=c)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert {"type": "text", "text": "ctx"} in content
    assert {"type": "image", "image": img} in content
    assert {"type": "audio", "audio": audio} in content
    assert {"type": "video", "video": video} in content
    assert {"type": "text", "text": "hello world"} in content

    # Batch
    q1 = Query(text="q1", images=[img], audios=[audio], videos=[video])
    q2 = Query(text="q2", images=None, audios=None, videos=None)
    c1 = Context(text="ctx1", images=None, audios=None, videos=None)
    c2 = Context(text="ctx2", images=None, audios=None, videos=None)
    messages_batch = generator._pack_messages([q1, q2], context=[c1, c2])
    assert len(messages_batch) == 2
    for msg, q in zip(messages_batch, [q1, q2]):
        assert {"type": "text", "text": q.text} in msg["content"]


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
    mock_proc.apply_chat_template.return_value = {
        "input_ids": torch.ones((1, 8), dtype=torch.long)
    }
    mock_model.generate.return_value = torch.ones((1, 10), dtype=torch.long)
    mock_proc.batch_decode.return_value = ["a test response"]

    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")

    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    q = Query(
        text="what do you see?", images=[img], audios=[audio], videos=[video]
    )
    c = Context(text="", images=None, audios=None, videos=None)

    out = generator.generate(
        query=q,
        context=c,
        max_new_tokens=4,
    )
    assert isinstance(out, str)
    assert out == "a test response"
    # Test complete() calls generate with correct args
    p = Prompt(text="another prompt")
    out2 = generator.complete(prompt=p)
    assert out2 == "a test response"


def test_prompt_template_setter():
    generator = MagicMock(spec=HFMultimodalModelGenerator)
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
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    logits = torch.randn(1, 10, 100)
    mock_model.return_value = MagicMock(logits=logits)
    mock_torch_functional.log_softmax.return_value = torch.zeros(100)
    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")
    p = Prompt(text="what is this?", images=None, audios=None, videos=None)
    c = Context(text="context", images=None, audios=None, videos=None)
    prob = generator.compute_target_sequence_proba(
        prompt=p,
        target="test",
        context=c,
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


def test_pack_messages_converts_ndarray_to_image():
    generator = MagicMock(spec=HFMultimodalModelGenerator)
    generator.to_query.side_effect = (
        lambda x: Query(
            text=x, images=[(np.random.rand(32, 32, 3) * 255).astype("uint8")]
        )
        if isinstance(x, str)
        else x
    )
    generator.to_context.side_effect = (
        lambda x: Context(text=x) if isinstance(x, str) else x
    )
    generator._pack_messages = (
        HFMultimodalModelGenerator._pack_messages.__get__(generator)
    )
    # Pass an ndarray
    img_np = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    q = Query(text="test image", images=[img_np])
    messages = generator._pack_messages(q)
    content = messages[0]["content"]
    assert any(
        x["type"] == "image" and isinstance(x["image"], Image.Image)
        for x in content
    )


@patch("fed_rag.generators.huggingface.hf_multimodal_model.F")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoProcessor")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoConfig")
@patch(
    "fed_rag.generators.huggingface.hf_multimodal_model.AutoModelForImageTextToText"
)
def test_compute_target_sequence_proba_ndarray_image(
    mock_auto_model,
    mock_auto_config,
    mock_auto_processor,
    mock_torch_functional,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_auto_processor.from_pretrained.return_value = mock_proc
    mock_auto_config.from_pretrained.return_value = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    logits = torch.randn(1, 10, 100)
    mock_model.return_value = MagicMock(logits=logits)
    mock_torch_functional.log_softmax.return_value = torch.zeros(100)
    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")
    # ndarray instead of PIL.Image
    img_np = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    img = Image.fromarray(img_np)
    q = Query(text="what is this?", images=[img], audios=None, videos=None)
    prob = generator.compute_target_sequence_proba(
        prompt=q,
        target="test",
    )
    assert isinstance(prob, torch.Tensor)


@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoProcessor")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoConfig")
@patch(
    "fed_rag.generators.huggingface.hf_multimodal_model.AutoModelForImageTextToText"
)
def test_generate_raises_runtimeerror_on_bad_batch_decode(
    mock_auto_model, mock_auto_config, mock_auto_processor
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_auto_processor.from_pretrained.return_value = mock_proc
    mock_auto_config.from_pretrained.return_value = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    mock_proc.apply_chat_template.return_value = {
        "input_ids": torch.ones((1, 8), dtype=torch.long)
    }
    mock_model.generate.return_value = torch.ones((1, 10), dtype=torch.long)
    mock_proc.batch_decode.return_value = [1234]  # Not a string!

    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")
    q = Query(text="what do you see?", images=None, audios=None, videos=None)
    c = Context(text="", images=None, audios=None, videos=None)
    with pytest.raises(
        RuntimeError, match="batch_decode did not return valid output"
    ):
        generator.generate(
            query=q,
            context=c,
        )


@patch("fed_rag.generators.huggingface.hf_multimodal_model.F")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoProcessor")
@patch("fed_rag.generators.huggingface.hf_multimodal_model.AutoConfig")
@patch(
    "fed_rag.generators.huggingface.hf_multimodal_model.AutoModelForImageTextToText"
)
@pytest.mark.parametrize("model_output", [object(), MagicMock(logits=None)])
def test_compute_target_sequence_proba_raises_on_missing_logits(
    mock_auto_model,
    mock_auto_config,
    mock_auto_processor,
    mock_torch_functional,
    model_output,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_auto_processor.from_pretrained.return_value = mock_proc
    mock_auto_config.from_pretrained.return_value = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    mock_model.return_value = model_output
    generator = HFMultimodalModelGenerator(model_name="fake-mm-model")
    img = dummy_image()
    q = Query(text="what is this?", images=[img], audios=None, videos=None)
    with pytest.raises(
        RuntimeError,
        match="Underlying model does not expose logits; cannot compute probabilities.",
    ):
        generator.compute_target_sequence_proba(
            prompt=q,
            target="test",
        )


def test_detect_model_class_all_branches():
    class DummyConfig:
        pass

    # Should return AutoModel if nothing
    assert (
        HFMultimodalModelGenerator._detect_model_class(DummyConfig())
        is AutoModel
    )

    # Should return AutoModelForImageTextToText if vision/audio/video config present
    for attr in ["vision_config", "audio_config", "video_config"]:
        c = DummyConfig()
        setattr(c, attr, object())
        assert (
            HFMultimodalModelGenerator._detect_model_class(c)
            is AutoModelForImageTextToText
        )
    # Should return AutoModelForImageTextToText if architectures includes ImageTextToText
    c = DummyConfig()
    c.architectures = ["SomeImageTextToTextModel"]
    assert (
        HFMultimodalModelGenerator._detect_model_class(c)
        is AutoModelForImageTextToText
    )
