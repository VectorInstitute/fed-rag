from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from fed_rag.data_structures.rag import Context, Prompt, Query
from fed_rag.exceptions.generator import GeneratorError
from fed_rag.generators.unsloth.unsloth_fast_multimodal_model import (
    UnslothFastMultimodalModelGenerator,
)


def dummy_image() -> Image.Image:
    return Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))


def dummy_audio() -> np.ndarray:
    return (np.random.rand(16000) * 2 - 1).astype("float32")


def dummy_video() -> np.ndarray:
    return (np.random.rand(1, 32, 32, 3) * 255).astype("uint8")


@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
def test_unsloth_multimodal_generator_init(mock_load_model):
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_load_model.return_value = (mock_model, mock_processor)

    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")

    assert generator.model_name == "fake-mm-model"
    assert generator._model is not None
    assert generator._processor is not None
    assert isinstance(generator, UnslothFastMultimodalModelGenerator)


def test_pack_messages_batch_mismatch_raises():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    generator.to_query.side_effect = lambda x: (
        x
        if isinstance(x, Query)
        else Query(text=x, images=[], audios=[], videos=[])
    )
    generator.to_context.side_effect = lambda x: (
        x
        if isinstance(x, Context)
        else Context(text=x, images=[], audios=[], videos=[])
    )
    generator._pack_messages = (
        UnslothFastMultimodalModelGenerator._pack_messages.__get__(generator)
    )

    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    qs = [
        Query(text="q1", images=[img], audios=[audio], videos=[video]),
        Query(text="q2", images=[img], audios=[audio], videos=[video]),
    ]
    c = Context(text="ctx1", images=[img], audios=[audio], videos=[video])
    with pytest.raises(
        GeneratorError,
        match="Batch mode requires query and context to be the same length",
    ):
        generator._pack_messages(qs, context=[c])


def test_pack_messages_single_and_batch():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    generator.to_query.side_effect = lambda x: (
        x
        if isinstance(x, Query)
        else Query(text=x, images=[], audios=[], videos=[])
    )
    generator.to_context.side_effect = lambda x: (
        x
        if isinstance(x, Context)
        else Context(text=x, images=[], audios=[], videos=[])
    )
    generator._pack_messages = (
        UnslothFastMultimodalModelGenerator._pack_messages.__get__(generator)
    )

    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    q = Query(text="hello world", images=[img], audios=[audio], videos=[video])
    c = Context(text="ctx", images=[img], audios=[audio], videos=[video])
    messages = generator._pack_messages(q, context=c)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert {"type": "text", "text": "ctx"} in content
    assert {"type": "image", "image": img} in content
    assert any(
        x["type"] == "audio" and np.allclose(x["audio"], audio)
        for x in content
    )
    assert {"type": "video", "video": video} in content
    assert {"type": "text", "text": "hello world"} in content

    qs = [
        Query(text="q1", images=[img], audios=[audio], videos=[video]),
        Query(text="q2", images=[img], audios=[audio], videos=[video]),
    ]
    cs = [
        Context(text="ctx1", images=[img], audios=[audio], videos=[video]),
        Context(text="ctx2", images=[img], audios=[audio], videos=[video]),
    ]

    messages_batch = generator._pack_messages(qs, context=cs)
    assert len(messages_batch) == 2
    for msg, qobj in zip(messages_batch, qs):
        assert {"type": "text", "text": qobj.text} in msg["content"]


@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
def test_generate_returns_batch(mock_model_property, mock_load_model):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.return_value = {
        "input_ids": torch.ones((2, 8), dtype=torch.long)
    }
    mock_model.generate.return_value = torch.ones((2, 10), dtype=torch.long)
    mock_proc.batch_decode.return_value = ["resp1", "resp2"]
    mock_load_model.return_value = (mock_model, mock_proc)

    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")
    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    qs = [
        Query(
            text="what do you see?",
            images=[img],
            audios=[audio],
            videos=[video],
        ),
        Query(
            text="what do you see next?",
            images=[img],
            audios=[audio],
            videos=[video],
        ),
    ]
    cs = [
        Context(text="", images=[img], audios=[audio], videos=[video]),
        Context(text="", images=[img], audios=[audio], videos=[video]),
    ]
    out = generator.generate(query=qs, context=cs)
    assert isinstance(out, list)
    assert out == ["resp1", "resp2"]


def test_to_query_and_to_context_types():
    gen = UnslothFastMultimodalModelGenerator.__new__(
        UnslothFastMultimodalModelGenerator
    )
    assert gen.to_query("abc").text == "abc"
    prompt = Prompt(text="prompt")
    q = gen.to_query(prompt)
    assert isinstance(q, Query) and q.text == "prompt"
    real_q = Query(text="qtext")
    assert gen.to_query(real_q) is real_q
    assert gen.to_context("ctx").text == "ctx"
    ctx = Context(text="ctx_obj")
    assert gen.to_context(ctx) is ctx


@patch("torch.nn.functional.log_softmax")
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
def test_compute_target_sequence_proba_with_modalities(
    mock_model_property,
    mock_load_model,
    mock_log_softmax,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    logits = torch.randn(1, 10, 100)
    mock_model.return_value = MagicMock(logits=logits)
    mock_proc.batch_decode.return_value = ["dummy"]
    mock_load_model.return_value = (mock_model, mock_proc)
    mock_log_softmax.return_value = torch.zeros(100)

    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")
    img_np = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    audio_np = (np.random.rand(16000) * 2 - 1).astype("float32")
    video_np = (np.random.rand(1, 32, 32, 3) * 255).astype("uint8")

    q = Query(
        text="context" + "what is this?",
        images=[dummy_image(), Image.fromarray(img_np)],
        audios=[audio_np, audio_np],
        videos=[video_np, video_np],
    )
    prob = generator.compute_target_sequence_proba(
        prompt=q,
        target="test",
    )
    assert isinstance(prob, torch.Tensor)


def test_pack_messages_with_ndarray_inputs():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    generator.to_query.side_effect = lambda x: (
        x
        if isinstance(x, Query)
        else Query(text=x, images=[], audios=[], videos=[])
    )
    generator.to_context.side_effect = lambda x: (
        x
        if isinstance(x, Context)
        else Context(text=x, images=[], audios=[], videos=[])
    )
    generator._pack_messages = (
        UnslothFastMultimodalModelGenerator._pack_messages.__get__(generator)
    )

    img_np = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    audio_np = dummy_audio()
    video_np = dummy_video()
    q = Query(
        text="hello ndarray",
        images=[dummy_image()],
        audios=[audio_np],
        videos=[video_np],
    )
    c = Context(
        text="ctx ndarray",
        images=[dummy_image()],
        audios=[audio_np],
        videos=[video_np],
    )
    q.images = [img_np]
    c.images = [img_np]

    messages = generator._pack_messages(q, context=c)
    imgs = [x["image"] for x in messages[0]["content"] if x["type"] == "image"]
    assert all(isinstance(im, Image.Image) for im in imgs)
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert any(x["type"] == "image" for x in content)
    assert any(x["type"] == "audio" for x in content)
    assert any(x["type"] == "video" for x in content)


@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
def test_generate_and_complete(mock_model_property, mock_load_model):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.return_value = {
        "input_ids": torch.ones((1, 8), dtype=torch.long)
    }
    mock_model.generate.return_value = torch.ones((1, 10), dtype=torch.long)
    mock_proc.batch_decode.return_value = ["a test response"]
    mock_load_model.return_value = (mock_model, mock_proc)

    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")

    img = dummy_image()
    audio = dummy_audio()
    video = dummy_video()
    q = Query(
        text="what do you see?", images=[img], audios=[audio], videos=[video]
    )
    c = Context(text="", images=[img], audios=[audio], videos=[video])

    out = generator.generate(
        query=q,
        context=c,
        max_new_tokens=4,
    )
    assert isinstance(out, str)
    assert out == "a test response"
    p = Prompt(text="another prompt")
    out2 = generator.complete(prompt=p)
    assert out2 == "a test response"


def test_prompt_template_setter():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    generator._prompt_template = ""
    UnslothFastMultimodalModelGenerator.prompt_template.fset(
        generator, "abc {context}"
    )
    assert generator._prompt_template == "abc {context}"


@patch("torch.nn.functional.log_softmax")
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
def test_compute_target_sequence_proba(
    mock_model_property,
    mock_load_model,
    mock_log_softmax,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    logits = torch.randn(1, 10, 100)
    mock_model.return_value = MagicMock(logits=logits)
    mock_log_softmax.return_value = torch.zeros(100)
    mock_proc.batch_decode.return_value = ["dummy"]
    mock_load_model.return_value = (mock_model, mock_proc)
    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")

    q = Query(
        text="context" + "what is this?",  # concatenate
        images=[],
        audios=[],
        videos=[],
    )
    prob = generator.compute_target_sequence_proba(
        prompt=q,
        target="test",
    )
    assert isinstance(prob, torch.Tensor)


def test_model_property_and_tokenizer():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    fake_model = MagicMock()
    fake_processor = MagicMock()
    fake_tokenizer = MagicMock()
    generator._model = fake_model
    generator._processor = fake_processor
    fake_processor.tokenizer = fake_tokenizer

    assert (
        UnslothFastMultimodalModelGenerator.model.fget(generator) is fake_model
    )
    assert (
        UnslothFastMultimodalModelGenerator.tokenizer.fget(generator)
        is fake_tokenizer
    )
    assert (
        UnslothFastMultimodalModelGenerator.processor.fget(generator)
        is fake_processor
    )


def test_tokenizer_returns_processor_tokenizer():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    fake_processor = MagicMock()
    fake_tokenizer = MagicMock()
    fake_processor.tokenizer = fake_tokenizer
    generator._processor = fake_processor
    assert (
        UnslothFastMultimodalModelGenerator.tokenizer.fget(generator)
        is fake_tokenizer
    )


def test_tokenizer_returns_processor_if_encode():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    fake_processor = MagicMock()
    if hasattr(fake_processor, "tokenizer"):
        del fake_processor.tokenizer
    fake_processor.encode = MagicMock()
    generator._processor = fake_processor
    assert (
        UnslothFastMultimodalModelGenerator.tokenizer.fget(generator)
        is fake_processor
    )


def test_tokenizer_raises_if_neither_tokenizer_nor_encode():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    fake_processor = MagicMock()
    if hasattr(fake_processor, "tokenizer"):
        del fake_processor.tokenizer
    if hasattr(fake_processor, "encode"):
        del fake_processor.encode
    generator._processor = fake_processor
    with pytest.raises(
        AttributeError, match="does not have a `.tokenizer` attribute"
    ):
        UnslothFastMultimodalModelGenerator.tokenizer.fget(generator)


def test_processor_property():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    fake_processor = MagicMock()
    generator._processor = fake_processor
    assert (
        UnslothFastMultimodalModelGenerator.processor.fget(generator)
        is fake_processor
    )


def test_prompt_template_property():
    generator = MagicMock(spec=UnslothFastMultimodalModelGenerator)
    generator._prompt_template = "prompt!"
    assert (
        UnslothFastMultimodalModelGenerator.prompt_template.fget(generator)
        == "prompt!"
    )
    UnslothFastMultimodalModelGenerator.prompt_template.fset(
        generator, "new template"
    )
    assert generator._prompt_template == "new template"


@patch("torch.nn.functional.log_softmax")
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
def test_compute_target_sequence_proba_ndarray_image(
    mock_model_property,
    mock_load_model,
    mock_log_softmax,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    logits = torch.randn(1, 10, 100)
    mock_model.return_value = MagicMock(logits=logits)
    mock_log_softmax.return_value = torch.zeros(100)
    mock_proc.batch_decode.return_value = ["dummy"]
    mock_load_model.return_value = (mock_model, mock_proc)
    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")
    img_np = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    img = Image.fromarray(img_np)
    q = Query(
        text="what is this?",
        images=[img, Image.fromarray(img_np)],
        audios=None,
        videos=None,
    )
    prob = generator.compute_target_sequence_proba(
        prompt=q,
        target="test",
    )
    assert isinstance(prob, torch.Tensor)


@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
def test_generate_raises_generatorerror_on_bad_batch_decode(
    mock_model_property,
    mock_load_model,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.return_value = {
        "input_ids": torch.ones((1, 8), dtype=torch.long)
    }
    mock_model.generate.return_value = torch.ones((1, 10), dtype=torch.long)
    mock_proc.batch_decode.return_value = [1234]
    mock_load_model.return_value = (mock_model, mock_proc)
    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")
    q = Query(text="what do you see?", images=None, audios=None, videos=None)
    c = Context(text="ctx", images=[], audios=[], videos=[])
    with pytest.raises(
        GeneratorError, match="batch_decode did not return valid output"
    ):
        generator.generate(
            query=q,
            context=c,
        )


@patch("torch.nn.functional.log_softmax")
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator.model",
    new_callable=PropertyMock,
)
@pytest.mark.parametrize("model_output", [object(), MagicMock(logits=None)])
def test_compute_target_sequence_proba_raises_on_missing_logits(
    mock_model_property,
    mock_load_model,
    mock_log_softmax,
    model_output,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model_property.return_value = mock_model
    mock_proc.apply_chat_template.side_effect = [
        {"input_ids": torch.arange(10).unsqueeze(0)},
        {"input_ids": torch.arange(5).unsqueeze(0)},
    ]
    mock_model.return_value = model_output
    mock_load_model.return_value = (mock_model, mock_proc)
    generator = UnslothFastMultimodalModelGenerator(model_name="fake-mm-model")
    img = dummy_image()
    q = Query(text="what is this?", images=[img], audios=[], videos=[])
    with pytest.raises(
        GeneratorError,
        match="Underlying model does not expose logits; cannot compute probabilities.",
    ):
        generator.compute_target_sequence_proba(
            prompt=q,
            target="test",
        )


@patch(
    "fed_rag.generators.unsloth.unsloth_fast_multimodal_model.UnslothFastMultimodalModelGenerator._load_model_from_unsloth"
)
def test_lazy_loading_model(
    mock_load_model,
):
    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_load_model.return_value = (mock_model, mock_proc)
    generator = UnslothFastMultimodalModelGenerator(
        model_name="fake-mm-model", load_model_at_init=False
    )
    assert generator._model is None
    _ = generator.model
    assert generator._model is not None
    _ = generator.model
    assert generator._model is not None
