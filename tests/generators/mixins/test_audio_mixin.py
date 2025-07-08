import pytest
from pydantic import BaseModel

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.generator_mixins.audio import (
    AudioModalityMixin,
    GeneratorHasAudioModality,
)
from fed_rag.exceptions.generator import GeneratorError

from ..conftest import MockGenerator


class MockAudioGenerator(AudioModalityMixin, MockGenerator):
    pass


def test_audio_mixin() -> None:
    mixed_generator = MockAudioGenerator()
    assert isinstance(mixed_generator, GeneratorHasAudioModality)
    assert isinstance(mixed_generator, BaseGenerator)


def test_audio_mixin_fails_validation() -> None:
    with pytest.raises(
        GeneratorError,
        match="`AudioModalityMixin` must be mixed with `BaseGenerator`.",
    ):

        class InvalidMockAudioGenerator(AudioModalityMixin, BaseModel):
            pass
