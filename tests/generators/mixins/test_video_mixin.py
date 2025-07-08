import pytest
from pydantic import BaseModel

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.generator_mixins.video import (
    GeneratorHasVideoModality,
    VideoModalityMixin,
)
from fed_rag.exceptions.generator import GeneratorError

from ..conftest import MockGenerator


class MockVideoGenerator(VideoModalityMixin, MockGenerator):
    pass


def test_video_mixin() -> None:
    mixed_generator = MockVideoGenerator()
    assert isinstance(mixed_generator, GeneratorHasVideoModality)
    assert isinstance(mixed_generator, BaseGenerator)


def test_video_mixin_fails_validation() -> None:
    with pytest.raises(
        GeneratorError,
        match="`VideoModalityMixin` must be mixed with `BaseGenerator`.",
    ):

        class InvalidMockVideoGenerator(VideoModalityMixin, BaseModel):
            pass
