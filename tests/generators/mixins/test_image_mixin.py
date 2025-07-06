import pytest
from pydantic import BaseModel

from fed_rag.base.generator import BaseGenerator
from fed_rag.base.generator_mixins import (
    GeneratorHasImageModality,
    ImageModalityMixin,
)
from fed_rag.exceptions.generator import GeneratorError

from ..conftest import MockGenerator


class MockMMGenerator(ImageModalityMixin, MockGenerator):
    pass


def test_mixin() -> None:
    mixed_generator = MockMMGenerator()

    assert isinstance(mixed_generator, GeneratorHasImageModality)
    assert isinstance(mixed_generator, BaseGenerator)


def test_mixin_fails_validation() -> None:
    with pytest.raises(
        GeneratorError,
        match="`ImageModalityMixin` must be mixed with `BaseGenerator`.",
    ):

        class InvalidMockMMGenerator(ImageModalityMixin, BaseModel):
            pass
