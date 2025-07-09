from typing import Any

import pytest
import torch
from pydantic import BaseModel, PrivateAttr

from fed_rag.base.retriever import BaseRetriever
from fed_rag.base.retriever_mixins import (
    AudioRetrieverMixin,
    RetrieverHasAudioModality,
)
from fed_rag.exceptions.retriever import RetrieverError

from ..conftest import MockRetriever


class MockMMRetriever(AudioRetrieverMixin, MockRetriever):
    _audio_encoder: torch.nn.Module = PrivateAttr(
        default=torch.nn.Linear(2, 1)
    )

    @property
    def audio_encoder(self) -> torch.nn.Module | None:
        return self._audio_encoder

    def encode_audio(
        self, audio: Any | list[Any], **kwargs: Any
    ) -> torch.Tensor:
        return self._audio_encoder.forward(torch.ones(2))


def test_audio_retriever_mixin() -> None:
    mixed_retriever = MockMMRetriever()

    assert isinstance(mixed_retriever, RetrieverHasAudioModality)
    assert isinstance(mixed_retriever, BaseRetriever)


def test_audio_retriever_mixin_fails_validation() -> None:
    with pytest.raises(
        RetrieverError,
        match="`AudioRetrieverMixin` must be mixed with `BaseRetriever`.",
    ):

        class InvalidMockMMRetriever(AudioRetrieverMixin, BaseModel):
            _audio_encoder: torch.nn.Module = PrivateAttr(
                default=torch.nn.Linear(2, 1)
            )

            @property
            def audio_encoder(self) -> torch.nn.Module | None:
                return self._audio_encoder

            def encode_audio(
                self, audio: Any | list[Any], **kwargs: Any
            ) -> torch.Tensor:
                return torch.ones(2)
