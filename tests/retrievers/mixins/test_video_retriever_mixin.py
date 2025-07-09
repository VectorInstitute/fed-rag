from typing import Any

import pytest
import torch
from pydantic import BaseModel, PrivateAttr

from fed_rag.base.retriever import BaseRetriever
from fed_rag.base.retriever_mixins import (
    RetrieverHasVideoModality,
    VideoRetrieverMixin,
)
from fed_rag.exceptions.retriever import RetrieverError

from ..conftest import MockRetriever


class MockMMRetriever(VideoRetrieverMixin, MockRetriever):
    _video_encoder: torch.nn.Module = PrivateAttr(
        default=torch.nn.Linear(2, 1)
    )

    @property
    def video_encoder(self) -> torch.nn.Module | None:
        return self._video_encoder

    def encode_video(
        self, video: Any | list[Any], **kwargs: Any
    ) -> torch.Tensor:
        return self._video_encoder.forward(torch.ones(2))


def test_video_retriever_mixin() -> None:
    mixed_retriever = MockMMRetriever()

    assert isinstance(mixed_retriever, RetrieverHasVideoModality)
    assert isinstance(mixed_retriever, BaseRetriever)


def test_video_retriever_mixin_fails_validation() -> None:
    with pytest.raises(
        RetrieverError,
        match="`VideoRetrieverMixin` must be mixed with `BaseRetriever`.",
    ):

        class InvalidMockMMRetriever(VideoRetrieverMixin, BaseModel):
            _video_encoder: torch.nn.Module = PrivateAttr(
                default=torch.nn.Linear(2, 1)
            )

            @property
            def video_encoder(self) -> torch.nn.Module | None:
                return self._video_encoder

            def encode_video(
                self, video: Any | list[Any], **kwargs: Any
            ) -> torch.Tensor:
                return torch.ones(2)
