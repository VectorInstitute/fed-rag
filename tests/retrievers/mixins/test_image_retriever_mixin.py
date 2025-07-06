from typing import Any

import pytest
import torch
from PIL import Image
from pydantic import BaseModel, PrivateAttr

from fed_rag.base.retriever import BaseRetriever
from fed_rag.base.retriever_mixins import HasImageModality, ImageRetrieverMixin
from fed_rag.exceptions.retriever import RetrieverError

from ..conftest import MockRetriever


class MockMMRetriever(ImageRetrieverMixin, MockRetriever):
    _image_encoder: torch.nn.Module = PrivateAttr(
        default=torch.nn.Linear(2, 1)
    )

    @property
    def image_encoder(self) -> torch.nn.Module | None:
        return self._image_encoder

    def encode_image(
        self, image: Image.Image | list[Image.Image], **kwargs: Any
    ) -> torch.Tensor:
        return self._image_encoder.forward(torch.ones(2))


def test_mixin() -> None:
    mixed_retriever = MockMMRetriever()

    assert isinstance(mixed_retriever, HasImageModality)
    assert isinstance(mixed_retriever, BaseRetriever)


def test_mixin_fails_validation() -> None:
    with pytest.raises(
        RetrieverError,
        match="`ImageRetrieverMixin` must be mixed with `BaseRetriever`.",
    ):

        class InvalidMockMMRetriever(ImageRetrieverMixin, BaseModel):
            _image_encoder: torch.nn.Module = PrivateAttr(
                default=torch.nn.Linear(2, 1)
            )

            @property
            def image_encoder(self) -> torch.nn.Module | None:
                return self._image_encoder

            def encode_image(
                self, image: Image.Image | list[Image.Image], **kwargs: Any
            ) -> torch.Tensor:
                return torch.ones(2)
