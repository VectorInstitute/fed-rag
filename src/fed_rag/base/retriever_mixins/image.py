"""Retriever Mixins."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import torch
from PIL import Image

from fed_rag.exceptions.retriever import RetrieverError


@runtime_checkable
class RetrieverHasImageModality(Protocol):
    """Associated protocol for `ImageRetrieverMixin`."""

    def encode_image(
        self, image: Image.Image | list[Image.Image], **kwargs: Any
    ) -> torch.Tensor:
        ...  # pragma: no cover

    @property
    def image_encoder(self) -> torch.nn.Module | None:
        ...  # pragma: no cover


class ImageRetrieverMixin(ABC):
    """Image Retriever Mixin.

    Meant to be mixed with a `BaseRetriever` to add image modality for
    retrieval.
    """

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseRetriever`."""
        super().__init_subclass__()

        if "BaseRetriever" not in [t.__name__ for t in cls.__mro__]:
            raise RetrieverError(
                "`ImageRetrieverMixin` must be mixed with `BaseRetriever`."
            )

    @abstractmethod
    def encode_image(
        self, image: Image.Image | list[Image.Image], **kwargs: Any
    ) -> torch.Tensor:
        """Encode a PIL Image or a list of PIL Images into a ~torch.Tensor.

        Args:
            image (Image.Image | list[Image.Image]): image or list of images to
                encode.

        Returns:
            torch.Tensor: The encoded representations of the image(s).
        """

    @property
    @abstractmethod
    def image_encoder(self) -> torch.nn.Module | None:
        """PyTorch model associated with the image encoder associated with retriever."""
