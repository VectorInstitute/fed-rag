"""Retriever Mixins."""

from abc import ABC
from typing import Protocol, runtime_checkable

from fed_rag.exceptions.retriever import RetrieverError


@runtime_checkable
class RetrieverHasImageModality(Protocol):
    """Associated protocol for `ImageRetrieverMixin`."""

    __supports_images__: bool = True


class ImageRetrieverMixin(ABC):
    """Image Retriever Mixin.

    Meant to be mixed with a `BaseRetriever` to add image modality for
    retrieval.
    """

    __supports_images__ = True

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseRetriever`."""
        super().__init_subclass__()

        if "BaseRetriever" not in [t.__name__ for t in cls.__mro__]:
            raise RetrieverError(
                "`ImageRetrieverMixin` must be mixed with `BaseRetriever`."
            )
