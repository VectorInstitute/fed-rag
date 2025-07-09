"""Retriever Mixins."""

from abc import ABC
from typing import Protocol, runtime_checkable

from fed_rag.exceptions.retriever import RetrieverError


@runtime_checkable
class RetrieverHasVideoModality(Protocol):
    """Associated protocol for `VideoRetrieverMixin`."""

    __supports_video__: bool = True


class VideoRetrieverMixin(ABC):
    """Video Retriever Mixin.

    Meant to be mixed with a `BaseRetriever` to add video modality for
    retrieval.
    """

    __supports_video__ = True

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseRetriever`."""
        super().__init_subclass__()

        if "BaseRetriever" not in [t.__name__ for t in cls.__mro__]:
            raise RetrieverError(
                "`VideoRetrieverMixin` must be mixed with `BaseRetriever`."
            )
