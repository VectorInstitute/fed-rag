"""Retriever Mixins."""

from abc import ABC
from typing import Protocol, runtime_checkable

from fed_rag.exceptions.retriever import RetrieverError


@runtime_checkable
class RetrieverHasAudioModality(Protocol):
    """Associated protocol for `AudioRetrieverMixin`."""

    __supports_audio__: bool = True


class AudioRetrieverMixin(ABC):
    """Audio Retriever Mixin.

    Meant to be mixed with a `BaseRetriever` to add audio modality for
    retrieval.
    """

    __supports_audio__ = True

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseRetriever`."""
        super().__init_subclass__()

        if "BaseRetriever" not in [t.__name__ for t in cls.__mro__]:
            raise RetrieverError(
                "`AudioRetrieverMixin` must be mixed with `BaseRetriever`."
            )
