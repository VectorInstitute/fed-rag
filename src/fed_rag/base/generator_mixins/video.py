"""Generator Mixins."""

from typing import Protocol, runtime_checkable

from fed_rag.exceptions.generator import GeneratorError


@runtime_checkable
class GeneratorHasVideoModality(Protocol):
    """Associated protocol for `VideoModalityMixin`."""

    __supports_video__: bool = True


class VideoModalityMixin:
    """Video Modality Mixin.

    Meant to be mixed with a `BaseGenerator` to indicate the ability to accept
    video inputs.
    """

    __supports_video__ = True

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseGenerator`."""
        super().__init_subclass__()

        if "BaseGenerator" not in [t.__name__ for t in cls.__mro__]:
            raise GeneratorError(
                "`VideoModalityMixin` must be mixed with `BaseGenerator`."
            )
