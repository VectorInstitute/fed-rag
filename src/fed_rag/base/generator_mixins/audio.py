"""Generator Mixins."""

from typing import Protocol, runtime_checkable

from fed_rag.exceptions.generator import GeneratorError


@runtime_checkable
class GeneratorHasAudioModality(Protocol):
    """Associated protocol for `AudioModalityMixin`."""

    __supports_audio__: bool = True


class AudioModalityMixin:
    """Audio Modality Mixin.

    Meant to be mixed with a `BaseGenerator` to indicate the ability to accept
    audio inputs.
    """

    __supports_audio__ = True

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseGenerator`."""
        super().__init_subclass__()

        if "BaseGenerator" not in [t.__name__ for t in cls.__mro__]:
            raise GeneratorError(
                "`AudioModalityMixin` must be mixed with `BaseGenerator`."
            )
