"""Generator Mixins."""

from typing import Protocol, runtime_checkable

from fed_rag.exceptions.generator import GeneratorError


@runtime_checkable
class GeneratorHasImageModality(Protocol):
    """Associated protocol for `ImageModalityMixin`."""

    __supports_images__: bool = True


class ImageModalityMixin:
    """Image Modality Mixin.

    Meant to be mixed with a `BaseGenerator` to indicate the ability to accept
    image inputs.
    """

    __supports_images__ = True

    def __init_subclass__(cls) -> None:
        """Validate this is mixed with `BaseGenerator`."""
        super().__init_subclass__()

        if "BaseGenerator" not in [t.__name__ for t in cls.__mro__]:
            raise GeneratorError(
                "`ImageModalityMixin` must be mixed with `BaseGenerator`."
            )
