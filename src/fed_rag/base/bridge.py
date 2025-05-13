"""Base Bridge"""

import importlib
import importlib.util
from typing import Any, ClassVar, Optional, TypedDict

from pydantic import BaseModel, ConfigDict

from fed_rag.exceptions import (
    MissingExtraError,
    MissingSpecifiedConversionMethod,
)


class BridgeMetadata(TypedDict):
    bridge_version: str
    framework: str
    compatible_versions: list[str]
    method_name: str


class BaseBridgeMixin(BaseModel):
    """Base Bridge Class."""

    # Version of the bridge implementaiton
    _bridge_version: ClassVar[str]
    _bridge_extra: ClassVar[Optional[str | None]]
    _framework: ClassVar[str]
    _compatible_versions: ClassVar[list[str]]
    _method_name: ClassVar[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init_subclass__(cls, **kwargs: Any):
        """Register bridge into ~RAGSystem bridge registry."""
        super().__init_subclass__(**kwargs)

        # Register this bridge's metadata to the parent RAGSystem
        for base in cls.__mro__:
            if base.__name__ == "RAGSystem" and hasattr(
                base, "_register_bridge"
            ):
                metadata = cls.get_bridge_metadata()

                # validate method exists
                if not hasattr(cls, metadata["method_name"]):
                    raise MissingSpecifiedConversionMethod(
                        f"Bridge mixin for `{metadata['framework']}` is missing conversion method `{metadata['method_name']}`."
                    )
                base._register_bridge(metadata)
                break

    @classmethod
    def get_bridge_metadata(cls) -> BridgeMetadata:
        metadata: BridgeMetadata = {
            "bridge_version": cls._bridge_version,
            "framework": cls._framework,
            "compatible_versions": cls._compatible_versions,
            "method_name": cls._method_name,
        }
        return metadata

    @classmethod
    def _validate_framework_installed(cls) -> None:
        if importlib.util.find_spec(cls._framework.replace("-", "_")) is None:
            missing_package_or_extra = (
                f"fed-rag[{cls._bridge_extra}]"
                if cls._bridge_extra
                else cls._framework
            )
            msg = (
                f"`{cls._framework}` module is missing but needs to be installed. "
                f"To fix please run `pip install {missing_package_or_extra}`."
            )
            raise MissingExtraError(msg)
