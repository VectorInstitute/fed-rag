"""Base Bridge"""

import importlib.metadata
from typing import ClassVar, Optional

from packaging.version import Version
from pydantic import BaseModel, ConfigDict

from fed_rag.data_structures import BridgeMetadata, CompatibleVersions
from fed_rag.exceptions import IncompatibleVersionError, MissingExtraError


class BridgeRegistryMixin:
    """Mixin to manage bridge registration."""

    bridges: ClassVar[dict[str, BridgeMetadata]] = {}

    @classmethod
    def _register_bridge(cls, metadata: BridgeMetadata) -> None:
        """Register a bridge's metadata."""
        if metadata["framework"] not in cls.bridges:
            cls.bridges[metadata["framework"]] = metadata

    def __init_subclass__(cls) -> None:
        """Register bridges on subclass creation."""
        super().__init_subclass__()
        for base in cls.__mro__:
            if issubclass(base, BaseBridgeMixin) and hasattr(
                base, "_method_name"
            ):
                metadata = base.get_bridge_metadata()
                if hasattr(base, metadata["method_name"]):
                    cls._register_bridge(metadata)


class BaseBridgeMixin(BaseModel):
    """Base Bridge Class."""

    # Version of the bridge implementation.
    _bridge_version: ClassVar[str]
    _bridge_extra: ClassVar[Optional[str | None]]
    _framework: ClassVar[str]
    _compatible_versions: ClassVar[CompatibleVersions]
    _method_name: ClassVar[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
        """Check if the framework is installed and compatible."""
        try:
            installed_version = Version(
                importlib.metadata.version(cls._framework.replace("-", "_"))
            )
            cls._check_bound("min", installed_version)
            cls._check_bound("max", installed_version)
        except importlib.metadata.PackageNotFoundError:
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

    @classmethod
    def _check_bound(
        cls,
        bound_name: str,
        installed: Version,
    ) -> None:
        """Check one side of compatibility and raise if it fails."""
        if bound_name not in cls._compatible_versions:
            return

        req = Version(cls._compatible_versions[bound_name])
        if (bound_name == "min" and installed < req) or (
            bound_name == "max" and installed > req
        ):
            direction = "Minimum" if bound_name == "min" else "Maximum"
            raise IncompatibleVersionError(
                f"`{cls._framework}` version {installed} is incompatible. "
                f"{direction} required is {req}."
            )
