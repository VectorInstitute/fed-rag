"""Base Bridge"""

import importlib
import importlib.util
from typing import Any, ClassVar, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, model_validator

from fed_rag.exceptions import MissingExtraError


class BridgeMetadata(TypedDict):
    bridge_version: str
    framework: str
    compatible_versions: list[str]
    installed_version: str


class BaseBridgeMixin(BaseModel):
    """Base Bridge Class."""

    # Version of the bridge implementaiton
    _bridge_version: ClassVar[str]
    _bridge_extra: ClassVar[Optional[str | None]]
    _framework: ClassVar[str]
    _compatible_versions: ClassVar[list[str]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_bridge_metadata(self) -> BridgeMetadata:
        metadata: BridgeMetadata = {
            "bridge_version": self._bridge_version,
            "framework": self._framework,
            "compatible_versions": self._compatible_versions,
            "installed_version": self.get_installed_framework_version(),
        }
        return metadata

    @classmethod
    def _validate_framework_installed(cls) -> None:
        if importlib.util.find_spec(cls._framework) is None:
            if cls._bridge_extra:
                missing_package_or_extra = f"fed-rag[{cls._bridge_extra}]"
            else:
                missing_package_or_extra = cls._framework
            raise MissingExtraError(
                f"{cls.__name__} requires the {cls._framework} to be installed. "
                f"To fix please run `pip install {missing_package_or_extra}`."
            )

    @model_validator(mode="before")
    @classmethod
    def check_dependencies(cls, data: Any) -> Any:
        """Validate that qdrant dependencies are installed."""
        cls._validate_framework_installed()
        return data

    def get_installed_framework_version(self) -> str:
        self._validate_framework_installed()
        module = importlib.import_module(self._framework.replace("-", "_"))
        return getattr(module, "__version__", "unknown")
