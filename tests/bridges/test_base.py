import re
from contextlib import nullcontext as does_not_raise
from importlib.metadata import PackageNotFoundError
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fed_rag.base.bridge import BaseBridgeMixin, BridgeMetadata
from fed_rag.exceptions import (
    IncompatibleVersionError,
    MissingExtraError,
    MissingSpecifiedConversionMethod,
)


class _TestBridgeMixin(BaseBridgeMixin):
    _bridge_version = "0.1.0"
    _bridge_extra = "my-bridge"
    _framework = "my-bridge-framework"
    _compatible_versions = {"min": "0.1.0", "max": "0.2.0"}
    _method_name = "to_bridge"

    def to_bridge(self) -> None:
        self._validate_framework_installed()
        return None


# overwrite RAGSystem for this test
class _RAGSystem(_TestBridgeMixin, BaseModel):
    bridges: ClassVar[dict[str, BridgeMetadata]] = {}

    @classmethod
    def _register_bridge(cls, metadata: BridgeMetadata) -> None:
        """To be used only by `BaseBridgeMixin`."""
        if metadata["framework"] not in cls.bridges:
            cls.bridges[metadata["framework"]] = metadata


def test_bridge_init() -> None:
    with does_not_raise():
        _TestBridgeMixin()


def test_bridge_get_metadata() -> None:
    bridge_mixin = (
        _TestBridgeMixin()
    )  # not really supposed to be instantiated on own

    metadata = bridge_mixin.get_bridge_metadata()

    assert metadata["bridge_version"] == "0.1.0"
    assert metadata["compatible_versions"] == {"min": "0.1.0", "max": "0.2.0"}
    assert metadata["framework"] == "my-bridge-framework"
    assert metadata["method_name"] == "to_bridge"


def test_rag_system_registry() -> None:
    rag_system = _RAGSystem()

    assert _TestBridgeMixin._framework in _RAGSystem.bridges

    metadata = rag_system.bridges["my-bridge-framework"]

    assert metadata == _TestBridgeMixin.get_bridge_metadata()


@patch(
    "fed_rag.base.bridge.importlib.metadata.version",
    side_effect=PackageNotFoundError(),
)
def test_validate_framework_not_installed_error(
    mock_version: MagicMock,
) -> None:
    # with bridge-extra
    msg = (
        "`my-bridge-framework` module is missing but needs to be installed. "
        "To fix please run `pip install fed-rag[my-bridge]`."
    )
    with pytest.raises(MissingExtraError, match=re.escape(msg)):
        rag_system = _RAGSystem()
        rag_system.to_bridge()

    # without bridge-extra
    msg = (
        "`my-bridge-framework` module is missing but needs to be installed. "
        "To fix please run `pip install my-bridge-framework`."
    )
    with pytest.raises(MissingExtraError, match=re.escape(msg)):
        _RAGSystem._bridge_extra = None  # type:ignore [assignment]
        rag_system = _RAGSystem()
        rag_system.to_bridge()


@patch("fed_rag.base.bridge.importlib.metadata.version")
def test_validate_framework_incompatible_error(
    mock_version: MagicMock,
) -> None:
    rag_system = _RAGSystem()
    # too low versions
    for ver in ["0.0.9", "0.0.10", "0.1.0b1", "0.1.0rc1"]:
        mock_version.return_value = ver
        msg = (
            f"`my-bridge-framework` version {ver} is incompatible. "
            "Minimum required is 0.1.0."
        )
        with pytest.raises(IncompatibleVersionError, match=re.escape(msg)):
            rag_system.to_bridge()

    # too high versions
    for ver in ["0.2.1", "0.3.0", "1.0.0"]:
        mock_version.return_value = ver
        msg = (
            f"`my-bridge-framework` version {ver} is incompatible. "
            "Maximum required is 0.2.0."
        )
        with pytest.raises(IncompatibleVersionError, match=re.escape(msg)):
            rag_system.to_bridge()


@patch("fed_rag.base.bridge.importlib.metadata.version")
def test_validate_framework_success(mock_version: MagicMock) -> None:
    rag_system = _RAGSystem()
    for ver in ["0.1.0", "0.1.1", "0.2.0b1", "0.2.0rc1", "0.2.0"]:
        mock_version.return_value = ver
        with does_not_raise():
            rag_system.to_bridge()


def test_invalid_mixin_raises_error() -> None:
    msg = "Bridge mixin for `mock` is missing conversion method `missing_method`."
    with pytest.raises(MissingSpecifiedConversionMethod, match=re.escape(msg)):

        class InvalidMixin(BaseBridgeMixin):
            _bridge_version = "0.1.0"
            _bridge_extra = None
            _framework = "mock"
            _compatible_versions = {"min": "0.1.0"}
            _method_name = "missing_method"

        # overwrite RAGSystem for this test
        class _RAGSystem(InvalidMixin, BaseModel):
            bridges: ClassVar[dict[str, BridgeMetadata]] = {}

            @classmethod
            def _register_bridge(cls, metadata: BridgeMetadata) -> None:
                """To be used only by `BaseBridgeMixin`."""
                if metadata["framework"] not in cls.bridges:
                    cls.bridges[metadata["framework"]] = metadata
