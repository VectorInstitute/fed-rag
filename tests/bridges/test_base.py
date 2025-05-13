import re
from contextlib import nullcontext as does_not_raise
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fed_rag.base.bridge import BaseBridgeMixin, BridgeMetadata
from fed_rag.exceptions import (
    MissingExtraError,
    MissingSpecifiedConversionMethod,
)


class _TestBridgeMixin(BaseBridgeMixin):
    _bridge_version = "0.1.0"
    _bridge_extra = "my-bridge"
    _framework = "my-bridge-framework"
    _compatible_versions = ["0.1.x"]
    _method_name = "to_bridge"

    def to_bridge(self) -> None:
        self._validate_framework_installed()
        return None


# overwrite RAGSystem for this test
class RAGSystem(_TestBridgeMixin, BaseModel):
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
    assert metadata["compatible_versions"] == ["0.1.x"]
    assert metadata["framework"] == "my-bridge-framework"
    assert metadata["method_name"] == "to_bridge"


def test_rag_system_registry() -> None:
    rag_system = RAGSystem()

    assert _TestBridgeMixin._framework in RAGSystem.bridges

    metadata = rag_system.bridges["my-bridge-framework"]

    assert metadata == _TestBridgeMixin.get_bridge_metadata()


@patch("fed_rag.base.bridge.importlib.util")
def test_validate_framework_installed(mock_importlib_util: MagicMock) -> None:
    mock_importlib_util.find_spec.return_value = None

    # with bridge-extra
    msg = (
        "`my-bridge-framework` module is missing but needs to be installed. "
        "To fix please run `pip install fed-rag[my-bridge]`."
    )
    with pytest.raises(MissingExtraError, match=re.escape(msg)):
        rag_system = RAGSystem()
        rag_system.to_bridge()

    # without bridge-extra
    msg = (
        "`my-bridge-framework` module is missing but needs to be installed. "
        "To fix please run `pip install my-bridge-framework`."
    )
    with pytest.raises(MissingExtraError, match=re.escape(msg)):
        RAGSystem._bridge_extra = None  # type:ignore [assignment]
        rag_system = RAGSystem()
        rag_system.to_bridge()


def test_invalid_mixin_raises_error() -> None:
    msg = "Bridge mixin for `mock` is missing conversion method `missing_method`."
    with pytest.raises(MissingSpecifiedConversionMethod, match=re.escape(msg)):

        class InvalidMixin(BaseBridgeMixin):
            _bridge_version = "0.1.0"
            _bridge_extra = None
            _framework = "mock"
            _compatible_versions = ["0.1.x"]
            _method_name = "missing_method"

        # overwrite RAGSystem for this test
        class RAGSystem(InvalidMixin, BaseModel):
            bridges: ClassVar[dict[str, BridgeMetadata]] = {}

            @classmethod
            def _register_bridge(cls, metadata: BridgeMetadata) -> None:
                """To be used only by `BaseBridgeMixin`."""
                if metadata["framework"] not in cls.bridges:
                    cls.bridges[metadata["framework"]] = metadata
