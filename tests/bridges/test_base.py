import re
from contextlib import nullcontext as does_not_raise
from importlib.metadata import PackageNotFoundError
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from fed_rag.base.bridge import BaseBridgeMixin, BridgeRegistryMixin
from fed_rag.exceptions import IncompatibleVersionError, MissingExtraError

ST_TYPE = tuple[type[BaseBridgeMixin], type[BridgeRegistryMixin]]


@pytest.fixture(autouse=True)
def setup_and_teardown() -> Iterator[ST_TYPE]:
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
    class _RAGSystem(_TestBridgeMixin, BridgeRegistryMixin, BaseModel):
        pass

    yield _TestBridgeMixin, _RAGSystem

    del BridgeRegistryMixin.bridges[_TestBridgeMixin._framework]


def test_bridge_init(setup_and_teardown: ST_TYPE) -> None:
    _TestBridgeMixin, _ = setup_and_teardown
    with does_not_raise():
        _TestBridgeMixin()


def test_bridge_get_metadata(setup_and_teardown: ST_TYPE) -> None:
    _TestBridgeMixin, _ = setup_and_teardown
    bridge_mixin = (
        _TestBridgeMixin()
    )  # not really supposed to be instantiated on own

    metadata = bridge_mixin.get_bridge_metadata()

    assert metadata["bridge_version"] == "0.1.0"
    assert metadata["compatible_versions"] == {"min": "0.1.0", "max": "0.2.0"}
    assert metadata["framework"] == "my-bridge-framework"
    assert metadata["method_name"] == "to_bridge"


def test_rag_system_registry(setup_and_teardown: ST_TYPE) -> None:
    _TestBridgeMixin, _RAGSystem = setup_and_teardown
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
    setup_and_teardown: ST_TYPE,
) -> None:
    _, _RAGSystem = setup_and_teardown
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
    setup_and_teardown: ST_TYPE,
) -> None:
    _, _RAGSystem = setup_and_teardown
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
def test_validate_framework_success(
    mock_version: MagicMock, setup_and_teardown: ST_TYPE
) -> None:
    _, _RAGSystem = setup_and_teardown
    rag_system = _RAGSystem()
    for ver in ["0.1.0", "0.1.1", "0.2.0b1", "0.2.0rc1", "0.2.0"]:
        mock_version.return_value = ver
        with does_not_raise():
            rag_system.to_bridge()


def test_invalid_mixin_not_registered() -> None:
    class InvalidMixin(BaseBridgeMixin):
        _bridge_version = "0.1.0"
        _bridge_extra = None
        _framework = "mock"
        _compatible_versions = {"min": "0.1.0"}
        _method_name = "missing_method"

    # backup the registry
    _bridges = BridgeRegistryMixin.bridges.copy()
    try:
        # clear the registry for this test
        BridgeRegistryMixin.bridges = {}

        # overwrite RAGSystem for this test
        class _RAGSystem(InvalidMixin, BridgeRegistryMixin, BaseModel):
            pass

        assert InvalidMixin._framework not in BridgeRegistryMixin.bridges
    finally:
        # restore the original registry
        BridgeRegistryMixin.bridges = _bridges
