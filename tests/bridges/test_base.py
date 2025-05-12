import re
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest

from fed_rag.base.bridge import BaseBridgeMixin
from fed_rag.exceptions import MissingExtraError


class _TestBridgeMixin(BaseBridgeMixin):
    _bridge_version = "0.1.0"
    _bridge_extra = "my-bridge"
    _framework = "my-bridge-framework"
    _compatible_versions = ["0.1.x"]
    _method_name = "to_bridge"

    def to_bridge(self) -> None:
        self._validate_framework_installed(self._method_name)
        return None


# overwrite RAGSystem for this test
class RAGSystem(_TestBridgeMixin):
    pass


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


@patch("fed_rag.base.bridge.importlib.util")
def test_validate_framework_installed(mock_importlib_util: MagicMock) -> None:
    mock_importlib_util.find_spec.return_value = None

    # with bridge-extra
    msg = (
        "to_bridge requires the my-bridge-framework to be installed. "
        "To fix please run `pip install fed-rag[my-bridge]`."
    )
    with pytest.raises(MissingExtraError, match=re.escape(msg)):
        rag_system = RAGSystem()
        rag_system.to_bridge()

    # without bridge-extra
    msg = (
        "to_bridge requires the my-bridge-framework to be installed. "
        "To fix please run `pip install my-bridge-framework`."
    )
    with pytest.raises(MissingExtraError, match=re.escape(msg)):
        RAGSystem._bridge_extra = None  # type:ignore [assignment]
        rag_system = RAGSystem()
        rag_system.to_bridge()
