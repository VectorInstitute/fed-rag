from types import ModuleType
from unittest.mock import MagicMock, _Call, patch

from fed_rag.base.bridge import BaseBridgeMixin


class _TestBridgeMixin(BaseBridgeMixin):
    _bridge_version = "0.1.0"
    _bridge_extra = "my-bridge"
    _framework = "my-bridge-framework"
    _compatible_versions = ["0.1.x"]


@patch.object(BaseBridgeMixin, "_validate_framework_installed")
def test_bridge_init(mock_validate_framework_installed: MagicMock) -> None:
    _TestBridgeMixin()

    mock_validate_framework_installed.assert_called_once()


@patch.object(BaseBridgeMixin, "_validate_framework_installed")
@patch("fed_rag.base.bridge.importlib")
def test_bridge_get_metadata(
    mock_importlib: MagicMock, mock_validate_framework_installed: MagicMock
) -> None:
    mock_module = ModuleType("my_bridge_framework")
    setattr(mock_module, "__version__", "0.1.1")
    mock_importlib.import_module.return_value = mock_module
    bridge_mixin = (
        _TestBridgeMixin()
    )  # not really supposed to be instantiated on own

    metadata = bridge_mixin.get_bridge_metadata()

    mock_validate_framework_installed.assert_has_calls(
        [_Call(((), {})), _Call(((), {}))]
    )
    mock_importlib.import_module.assert_called_once_with("my_bridge_framework")
    assert metadata["installed_version"] == "0.1.1"
    assert metadata["bridge_version"] == "0.1.0"
    assert metadata["compatible_versions"] == ["0.1.x"]
    assert metadata["framework"] == "my-bridge-framework"
