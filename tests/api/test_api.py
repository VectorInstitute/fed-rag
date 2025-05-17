import importlib

import pytest

from fed_rag import api


@pytest.mark.parametrize("name", api.__all__)
def test_types_all_importable(name: str) -> None:
    """Tests that all names listed in api __all__ are importable."""
    mod = importlib.import_module("fed_rag.api")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None


def test_api_version_exists() -> None:
    """Test that API_VERSION is defined and follows semantic versioning."""
    assert hasattr(api, "API_VERSION")
    version = api.API_VERSION
    assert isinstance(version, str)

    # Check semantic versioning format (major.minor.patch)
    parts = version.split(".")
    assert (
        len(parts) >= 2
    ), "Version should have at least major.minor components"
    assert all(
        part.isdigit() for part in parts
    ), "Version components should be numeric"


def test_get_api_version() -> None:
    assert api.API_VERSION == api.get_api_version()


def test_no_internal_leakage() -> None:
    """Test that internal implementation details aren't exposed in the API."""
    # Check that no private/internal classes are exposed
    for name in api.__all__:
        assert not name.startswith("_"), f"API exposes internal name: {name}"

    # no base classes in api
    assert (
        "BaseBridgeMixin" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseDataCollator" not in api.__all__
    ), "API shouldn't expose base classes"
    assert "BaseFLTask" not in api.__all__, "API shouldn't expose base classes"
    assert (
        "BaseGenerator" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseKnowledgeStore" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseRetriever" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseTokenizer" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseTrainerConfig" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseTrainer" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseGeneratorTrainer" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseRetrieverGenerator" not in api.__all__
    ), "API shouldn't expose base classes"
    assert (
        "BaseRAGTrainerManager" not in api.__all__
    ), "API shouldn't expose base classes"
