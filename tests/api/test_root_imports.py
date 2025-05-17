import importlib

import pytest

from fed_rag import __all__ as _root_all


@pytest.mark.parametrize("name", _root_all)
def test_root_names_all_importable(name: str) -> None:
    """Tests that all names listed in root __all__ are importable."""
    mod = importlib.import_module("fed_rag")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None


def test_no_internal_leakage() -> None:
    """Test that internal implementation details aren't exposed in the API."""
    # Check that no private/internal classes are exposed
    for name in _root_all:
        assert not name.startswith("_"), f"API exposes internal name: {name}"

    # no base classes in api
    assert (
        "BaseBridgeMixin" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseDataCollator" not in _root_all
    ), "API shouldn't expose base classes"
    assert "BaseFLTask" not in _root_all, "API shouldn't expose base classes"
    assert (
        "BaseGenerator" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseKnowledgeStore" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseRetriever" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseTokenizer" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseTrainerConfig" not in _root_all
    ), "API shouldn't expose base classes"
    assert "BaseTrainer" not in _root_all, "API shouldn't expose base classes"
    assert (
        "BaseGeneratorTrainer" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseRetrieverGenerator" not in _root_all
    ), "API shouldn't expose base classes"
    assert (
        "BaseRAGTrainerManager" not in _root_all
    ), "API shouldn't expose base classes"
