import importlib

import pytest

from fed_rag import generators, knowledge_stores, retrievers


@pytest.mark.parametrize("name", generators.__all__)
def test_public_generators_all_importable(name: str) -> None:
    """Tests that all names listed in generators __all__ are importable."""
    mod = importlib.import_module("fed_rag.generators")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None


@pytest.mark.parametrize("name", retrievers.__all__)
def test_public_retrievers_all_importable(name: str) -> None:
    """Tests that all names listed in retrievers __all__ are importable."""
    mod = importlib.import_module("fed_rag.retrievers")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None


@pytest.mark.parametrize("name", knowledge_stores.__all__)
def test_public_knowledge_stores_all_importable(name: str) -> None:
    """Tests that all names listed in knowledge_stores __all__ are importable."""
    mod = importlib.import_module("fed_rag.knowledge_stores")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
