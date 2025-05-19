import importlib

import pytest

from fed_rag.evals import __all__ as _evals_all
from fed_rag.evals.benchmarks import __all__ as _benchmarks_all


@pytest.mark.parametrize("name", _evals_all)
def test_evals_all_importable(name: str) -> None:
    """Tests that all names listed in evals __all__ are importable."""
    mod = importlib.import_module("fed_rag.evals")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None


@pytest.mark.parametrize("name", _benchmarks_all)
def test_evals_benchmarks_all_importable(name: str) -> None:
    """Tests that all names listed in evals.benchmarks __all__ are importable."""
    mod = importlib.import_module("fed_rag.evals.benchmarks")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
