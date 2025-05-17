import importlib

import pytest

from fed_rag.data_structures import __all__ as _data_structures_all


@pytest.mark.parametrize("name", _data_structures_all)
def test_data_structures_all_importable(name: str) -> None:
    """Tests that all names listed in generators __all__ are importable."""
    mod = importlib.import_module("fed_rag.data_structures")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
