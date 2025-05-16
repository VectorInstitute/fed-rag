from contextlib import nullcontext as does_not_raise


def test_no_circular_imports() -> None:
    """Test that there are no circular imports in the types module."""
    with does_not_raise():
        pass


def test_root_imports() -> None:
    """Test that core types can be imported from the root."""
    with does_not_raise():
        pass


def test_base_direct_imports() -> None:
    """Test that base classes can be imported directly."""
    with does_not_raise():
        pass
