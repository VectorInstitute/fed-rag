import asyncio
import re

import pytest

from fed_rag.exceptions import FedRAGError
from fed_rag.utils.asyncio import asyncio_run


async def simple_async_function(value: int = 42) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.001)
    return value


async def async_function_with_exception() -> None:
    """Async function that raises an exception."""
    await asyncio.sleep(0.001)
    raise RuntimeError("Test exception")


def test_simple_coroutine_execution() -> None:
    """Test running a simple coroutine."""
    result = asyncio_run(simple_async_function(123))
    assert result == 123


def test_coroutine_with_default_args() -> None:
    """Test running a coroutine with default arguments."""
    result = asyncio_run(simple_async_function())
    assert result == 42


def test_existing_but_not_running_loop() -> None:
    """Test behavior with an existing but not running loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Loop exists but is not running
        result = asyncio_run(simple_async_function(789))
        assert result == 789
    finally:
        loop.close()


def test_nested_asyncio_run_calls() -> None:
    """Test that nested calls work correctly."""

    async def outer_async() -> int:
        # This will run in a separate thread due to nested context
        inner_result = asyncio_run(simple_async_function(111))
        return int(inner_result * 2)

    result = asyncio_run(outer_async())
    assert result == 222


def test_coroutine_exception_propagation() -> None:
    """Test that exceptions from coroutines are properly propagated."""
    msg = (
        "Unable to execute async operation in current context. "
        "This may be due to nested event loops. Consider using nest_asyncio.apply() "
        "to allow nested event loops, or use async methods directly."
    )

    with pytest.raises(FedRAGError, match=re.escape(msg)):
        asyncio_run(async_function_with_exception())
