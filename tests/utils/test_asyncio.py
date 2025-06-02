import asyncio

from fed_rag.utils.asyncio import asyncio_run


async def simple_async_function(value: int = 42) -> int:
    """Simple async function for testing."""
    await asyncio.sleep(0.001)
    return value


async def async_function_with_exception() -> None:
    """Async function that raises an exception."""
    await asyncio.sleep(0.001)
    raise ValueError("Test exception")


def test_simple_coroutine_execution() -> None:
    """Test running a simple coroutine."""
    result = asyncio_run(simple_async_function(123))
    assert result == 123


def test_coroutine_with_default_args() -> None:
    """Test running a coroutine with default arguments."""
    result = asyncio_run(simple_async_function())
    assert result == 42
