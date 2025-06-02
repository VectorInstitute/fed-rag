import asyncio
import concurrent.futures
from typing import Any, Coroutine

from fed_rag.exceptions import FedRAGError


def asyncio_run(coro: Coroutine) -> Any:
    """
    Safely run a coroutine in any async context.

    Gets an existing event loop to run the coroutine if available.
    If there is no existing event loop, creates a new one.
    If an event loop is already running, uses threading to run in a separate thread.

    Args:
        coro: The coroutine to execute

    Returns:
        The result of the coroutine execution

    Raises:
        FedRAGError: If unable to execute the coroutine in any context

    Note:
        Inspired by LlamaIndex's approach to handling nested event loops.
        See: https://github.com/run-llama/llama_index
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # Check if the loop is already running
        if loop.is_running():
            # If loop is already running, run in a separate thread
            def run_coro_in_thread() -> Any:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as executor:
                future = executor.submit(run_coro_in_thread)
                return future.result()
        else:
            # If we're here, there's an existing loop but it's not running
            return loop.run_until_complete(coro)

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise FedRAGError(
                "Unable to execute async operation in current context. "
                "This may be due to nested event loops. Consider using nest_asyncio.apply() "
                "to allow nested event loops, or use async methods directly."
            )
