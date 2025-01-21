"""Decorators"""

from functools import wraps
from typing import Any, Callable


def trainer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*func_args: Any, **func_kwargs: Any) -> Any:
        ...

    return wrapper
