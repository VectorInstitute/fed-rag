"""Decorators"""

from functools import wraps
from typing import Any, Callable


class TrainerDecorators:

    def pytorch(func: Callable) -> Callable:
        @wraps(func)
        def decorator(*func_args: Any, **func_kwargs: Any) -> Any:
            # inspect func sig

            # find nn.Model

            # store fl_task config
            func.__fl_task_config = ...

        return decorator


federate = TrainerDecorators()
