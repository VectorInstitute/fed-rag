"""Decorators"""

from typing import Callable


class TrainerDecorators:
    def pytorch(self, func: Callable) -> Callable:
        def decorator(func: Callable) -> Callable:
            # inspect func sig

            # find nn.Model

            # store fl_task config
            func.__setattr__("__fl_task_config", {})  # type: ignore[attr-defined]

            return func

        return decorator(func)


federate = TrainerDecorators()
