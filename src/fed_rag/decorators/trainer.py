"""Trainer Decorators"""

from typing import Callable


class TrainerDecorators:
    def pytorch(self, func: Callable) -> Callable:
        from fed_rag.inspectors.pytorch import _inspect_signature

        def decorator(func: Callable) -> Callable:
            # inspect func sig
            _sig = _inspect_signature(func)

            # find nn.Model

            # store fl_task config
            func.__setattr__("__fl_task_trainer_config", {})  # type: ignore[attr-defined]

            return func

        return decorator(func)
