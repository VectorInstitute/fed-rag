"""Tester Decorators"""

from typing import Callable


class TesterDecorators:
    def pytorch(self, func: Callable) -> Callable:
        from fed_rag.inspectors.pytorch import inspect_tester_signature

        def decorator(func: Callable) -> Callable:
            # inspect func sig
            spec = inspect_tester_signature(
                func
            )  # may need to create a cfg for this if decorater accepts params

            # store fl_task config
            func.__setattr__("__fl_task_tester_config", spec)  # type: ignore[attr-defined]

            return func

        return decorator(func)

    def huggingface(self, func: Callable) -> Callable:
        from fed_rag.inspectors.huggingface import inspect_tester_signature

        def decorator(func: Callable) -> Callable:
            # inspect func sig
            spec = inspect_tester_signature(
                func
            )  # may need to create a cfg for this if decorater accepts params

            # store fl_task config
            func.__setattr__("__fl_task_tester_config", spec)  # type: ignore[attr-defined]

            return func

        return decorator(func)
