"""Top-level module"""

from typing import Any, Callable

from fed_rag.base.models.base_fl_system import BaseFLSystem


def fl_task_from_trainloop(train_loop: Callable) -> BaseFLSystem: ...
