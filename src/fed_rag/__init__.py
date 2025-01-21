"""Top-level module"""

from typing import Any

from fed_rag.base.models.base_fl_system import BaseFLSystem


def create_fl_system_from_trainer(trainer: Any) -> BaseFLSystem:
    ...
