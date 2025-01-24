"""Models"""

from typing import Any, Protocol

from fed_rag.base.models.base_fl_task import BaseFLTask


class TrainCallback(Protocol):
    def __call__(self, train_data: Any, val_data: Any) -> dict[str, float]:
        ...


def fl_task_from_trainer(trainer: TrainCallback) -> BaseFLTask:
    ...
