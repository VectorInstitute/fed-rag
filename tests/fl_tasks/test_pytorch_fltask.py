"""PyTorchFLTask Unit Tests"""

from typing import Callable

from fed_rag.fl_tasks.pytorch import PyTorchFLTask


def test_init_from_trainer_tester(trainer: Callable, tester: Callable) -> None:
    fl_task = PyTorchFLTask.from_trainer_and_tester(
        trainer=trainer, tester=tester
    )

    assert fl_task._trainer_spec == getattr(
        trainer, "__fl_task_trainer_config"
    )
    assert fl_task._tester_spec == getattr(tester, "__fl_task_tester_config")
    assert fl_task._tester == tester
    assert fl_task._trainer == trainer
