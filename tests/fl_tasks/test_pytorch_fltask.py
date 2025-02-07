"""PyTorchFLTask Unit Tests"""

from typing import Callable

import pytest
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg

from fed_rag.exceptions import MissingRequiredNetParam
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


def test_invoking_server_without_net_param_raises(
    trainer: Callable, tester: Callable
) -> None:
    fl_task = PyTorchFLTask.from_trainer_and_tester(
        trainer=trainer, tester=tester
    )
    with pytest.raises(
        MissingRequiredNetParam,
        match="Please pass in a model using the model param name net.",
    ):
        strategy = FedAvg()
        config = ServerConfig()
        fl_task.server(strategy=strategy, config=config)


def test_invoking_client_without_net_param_raises(
    trainer: Callable, tester: Callable
) -> None:
    fl_task = PyTorchFLTask.from_trainer_and_tester(
        trainer=trainer, tester=tester
    )
    with pytest.raises(
        MissingRequiredNetParam,
        match="Please pass in a model using the model param name net.",
    ):
        fl_task.client()
