"""PyTorchFLTask Unit Tests"""

from typing import Callable

import pytest
import torch
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from fed_rag.exceptions import MissingRequiredNetParam
from fed_rag.fl_tasks.pytorch import (
    BaseFLTaskBundle,
    PyTorchFlowerClient,
    PyTorchFLTask,
)


def test_init_flower_client(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    trainer: Callable,
    tester: Callable,
) -> None:
    bundle = BaseFLTaskBundle(
        net=torch.nn.Linear(2, 1),
        trainloader=train_dataloader,
        valloader=val_dataloader,
        trainer=trainer,
        tester=tester,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = PyTorchFlowerClient(task_bundle=bundle)

    assert client.task_bundle.tester == tester
    assert client.task_bundle.trainer == trainer
    assert client.task_bundle.trainloader == train_dataloader
    assert client.task_bundle.valloader == val_dataloader
    assert client.task_bundle.extra_train_kwargs == {}
    assert client.task_bundle.extra_test_kwargs == {}


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
