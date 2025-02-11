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

    assert client.tester == tester
    assert client.trainer == trainer
    assert client.trainloader == train_dataloader
    assert client.valloader == val_dataloader
    assert client.extra_train_kwargs == {}
    assert client.extra_test_kwargs == {}


def test_init_flower_client_get_weights(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    trainer: Callable,
    tester: Callable,
) -> None:
    net = torch.nn.Linear(2, 1)
    bundle = BaseFLTaskBundle(
        net=net,
        trainloader=train_dataloader,
        valloader=val_dataloader,
        trainer=trainer,
        tester=tester,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = PyTorchFlowerClient(task_bundle=bundle)
    expected_weights = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]

    assert all((client.get_weights()[0] == expected_weights[0]).flatten())
    assert all((client.get_weights()[1] == expected_weights[1]).flatten())


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
