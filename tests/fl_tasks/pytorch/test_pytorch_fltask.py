"""PyTorchFLTask Unit Tests"""

from typing import Callable, OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from torch.nn import Module
from torch.utils.data import DataLoader

from fed_rag.exceptions import MissingRequiredNetParam, UnequalNetParamWarning
from fed_rag.fl_tasks.pytorch import (
    BaseFLTaskBundle,
    PyTorchFlowerClient,
    PyTorchFLTask,
    _get_weights,
)
from fed_rag.types import TestResult, TrainResult


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


def test_flower_client_get_weights(
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


@patch.object(Module, "load_state_dict")
def test_flower_client_set_weights(
    mock_load_state_dict: MagicMock,
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
    parameters = client.get_weights()

    # act
    client.set_weights(parameters)

    # assert
    mock_load_state_dict.assert_called_once()
    args, kwargs = mock_load_state_dict.call_args
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    assert (args[0].get("weight") == state_dict["weight"]).all()
    assert kwargs == {"strict": True}


def test_flower_client_fit(
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
    parameters = client.get_weights()
    mock_trainer = MagicMock()
    client.trainer = mock_trainer
    mock_trainer.return_value = TrainResult(loss=0.01)

    # act
    result = client.fit(parameters, config={})

    # assert
    mock_trainer.assert_called_once()
    for a, b in zip(result[0], client.get_weights()):
        assert (a == b).all()
    assert result[1] == len(client.trainloader.dataset)
    assert result[2] == {"loss": 0.01}


def test_flower_client_evaluate(
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
    parameters = client.get_weights()
    mock_tester = MagicMock()
    client.tester = mock_tester
    mock_tester.return_value = TestResult(
        loss=0.01, metrics={"accuracy": 0.88}
    )

    # act
    result = client.evaluate(parameters=parameters, config={})

    # assert
    mock_tester.assert_called_once()
    assert result[0] == 0.01
    assert result[1] == len(client.valloader.dataset)
    assert result[2] == {"accuracy": 0.88}


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


def test_invoking_server_without_strategy_and_net_param_raises(
    trainer: Callable, tester: Callable
) -> None:
    fl_task = PyTorchFLTask.from_trainer_and_tester(
        trainer=trainer, tester=tester
    )
    with pytest.raises(
        MissingRequiredNetParam,
        match="Please pass in a model using the model param name net.",
    ):
        client_manager = SimpleClientManager()
        fl_task.server(client_manager=client_manager)


def test_invoking_server(trainer: Callable, tester: Callable) -> None:
    fl_task = PyTorchFLTask.from_trainer_and_tester(
        trainer=trainer, tester=tester
    )
    strategy = FedAvg()
    client_manager = SimpleClientManager()
    server = fl_task.server(strategy=strategy, client_manager=client_manager)

    assert server.client_manager() == client_manager
    assert server.strategy == strategy


def test_invoking_server_using_defaults(
    trainer: Callable, tester: Callable
) -> None:
    fl_task = PyTorchFLTask.from_trainer_and_tester(
        trainer=trainer, tester=tester
    )
    net = torch.nn.Linear(2, 1)
    ndarrays = _get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # act
    server = fl_task.server(net=net)

    assert type(server.client_manager()) is SimpleClientManager
    assert type(server.strategy) is FedAvg
    assert server.strategy.initial_parameters == parameters


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


def test_creating_fl_task_with_mismatched_net_params_raises_warning(
    trainer: Callable,
    mismatch_tester: Callable,
) -> None:
    msg = (
        "`trainer`'s model parameter name is not the same as that for `tester`. "
        "Will use the name supplied in `trainer`."
    )
    with pytest.warns(UnequalNetParamWarning, match=msg):
        PyTorchFLTask.from_trainer_and_tester(
            trainer=trainer,
            tester=mismatch_tester,
        )


def test_init_fl_task_with_mismatched_net_params_raises_warning(
    trainer: Callable,
    mismatch_tester: Callable,
) -> None:
    msg = (
        "`trainer`'s model parameter name is not the same as that for `tester`. "
        "Will use the name supplied in `trainer`."
    )
    with pytest.warns(UnequalNetParamWarning, match=msg):
        PyTorchFLTask(
            trainer=trainer,
            trainer_spec=trainer.__fl_task_trainer_config,  # type: ignore[attr-defined]
            tester=mismatch_tester,
            tester_spec=mismatch_tester.__fl_task_tester_config,  # type: ignore[attr-defined]
        )
