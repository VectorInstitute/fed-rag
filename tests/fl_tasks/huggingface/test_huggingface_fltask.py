"""HuggingFaceFLTask Unit Tests"""

from typing import Callable, OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from torch.nn import Module
from transformers import PreTrainedModel

from fed_rag.exceptions import MissingRequiredNetParam
from fed_rag.fl_tasks.huggingface import (
    BaseFLTaskBundle,
    HuggingFaceFlowerClient,
    HuggingFaceFLTask,
    MissingTrainerSpec,
    _get_weights,
)
from fed_rag.types import TestResult, TrainResult


def test_init_flower_client(
    train_dataset: Dataset,
    val_dataset: Dataset,
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
    hf_pretrained_model: PreTrainedModel,
) -> None:
    bundle = BaseFLTaskBundle(
        net=hf_pretrained_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer=trainer_pretrained_model,
        tester=tester_pretrained_model,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = HuggingFaceFlowerClient(task_bundle=bundle)

    assert client.tester == tester_pretrained_model
    assert client.trainer == trainer_pretrained_model
    assert client.train_dataset == train_dataset
    assert client.val_dataset == val_dataset
    assert client.extra_train_kwargs == {}
    assert client.extra_test_kwargs == {}


def test_flower_client_get_weights(
    train_dataset: Dataset,
    val_dataset: Dataset,
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
    hf_pretrained_model: PreTrainedModel,
) -> None:
    net = hf_pretrained_model
    bundle = BaseFLTaskBundle(
        net=hf_pretrained_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer=trainer_pretrained_model,
        tester=tester_pretrained_model,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = HuggingFaceFlowerClient(task_bundle=bundle)
    expected_weights = [
        val.cpu().numpy() for _, val in net.state_dict().items()
    ]

    assert all((client.get_weights()[0] == expected_weights[0]).flatten())
    assert all((client.get_weights()[1] == expected_weights[1]).flatten())


@patch.object(Module, "load_state_dict")
def test_flower_client_set_weights(
    mock_load_state_dict: MagicMock,
    train_dataset: Dataset,
    val_dataset: Dataset,
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
    hf_pretrained_model: PreTrainedModel,
) -> None:
    net = hf_pretrained_model
    bundle = BaseFLTaskBundle(
        net=net,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer=trainer_pretrained_model,
        tester=tester_pretrained_model,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = HuggingFaceFlowerClient(task_bundle=bundle)
    parameters = client.get_weights()

    # act
    client.set_weights(parameters)

    # assert
    mock_load_state_dict.assert_called_once()
    args, kwargs = mock_load_state_dict.call_args
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    assert (args[0].get("model.weight") == state_dict["model.weight"]).all()
    assert (args[0].get("model.bias") == state_dict["model.bias"]).all()
    assert kwargs == {"strict": True}
    assert client.task_bundle == bundle


def test_flower_client_fit(
    train_dataset: Dataset,
    val_dataset: Dataset,
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
    hf_pretrained_model: PreTrainedModel,
) -> None:
    net = hf_pretrained_model
    bundle = BaseFLTaskBundle(
        net=net,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer=trainer_pretrained_model,
        tester=tester_pretrained_model,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = HuggingFaceFlowerClient(task_bundle=bundle)
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
    assert result[1] == len(client.train_dataset)
    assert result[2] == {"loss": 0.01}


def test_flower_client_evaluate(
    train_dataset: Dataset,
    val_dataset: Dataset,
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
    hf_pretrained_model: PreTrainedModel,
) -> None:
    net = hf_pretrained_model
    bundle = BaseFLTaskBundle(
        net=net,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trainer=trainer_pretrained_model,
        tester=tester_pretrained_model,
        extra_test_kwargs={},
        extra_train_kwargs={},
    )
    client = HuggingFaceFlowerClient(task_bundle=bundle)
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
    assert result[1] == len(client.val_dataset)
    assert result[2] == {"accuracy": 0.88}


def test_init_from_trainer_tester(
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
) -> None:
    fl_task = HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainer_pretrained_model, tester=tester_pretrained_model
    )

    assert fl_task._trainer_spec == getattr(
        trainer_pretrained_model, "__fl_task_trainer_config"
    )
    assert fl_task._tester_spec == getattr(
        tester_pretrained_model, "__fl_task_tester_config"
    )
    assert fl_task._tester == tester_pretrained_model
    assert fl_task._trainer == trainer_pretrained_model
    assert fl_task.training_loop == trainer_pretrained_model


def test_invoking_server_without_strategy_and_net_param_raises(
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
) -> None:
    fl_task = HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainer_pretrained_model, tester=tester_pretrained_model
    )
    with pytest.raises(
        MissingRequiredNetParam,
        match="Please pass in a model using the model param name net.",
    ):
        client_manager = SimpleClientManager()
        fl_task.server(client_manager=client_manager)


def test_invoking_server(
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
) -> None:
    fl_task = HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainer_pretrained_model, tester=tester_pretrained_model
    )
    strategy = FedAvg()
    client_manager = SimpleClientManager()
    server = fl_task.server(strategy=strategy, client_manager=client_manager)

    assert server.client_manager() == client_manager
    assert server.strategy == strategy


def test_invoking_server_using_defaults(
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
    hf_pretrained_model: PreTrainedModel,
) -> None:
    fl_task = HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainer_pretrained_model, tester=tester_pretrained_model
    )
    ndarrays = _get_weights(hf_pretrained_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # act
    server = fl_task.server(net=hf_pretrained_model)

    assert type(server.client_manager()) is SimpleClientManager
    assert type(server.strategy) is FedAvg
    assert server.strategy.initial_parameters == parameters


def test_invoking_client_without_net_param_raises(
    trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
) -> None:
    fl_task = HuggingFaceFLTask.from_trainer_and_tester(
        trainer=trainer_pretrained_model, tester=tester_pretrained_model
    )
    with pytest.raises(
        MissingRequiredNetParam,
        match="Please pass in a model using the model param name net.",
    ):
        fl_task.client()


def test_creating_fl_task_with_undecorated_trainer_raises_error(
    undecorated_trainer_pretrained_model: Callable,
    tester_pretrained_model: Callable,
) -> None:
    with pytest.raises(
        MissingTrainerSpec,
        match="Cannot extract `TrainerSignatureSpec` from supplied `trainer`.",
    ):
        HuggingFaceFLTask.from_trainer_and_tester(
            trainer=undecorated_trainer_pretrained_model,
            tester=tester_pretrained_model,
        )
