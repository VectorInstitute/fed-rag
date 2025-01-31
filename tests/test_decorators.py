"""Decorators unit tests"""

from typing import Any

import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from fed_rag.decorators import federate
from fed_rag.exceptions.inspectors import (
    MissingDataParam,
    MissingMultipleDataParams,
    MissingNetParam,
)
from fed_rag.inspectors.pytorch import (
    TesterSignatureSpec,
    TrainerSignatureSpec,
)


def test_decorated_trainer() -> None:
    def fn(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    decorated = federate.trainer.pytorch(fn)
    config: TrainerSignatureSpec = getattr(
        decorated, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]


def test_decorated_trainer_raises_missing_net_param_error() -> None:
    def fn(
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Any:
        pass

    with pytest.raises(MissingNetParam):
        federate.trainer.pytorch(fn)


def test_decorated_trainer_fails_to_find_two_data_params() -> None:
    def fn(
        model: nn.Module,
    ) -> Any:
        pass

    msg = (
        "Inspection failed to find two data params for train and val datasets."
        "For PyTorch these params must be of type `torch.utils.data.DataLoader`"
    )
    with pytest.raises(MissingMultipleDataParams, match=msg):
        federate.trainer.pytorch(fn)


def test_decorated_trainer_fails_due_to_missing_data_loader() -> None:
    def fn(model: nn.Module, val_loader: DataLoader) -> Any:
        pass

    msg = (
        "Inspection found one data param but failed to find another. "
        "Two data params are required for train and val datasets."
        "For PyTorch these params must be of type `torch.utils.data.DataLoader`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.trainer.pytorch(fn)


def test_decorated_tester() -> None:
    def fn(
        mdl: nn.Module,
        test_loader: DataLoader,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    decorated = federate.tester.pytorch(fn)
    config: TesterSignatureSpec = getattr(decorated, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1", "extra_param_2"]
