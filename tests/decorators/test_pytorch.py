"""Decorators unit tests"""

from typing import Any

import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from fed_rag.decorators import federate
from fed_rag.exceptions.inspectors import (
    InvalidReturnType,
    MissingDataParam,
    MissingMultipleDataParams,
    MissingNetParam,
)
from fed_rag.inspectors.pytorch import (
    TesterSignatureSpec,
    TrainerSignatureSpec,
)
from fed_rag.types import TestResult, TrainResult


def test_decorated_trainer() -> None:
    def fn(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TrainResult:
        pass

    decorated = federate.trainer.pytorch(fn)
    config: TrainerSignatureSpec = getattr(
        decorated, "__fl_task_trainer_config"
    )
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "Module"


def test_decorated_trainer_raises_invalid_return_type_error() -> None:
    def fn(
        net: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Trainer should return a fed_rag.types.TrainResult or a subclsas of it.",
    ):
        federate.trainer.pytorch(fn)


def test_decorated_trainer_raises_missing_net_param_error() -> None:
    def fn(
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainResult:
        pass

    with pytest.raises(MissingNetParam):
        federate.trainer.pytorch(fn)


def test_decorated_trainer_fails_to_find_two_data_params() -> None:
    def fn(
        model: nn.Module,
    ) -> TrainResult:
        pass

    msg = (
        "Inspection failed to find two data params for train and val datasets."
        "For PyTorch these params must be of type `torch.utils.data.DataLoader`"
    )
    with pytest.raises(MissingMultipleDataParams, match=msg):
        federate.trainer.pytorch(fn)


def test_decorated_trainer_fails_due_to_missing_data_loader() -> None:
    def fn(model: nn.Module, val_loader: DataLoader) -> TrainResult:
        pass

    msg = (
        "Inspection found one data param but failed to find another. "
        "Two data params are required for train and val datasets."
        "For PyTorch these params must be of type `torch.utils.data.DataLoader`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.trainer.pytorch(fn)


def test_decorated_trainer_from_instance_method() -> None:
    class _TestClass:
        @federate.trainer.pytorch
        def fn(
            self,
            net: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(obj.fn, "__fl_task_trainer_config")
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "Module"


def test_decorated_trainer_from_class_method() -> None:
    class _TestClass:
        @classmethod
        @federate.trainer.pytorch
        def fn(
            cls,
            net: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            extra_param_1: int,
            extra_param_2: float | None,
        ) -> TrainResult:
            pass

    obj = _TestClass()
    config: TrainerSignatureSpec = getattr(obj.fn, "__fl_task_trainer_config")
    assert config.net_parameter == "net"
    assert config.train_data_param == "train_loader"
    assert config.val_data_param == "val_loader"
    assert config.extra_train_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "Module"


def test_decorated_tester() -> None:
    def fn(
        mdl: nn.Module,
        test_loader: DataLoader,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> TestResult:
        pass

    decorated = federate.tester.pytorch(fn)
    config: TesterSignatureSpec = getattr(decorated, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1", "extra_param_2"]
    assert config.net_parameter_class_name == "Module"


def test_decorated_tester_raises_invalid_return_type() -> None:
    def fn(
        mdl: nn.Module,
        test_loader: DataLoader,
        extra_param_1: int,
        extra_param_2: float | None,
    ) -> Any:
        pass

    with pytest.raises(
        InvalidReturnType,
        match="Tester should return a fed_rag.types.TestResult or a subclsas of it.",
    ):
        federate.tester.pytorch(fn)


def test_decorated_tester_raises_missing_net_param_error() -> None:
    def fn(
        test_loader: DataLoader,
    ) -> TestResult:
        pass

    with pytest.raises(MissingNetParam):
        federate.tester.pytorch(fn)


def test_decorated_tester_fails_to_a_data_params() -> None:
    def fn(
        model: nn.Module,
    ) -> TestResult:
        pass

    msg = (
        "Inspection failed to find a data param for a test dataset."
        "For PyTorch this params must be of type `torch.utils.data.DataLoader`"
    )
    with pytest.raises(MissingDataParam, match=msg):
        federate.tester.pytorch(fn)


def test_decorated_tester_from_instance_method() -> None:
    class _TestClass:
        @federate.tester.pytorch
        def fn(
            self,
            mdl: nn.Module,
            test_loader: DataLoader,
            extra_param_1: int,
        ) -> TestResult:
            pass

    obj = _TestClass()
    config: TesterSignatureSpec = getattr(obj.fn, "__fl_task_tester_config")
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "Module"


def test_decorated_tester_from_class_method() -> None:
    class _TestClass:
        @classmethod
        @federate.tester.pytorch
        def fn(
            cls,
            mdl: nn.Module,
            test_loader: DataLoader,
            extra_param_1: int,
        ) -> TestResult:
            pass

    config: TesterSignatureSpec = getattr(
        _TestClass.fn, "__fl_task_tester_config"
    )
    assert config.net_parameter == "mdl"
    assert config.test_data_param == "test_loader"
    assert config.extra_test_kwargs == ["extra_param_1"]
    assert config.net_parameter_class_name == "Module"
