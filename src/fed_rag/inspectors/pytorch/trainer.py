"""PyTorch Trainer Inspector"""

import inspect
from typing import Any, Callable

from fed_rag.exceptions import (
    InvalidReturnType,
    MissingDataParam,
    MissingMultipleDataParams,
    MissingNetParam,
)
from fed_rag.inspectors.common import TrainerSignatureSpec
from fed_rag.types import TrainResult


def inspect_trainer_signature(fn: Callable) -> TrainerSignatureSpec:
    sig = inspect.signature(fn)

    # validate return type
    return_type = sig.return_annotation
    if (return_type is Any) or not issubclass(return_type, TrainResult):
        msg = "Trainer should return a fed_rag.types.TrainResult or a subclsas of it."
        raise InvalidReturnType(msg)

    # inspect fn params
    extra_train_kwargs = []
    net_param = None
    train_data_param = None
    val_data_param = None
    net_parameter_class_name = None

    for name, t in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        if type_name := getattr(t.annotation, "__name__", None):
            if type_name == "Module" and net_param is None:
                net_param = name
                net_parameter_class_name = type_name
                continue

            if type_name == "DataLoader" and train_data_param is None:
                train_data_param = name
                continue

            if type_name == "DataLoader" and val_data_param is None:
                val_data_param = name
                continue

        extra_train_kwargs.append(name)

    if net_param is None:
        msg = (
            "Inspection failed to find a model param. "
            "For PyTorch this param must have type `nn.Module`."
        )
        raise MissingNetParam(msg)

    if train_data_param is None:
        msg = (
            "Inspection failed to find two data params for train and val datasets."
            "For PyTorch these params must be of type `torch.utils.data.DataLoader`"
        )
        raise MissingMultipleDataParams(msg)

    if val_data_param is None:
        msg = (
            "Inspection found one data param but failed to find another. "
            "Two data params are required for train and val datasets."
            "For PyTorch these params must be of type `torch.utils.data.DataLoader`"
        )
        raise MissingDataParam(msg)

    spec = TrainerSignatureSpec(
        net_parameter=net_param,
        train_data_param=train_data_param,
        val_data_param=val_data_param,
        extra_train_kwargs=extra_train_kwargs,
        net_parameter_class_name=net_parameter_class_name,
    )
    return spec
