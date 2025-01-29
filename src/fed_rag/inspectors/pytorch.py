"""PyTorch Inspectors"""

import inspect
from typing import Any, Callable, List, Optional, Union, get_args, get_origin

from fed_rag.trainer_configs import PyTorchTrainerConfig


def _get_param_types(param: inspect.Parameter) -> List[Any]:
    """Extract the types of a parameter. Handles Union and Optional types."""
    typ = param.annotation
    if typ is inspect.Parameter.empty:
        return [Any]
    if get_origin(typ) in (Union, Optional):
        return [t for t in get_args(typ) if t is not type(None)]
    return [typ]


def _inspect_signature(fn: Callable) -> PyTorchTrainerConfig:
    sig = inspect.signature(fn)

    # inspect fn params
    extra_train_kwargs = []
    net_param = None
    train_data_param = None
    val_data_param = None

    for name, t in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        if type_name := getattr(t.annotation, "__name__", None):
            if type_name == "Module" and net_param is None:
                net_param = name
                continue

            if type_name == "DataLoader" and train_data_param is None:
                train_data_param = name
                continue

            if type_name == "DataLoader" and val_data_param is None:
                val_data_param = name
                continue

        extra_train_kwargs.append(name)

    print(f"net_param: {net_param}", flush=True)
    print(f"train_data_param: {train_data_param}", flush=True)
    print(f"val_data_param: {val_data_param}", flush=True)
    print(f"extra_train_kwargs: {extra_train_kwargs}", flush=True)
