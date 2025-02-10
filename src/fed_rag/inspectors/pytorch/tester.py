"""PyTorch Tester Inspector"""

import inspect
from typing import Any, Callable, List

from pydantic import BaseModel

from fed_rag.exceptions import (
    InvalidReturnType,
    MissingDataParam,
    MissingNetParam,
)
from fed_rag.types import TestResult


class TesterSignatureSpec(BaseModel):
    __test__ = (
        False  # needed for Pytest collision. Avoids PytestCollectionWarning
    )
    net_parameter: str
    test_data_param: str
    extra_test_kwargs: List[str] = []


def inspect_tester_signature(fn: Callable) -> TesterSignatureSpec:
    sig = inspect.signature(fn)

    # validate return type
    return_type = sig.return_annotation
    if (return_type is Any) or not issubclass(return_type, TestResult):
        msg = "Tester should return a fed_rag.types.TestResult or a subclsas of it."
        raise InvalidReturnType(msg)

    # inspect fn params
    extra_tester_kwargs = []
    net_param = None
    test_data_param = None

    for name, t in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        if type_name := getattr(t.annotation, "__name__", None):
            if type_name == "Module" and net_param is None:
                net_param = name
                continue

            if type_name == "DataLoader" and test_data_param is None:
                test_data_param = name
                continue

        extra_tester_kwargs.append(name)

    if net_param is None:
        msg = (
            "Inspection failed to find a model param. "
            "For PyTorch this param must have type `nn.Module`."
        )
        raise MissingNetParam(msg)

    if test_data_param is None:
        msg = (
            "Inspection failed to find a data param for a test dataset."
            "For PyTorch this params must be of type `torch.utils.data.DataLoader`"
        )
        raise MissingDataParam(msg)

    spec = TesterSignatureSpec(
        net_parameter=net_param,
        test_data_param=test_data_param,
        extra_test_kwargs=extra_tester_kwargs,
    )
    return spec
