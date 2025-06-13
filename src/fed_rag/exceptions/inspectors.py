"""Exceptions for inspectors."""

from .core import FedRAGError, FedRAGWarning


class InspectorError(FedRAGError):
    """Base inspector error for all inspector-related exceptions."""

    pass


class InspectorWarning(FedRAGWarning):
    """Base inspector warning for all inspector-related warnings."""

    pass


class MissingNetParam(InspectorError):
    """Raised if function is missing nn.Module param."""

    pass


class MissingMultipleDataParams(InspectorError):
    """Raised if multiple data params for training, testing and validation are missing."""

    pass


class MissingDataParam(InspectorError):
    """Raised if a single data param is missing."""

    pass


class MissingTrainerSpec(InspectorError):
    """Raised during inspection if trainer is missing `__fl_task_trainer_config` attr."""

    pass


class MissingTesterSpec(InspectorError):
    """Raised during inspection if tester is missing `__fl_task_trainer_config` attr."""

    pass


class UnequalNetParamWarning(InspectorWarning):
    """Thrown if trainer and testers have different parameter names for their nn.Module param."""

    pass


class InvalidReturnType(InspectorError):
    """Raised if the return type of a function is not the expected one."""

    pass
