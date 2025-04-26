"""Exceptions for inspectors."""

from .core import FedRAGError, FedRAGWarning


class InspectorError(FedRAGError):
    """Base inspector error for all inspector-related exceptions."""

    pass


class InspectorWarning(FedRAGWarning):
    """Base inspector warning for all inspector-related warnings."""

    pass


class MissingNetParam(InspectorError):
    pass


class MissingMultipleDataParams(InspectorError):
    pass


class MissingDataParam(InspectorError):
    pass


class MissingTrainerSpec(InspectorError):
    pass


class MissingTesterSpec(InspectorError):
    pass


class UnequalNetParamWarning(InspectorWarning):
    pass


class InvalidReturnType(InspectorError):
    pass
