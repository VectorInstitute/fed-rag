"""Exceptions for inspectors."""


class InspectorError(Exception):
    """Base inspector error for all inspector-related exceptions."""

    pass


class InspectorWarning(Warning):
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
