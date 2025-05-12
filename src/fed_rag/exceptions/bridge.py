"""Exceptions for Bridges."""

from .core import FedRAGError


class BridgeError(FedRAGError):
    """Base bridge error for all fl-task-related exceptions."""

    pass


class MissingSpecifiedConversionMethod(BridgeError):
    pass
