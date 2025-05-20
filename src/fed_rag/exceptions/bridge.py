"""Exceptions for Bridges."""

from .core import FedRAGError


class BridgeError(FedRAGError):
    """Base bridge error for all bridge-related exceptions."""

    pass


class MissingSpecifiedConversionMethod(BridgeError):
    pass
