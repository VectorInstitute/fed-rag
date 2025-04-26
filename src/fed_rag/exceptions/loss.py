"""Exceptions for loss."""

from .core import FedRAGError


class LossError(FedRAGError):
    """Base loss errors for all loss-related exceptions."""

    pass


class InvalidReductionParam(LossError):
    pass
