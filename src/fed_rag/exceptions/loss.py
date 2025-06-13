"""Exceptions for loss."""

from .core import FedRAGError


class LossError(FedRAGError):
    """Base loss errors for all loss-related exceptions."""

    pass


class InvalidReductionParam(LossError):
    """Raised if an invalid aggregation mode is provided."""

    pass
