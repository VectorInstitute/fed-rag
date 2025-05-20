"""Exceptions for Evals."""

from .core import FedRAGError, FedRAGWarning


class EvalsError(FedRAGError):
    """Base evals error for all evals-related exceptions."""

    pass


class EvalsWarning(FedRAGWarning):
    """Base inspector warning for all evals-related warnings."""

    pass
