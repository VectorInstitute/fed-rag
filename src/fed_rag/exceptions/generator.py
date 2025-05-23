"""Exceptions for Evals."""

from .core import FedRAGError, FedRAGWarning


class GeneratorError(FedRAGError):
    """Base evals error for all generator-related exceptions."""

    pass


class GeneratorWarning(FedRAGWarning):
    """Base inspector warning for all generator-related warnings."""

    pass
